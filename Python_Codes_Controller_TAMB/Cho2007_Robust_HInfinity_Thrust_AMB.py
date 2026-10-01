#!/usr/bin/env python3
"""Robust mixed-sensitivity H-infinity control of a Cho et al. thrust AMB.

Reference
---------
Y. M. Cho, S. Srinivasan, J.-H. Oh and H. S. Kim, "Modelling and system
identification of active magnetic bearing systems", Mathematical and Computer
Modelling of Dynamical Systems, 13(2), 125-142, 2007.

Cho et al. give the bias-linearized actuator and PWM transconductance amplifier
in their Eqs. (13)-(14). With the user-requested reduced axial model, the plant
implemented here is exactly

                  k_i K_amp
 G(s)=X/V_sp = --------------------------- .
                (m s^2-k_x)(tau s+1)

The state realization is
    x_dot = v
    m v_dot = k_x*x + k_i*i_c + F_disturbance
    tau i_c_dot = -i_c + K_amp*V_sp.

IMPORTANT SIGN CONVENTION: ``k_x`` below is the positive magnitude of the
destabilizing (negative mechanical stiffness) term in ``m*x_ddot-k_x*x``.
Cho et al. describe their actuator coefficient k_x as signed and negative.
Convert an experimentally reported signed coefficient to the convention used
here before entering it.

The script contains an automatic RK4-versus-analytical-transfer-function test,
a noisy filtered sensor, digital delay, PWM/voltage/current limits, anti-windup,
a robust digital controller, explicit disturbance injection, plots and
validation metrics. It is a small-signal bias-linearized model; x is axial
displacement from the nominal gap, not the absolute gap itself.

Dependencies: numpy, scipy, matplotlib
Run: python Cho2007_Robust_HInfinity_Thrust_AMB.py
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, signal


@dataclass
class AMBPlantParameters:
    """All replaceable physical and implementation parameters in SI units."""

    # Bias-linearized axial plant: m*x_ddot-kx*x=ki*ic+F_d.
    rotor_mass: float = 0.42                    # m [kg]
    negative_stiffness_magnitude: float = 120.0 # k_x [N/m], enter POSITIVE magnitude
    current_force_gain: float = 3.8             # k_i [N/A]
    amplifier_gain: float = 0.45                # K_amp [A/V]
    amplifier_time_constant: float = 0.006      # tau [s]
    nominal_air_gap: float = 0.014              # [m]
    bias_current: float = 2.20                   # physical DC bias [A]

    # Constraints. ic is incremental control current about bias_current.
    control_current_min: float = -1.80           # [A]
    control_current_max: float = 1.80            # [A]
    voltage_setpoint_limit: float = 10.0         # |V_sp| [V]
    dc_bus_voltage: float = 48.0                 # PWM DC link [V]
    pwm_duty_limit: float = 0.95                 # absolute duty ratio
    amplifier_efficiency: float = 0.97

    # Position safety region expressed about nominal_air_gap.
    minimum_air_gap: float = 0.010               # [m]
    maximum_air_gap: float = 0.018               # [m]
    initial_displacement: float = 0.25e-3        # x(0) [m]
    initial_velocity: float = 0.0                # [m/s]

    def effective_voltage_limit(self) -> float:
        return min(self.voltage_setpoint_limit,
                   self.dc_bus_voltage*self.pwm_duty_limit)

    def analytical_tf(self) -> Tuple[np.ndarray, np.ndarray]:
        """Numerator/denominator of the exact requested G(s)."""
        m, kx, tau = (self.rotor_mass, self.negative_stiffness_magnitude,
                      self.amplifier_time_constant)
        numerator = np.array([self.current_force_gain*self.amplifier_gain])
        # (m*s^2-kx)*(tau*s+1)
        denominator = np.array([m*tau, m, -kx*tau, -kx])
        return numerator, denominator


@dataclass
class SensorParameters:
    sensitivity: float = 800.0       # [V/m]
    bias_voltage: float = 0.0008     # residual uncompensated offset [V]
    noise_variance: float = 4.0e-7   # voltage-noise variance [V^2]
    lowpass_cutoff_hz: float = 100.0 # 2nd-order Butterworth anti-alias filter


@dataclass
@dataclass
class HInfinityParameters:
    """Digital execution, uncertainty and mixed-sensitivity specifications."""
    sample_time: float = 0.0005
    computational_delay_samples: int = 1
    # Relative uncertainty half-widths used to build all 2^4 plant vertices.
    mass_uncertainty: float = 0.15
    stiffness_uncertainty: float = 0.20
    gain_uncertainty: float = 0.15
    time_constant_uncertainty: float = 0.25
    # Weight shapes. An overall normalization sets the performance-unit scale.
    weight_normalization: float = 0.03
    performance_bandwidth_rad_s: float = 12.0
    performance_Ms: float = 2.0
    performance_epsilon: float = 0.20
    control_corner_rad_s: float = 150.0
    control_low_gain: float = 0.02
    control_high_gain: float = 0.01
    uncertainty_corner_rad_s: float = 250.0
    uncertainty_low_gain: float = 0.03
    uncertainty_high_gain: float = 1.5
    # Fixed-order proper controller K=(Kp+Ki/s+Kd*s)/(1+s/wd).
    kp_bounds: Tuple[float,float] = (450.0,900.0)
    ki_bounds: Tuple[float,float] = (1000.0,3500.0)
    kd_bounds: Tuple[float,float] = (15.0,50.0)
    derivative_rolloff_bounds: Tuple[float,float] = (150.0,800.0)
    frequency_min_rad_s: float = 0.1
    frequency_max_rad_s: float = 1.0e4
    frequency_points: int = 500
    optimizer_iterations: int = 45
    optimizer_population: int = 8
    optimizer_seed: int = 17
    stability_margin_rad_s: float = 0.05


class ThrustAMBPlant:
    """Saturated RK4 realization of the requested third-order transfer model."""

    def __init__(self, p: AMBPlantParameters):
        self.p = p
        self.state = np.array([p.initial_displacement, p.initial_velocity, 0.0])
        self.voltage_applied = 0.0
        self.duty_ratio = 0.0

    def _limited_voltage(self, voltage_demand: float) -> float:
        p = self.p
        voltage = float(np.clip(voltage_demand, -p.effective_voltage_limit(),
                                p.effective_voltage_limit()))
        self.duty_ratio = voltage/p.dc_bus_voltage
        return p.amplifier_efficiency*voltage

    def derivative(self, state: np.ndarray, voltage_demand: float,
                   disturbance_force: float, apply_limits: bool = True) -> np.ndarray:
        p = self.p
        x, velocity, incremental_current = state
        voltage = self._limited_voltage(voltage_demand) if apply_limits else voltage_demand
        current_target = p.amplifier_gain*voltage
        if apply_limits:
            current_target = float(np.clip(current_target,p.control_current_min,
                                           p.control_current_max))
        current_dot = (current_target-incremental_current)/p.amplifier_time_constant
        # Stop integration from driving an already-saturated current farther out.
        if apply_limits and ((incremental_current <= p.control_current_min and current_dot < 0)
                             or (incremental_current >= p.control_current_max and current_dot > 0)):
            current_dot = 0.0
        acceleration = (p.negative_stiffness_magnitude*x
                        +p.current_force_gain*incremental_current
                        +disturbance_force)/p.rotor_mass
        return np.array([velocity,acceleration,current_dot])

    def step(self, voltage_demand: float, disturbance_force: float,
             dt: float, substeps: int = 4) -> None:
        """Classical fixed-step RK4 within one digital-control interval."""
        h = dt/substeps
        for _ in range(substeps):
            s = self.state
            k1 = self.derivative(s,voltage_demand,disturbance_force)
            k2 = self.derivative(s+0.5*h*k1,voltage_demand,disturbance_force)
            k3 = self.derivative(s+0.5*h*k2,voltage_demand,disturbance_force)
            k4 = self.derivative(s+h*k3,voltage_demand,disturbance_force)
            self.state = s+h*(k1+2*k2+2*k3+k4)/6.0
        self.state[2] = np.clip(self.state[2],self.p.control_current_min,
                                self.p.control_current_max)
        self.voltage_applied = self._limited_voltage(voltage_demand)


class SensorFilter:
    """Sensitivity, residual bias, Gaussian noise and Butterworth filtering."""

    def __init__(self, p: SensorParameters, sample_time: float,
                 initial_displacement: float, seed: int = 21):
        self.p = p
        self.rng = np.random.default_rng(seed)
        k = np.tan(np.pi*p.lowpass_cutoff_hz*sample_time)
        norm = 1.0/(1.0+np.sqrt(2.0)*k+k*k)
        self.b = np.array([k*k*norm,2*k*k*norm,k*k*norm])
        self.a = np.array([1.0,2*(k*k-1)*norm,
                           (1-np.sqrt(2.0)*k+k*k)*norm])
        initial_v = p.sensitivity*initial_displacement+p.bias_voltage
        self.input_history = np.array([initial_v,initial_v])
        self.output_history = np.array([initial_v,initial_v])

    def measure(self, displacement: float) -> float:
        sensor_voltage = (self.p.sensitivity*displacement+self.p.bias_voltage
                          +self.rng.normal(0.0,np.sqrt(self.p.noise_variance)))
        y = (self.b[0]*sensor_voltage+self.b[1]*self.input_history[0]
             +self.b[2]*self.input_history[1]-self.a[1]*self.output_history[0]
             -self.a[2]*self.output_history[1])
        self.input_history[1],self.input_history[0] = self.input_history[0],sensor_voltage
        self.output_history[1],self.output_history[0] = self.output_history[0],y
        # Divide by sensitivity; residual bias remains visible to the controller.
        return y/self.p.sensitivity


class HInfinityController:
    """Robust fixed-structure mixed-sensitivity H-infinity synthesis.

    For negative feedback, L=G*K, S=(1+L)^-1 and T=L*(1+L)^-1.
    The generalized plant performance channel is

        z = [ Wp*S ; Wu*K*S ; Wt*T ] w,

    where w represents normalized exogenous reference/disturbance signals.
    Thus the SISO structured singular-value objective is

        gamma = sup_w sqrt(|Wp*S|^2+|Wu*K*S|^2+|Wt*T|^2).

    This script solves the fixed-order nonsmooth H-infinity problem directly
    on a dense logarithmic frequency grid. Every candidate is first required
    to make the exact closed-loop characteristic polynomial Hurwitz for every
    vertex of the configured uncertainty box. This gives a reproducible
    robust-stability certificate over the tested vertices (not a continuous
    mu-analysis certificate between vertices).

    The optimized proper controller is converted with zero-order hold to a
    discrete state-space realization (Ad,Bd,Cd,Dd) for online execution.
    """
    def __init__(self,hp: HInfinityParameters,pp: AMBPlantParameters,
                 synthesize: bool=True):
        self.h,self.p=hp,pp
        self.omega=np.logspace(np.log10(hp.frequency_min_rad_s),
                               np.log10(hp.frequency_max_rad_s),hp.frequency_points)
        self.vertices=self._uncertainty_vertices()
        self.weights=self._weight_responses(self.omega)
        self.theta=self._synthesize() if synthesize else np.array([500.,1800.,20.,300.])
        self.k_num,self.k_den=self.controller_polynomials(self.theta)
        Ac,Bc,Cc,Dc=signal.tf2ss(self.k_num,self.k_den)
        self.Ad,self.Bd,self.Cd,self.Dd,_=signal.cont2discrete(
            (Ac,Bc,Cc,Dc),hp.sample_time,method='zoh')
        self.state=np.zeros(self.Ad.shape[0])
        self.gamma=self.robust_hinf_objective(self.theta,penalize_instability=False)

    def _uncertainty_vertices(self):
        p,h=self.p,self.h; vertices=[]
        for fm in (1-h.mass_uncertainty,1+h.mass_uncertainty):
            for fk in (1-h.stiffness_uncertainty,1+h.stiffness_uncertainty):
                for fg in (1-h.gain_uncertainty,1+h.gain_uncertainty):
                    for ft in (1-h.time_constant_uncertainty,1+h.time_constant_uncertainty):
                        vertices.append((p.rotor_mass*fm,
                            p.negative_stiffness_magnitude*fk,
                            p.current_force_gain*p.amplifier_gain*fg,
                            p.amplifier_time_constant*ft))
        return vertices

    def _weight_responses(self,w: np.ndarray):
        """Frequency responses of Wp, Wu and multiplicative-uncertainty Wt."""
        h=self.h; s=1j*w; q=h.weight_normalization
        # Wp=(s/Ms+wb)/(s+wb*epsilon): large at low frequency.
        wp=q*(s/h.performance_Ms+h.performance_bandwidth_rad_s)/(
            s+h.performance_bandwidth_rad_s*h.performance_epsilon)
        # Wu transitions from a small low-frequency penalty to its HF value.
        wu=q*h.control_high_gain*(s/h.control_corner_rad_s+h.control_low_gain)/(
            s/h.control_corner_rad_s+1.0)
        # Wt increases with frequency to bound neglected flexible/PWM dynamics.
        wt=q*(h.uncertainty_low_gain+s/h.uncertainty_corner_rad_s)/(
            1.0+s/(h.uncertainty_high_gain*h.uncertainty_corner_rad_s))
        return wp,wu,wt

    @staticmethod
    def controller_polynomials(theta: np.ndarray):
        kp,ki,kd,wd=theta
        # (Kp+Ki/s+Kd*s)/(1+s/wd), written as descending polynomials.
        numerator=np.array([kp/wd+kd,kp+ki/wd,ki])
        denominator=np.array([1.0/wd,1.0,0.0])
        return numerator,denominator

    @staticmethod
    def _polyadd(a: np.ndarray,b: np.ndarray) -> np.ndarray:
        n=max(len(a),len(b))
        return np.pad(a,(n-len(a),0))+np.pad(b,(n-len(b),0))

    def closed_loop_poles(self,theta: np.ndarray,vertex) -> np.ndarray:
        m,kx,combined_gain,tau=vertex
        kp_num,kp_den=self.controller_polynomials(theta)
        plant_den=np.array([m*tau,m,-kx*tau,-kx])
        characteristic=self._polyadd(np.convolve(plant_den,kp_den),
                                     np.convolve([combined_gain],kp_num))
        return np.roots(characteristic)

    def robust_hinf_objective(self,theta: np.ndarray,
                              penalize_instability: bool=True) -> float:
        w=self.omega; s=1j*w; wp,wu,wt=self.weights; worst=0.0
        kn,kd=self.controller_polynomials(np.asarray(theta))
        K=np.polyval(kn,s)/np.polyval(kd,s)
        for vertex in self.vertices:
            poles=self.closed_loop_poles(theta,vertex)
            rightmost=float(np.max(np.real(poles)))
            if rightmost>=-self.h.stability_margin_rad_s:
                return 1e3+100*max(rightmost,0.0) if penalize_instability else np.inf
            m,kx,b,tau=vertex
            G=b/((m*s*s-kx)*(tau*s+1.0))
            L=G*K; S=1.0/(1.0+L); T=L*S; KS=K*S
            sigma=np.sqrt(np.abs(wp*S)**2+np.abs(wu*KS)**2+np.abs(wt*T)**2)
            worst=max(worst,float(np.max(sigma)))
        return worst

    def _synthesize(self) -> np.ndarray:
        h=self.h
        bounds=[h.kp_bounds,h.ki_bounds,h.kd_bounds,h.derivative_rolloff_bounds]
        result=optimize.differential_evolution(self.robust_hinf_objective,bounds,
            seed=h.optimizer_seed,maxiter=h.optimizer_iterations,
            popsize=h.optimizer_population,tol=2e-3,polish=True,workers=1)
        if not np.isfinite(result.fun) or result.fun>=1e3:
            raise RuntimeError('H-infinity synthesis failed to find a robustly stable controller.')
        return np.asarray(result.x)

    def compute(self,reference: float,measured_displacement: float,
                voltage_limit: float) -> Tuple[float,float]:
        error=reference-measured_displacement
        raw=float((self.Cd@self.state).item()+self.Dd.ravel()[0]*error)
        self.state=self.Ad@self.state+self.Bd.ravel()*error
        return float(np.clip(raw,-voltage_limit,voltage_limit)),error

    def nominal_frequency_data(self):
        w=self.omega;s=1j*w;p=self.p
        G=(p.current_force_gain*p.amplifier_gain)/(
            (p.rotor_mass*s*s-p.negative_stiffness_magnitude)*
            (p.amplifier_time_constant*s+1))
        K=np.polyval(self.k_num,s)/np.polyval(self.k_den,s)
        L=G*K;S=1/(1+L);T=L*S;wp,wu,wt=self.weights
        return {'omega':w,'L':L,'S':S,'T':T,'WpS':wp*S,'WuKS':wu*K*S,'WtT':wt*T}


class SimulationEnvironment:
    def __init__(self,plant: ThrustAMBPlant,sensor: SensorFilter,
                 controller: HInfinityController,duration: float=3.0):
        self.plant,self.sensor,self.controller = plant,sensor,controller
        self.duration,self.ts = duration,controller.h.sample_time
        n=max(0,controller.h.computational_delay_samples)
        self.delay_line=deque([0.0]*(n+1),maxlen=n+1)

    @staticmethod
    def default_disturbance(t: float) -> float:
        """Step at 0.5 s, sinusoidal surge after 1.5 s and finite pulse."""
        step=0.55 if t>=0.5 else 0.0
        sine=0.10*np.sin(2*np.pi*1.2*(t-1.5)) if t>=1.5 else 0.0
        pulse=-0.30 if 2.15<=t<2.22 else 0.0
        return step+sine+pulse

    def run(self,disturbance: Optional[Callable[[float],float]]=None) -> Dict[str,np.ndarray]:
        disturbance=disturbance or self.default_disturbance
        n=int(round(self.duration/self.ts))+1
        names=('time','displacement','air_gap','measured_displacement','velocity',
               'incremental_current','total_current','voltage_demand','voltage_applied',
               'pwm_duty','disturbance','error')
        log={name:np.zeros(n) for name in names}
        voltage_limit=self.plant.p.effective_voltage_limit()
        for k in range(n):
            t=k*self.ts
            x,v,ic=self.plant.state
            measured=self.sensor.measure(x)
            voltage,error=self.controller.compute(0.0,measured,voltage_limit)
            delayed=self.delay_line.popleft(); self.delay_line.append(voltage)
            force=disturbance(t)
            values=(t,x,self.plant.p.nominal_air_gap+x,measured,v,ic,
                    self.plant.p.bias_current+ic,voltage,self.plant.voltage_applied,
                    self.plant.duty_ratio,force,error)
            for name,value in zip(names,values): log[name][k]=value
            if k<n-1: self.plant.step(delayed,force,self.ts)
        return log


def validate_transfer_function(p: AMBPlantParameters,dt: float=1e-4,
                               duration: float=0.12) -> Dict[str,float]:
    """Compare unsaturated RK4 state equations with scipy.signal G(s)."""
    t=np.arange(0.0,duration+0.5*dt,dt)
    u=0.08*np.sin(2*np.pi*7.0*t)+0.03*(t>=0.035)
    numerical=np.zeros_like(t); state=np.zeros(3)

    def f(s,uk):
        x,v,i=s
        return np.array([v,(p.negative_stiffness_magnitude*x+p.current_force_gain*i)/p.rotor_mass,
                         (p.amplifier_gain*uk-i)/p.amplifier_time_constant])
    for k in range(len(t)-1):
        h=dt; u0=u[k]; u1=u[k+1]; um=0.5*(u0+u1)
        k1=f(state,u0); k2=f(state+0.5*h*k1,um)
        k3=f(state+0.5*h*k2,um); k4=f(state+h*k3,u1)
        state += h*(k1+2*k2+2*k3+k4)/6.0
        numerical[k+1]=state[0]
    numerator,denominator=p.analytical_tf()
    _,analytical,_=signal.lsim(signal.TransferFunction(numerator,denominator),U=u,T=t)
    error=numerical-analytical
    scale=max(np.max(np.abs(analytical)),1e-12)
    return {'tf_validation_RMSE_m':float(np.sqrt(np.mean(error**2))),
            'tf_validation_normalized_RMSE_percent':float(100*np.sqrt(np.mean(error**2))/scale),
            'tf_validation_peak_error_m':float(np.max(np.abs(error)))}


def convergence_time(t: np.ndarray,x: np.ndarray,event: float=0.5,
                     tolerance: float=0.10e-3,dwell: float=0.15,
                     search_end: float=1.49) -> float:
    dt=t[1]-t[0]; width=max(1,int(round(dwell/dt)))
    start=int(np.searchsorted(t,event)); stop=min(int(np.searchsorted(t,search_end)),len(t)-width)
    inside=np.abs(x)<=tolerance
    for k in range(start,stop):
        if np.all(inside[k:k+width]): return float(t[k]-event)
    return float('nan')


def metrics(log: Dict[str,np.ndarray],p: AMBPlantParameters) -> Dict[str,float]:
    x=log['displacement']; gap=log['air_gap']
    return {'position_RMSE_um':float(1e6*np.sqrt(np.mean(x*x))),
            'peak_gap_deviation_um':float(1e6*np.max(np.abs(x))),
            'maximum_total_current_A':float(np.max(log['total_current'])),
            'maximum_incremental_current_A':float(np.max(np.abs(log['incremental_current']))),
            'maximum_voltage_demand_V':float(np.max(np.abs(log['voltage_demand']))),
            'convergence_time_after_step_s':convergence_time(log['time'],x),
            'minimum_air_gap_mm':float(1e3*np.min(gap)),
            'maximum_air_gap_mm':float(1e3*np.max(gap)),
            'safety_bound_violations':int(np.sum((gap<p.minimum_air_gap)|(gap>p.maximum_air_gap)))}


def plot_results(log: Dict[str,np.ndarray],p: AMBPlantParameters,path: Path) -> None:
    t=log['time']; fig,ax=plt.subplots(3,1,figsize=(11,8),sharex=True)
    ax[0].plot(t,1e3*log['air_gap'],label='actual air gap',lw=1.5)
    ax[0].axhline(1e3*p.nominal_air_gap,color='k',ls=':',label='reference')
    ax[0].axhspan(1e3*p.minimum_air_gap,1e3*p.maximum_air_gap,color='green',alpha=.08,label='safe region')
    ax[0].set_ylabel('Air gap [mm]'); ax[0].legend(ncol=3); ax[0].grid(alpha=.25)
    ax[1].plot(t,log['total_current'],label='total coil current')
    ax[1].plot(t,log['incremental_current'],'--',label='incremental current')
    axv=ax[1].twinx(); axv.plot(t,log['voltage_demand'],color='tab:red',alpha=.7,label='Vsp demand')
    lim=p.effective_voltage_limit(); axv.axhline(lim,color='tab:red',ls=':'); axv.axhline(-lim,color='tab:red',ls=':')
    ax[1].set_ylabel('Current [A]'); axv.set_ylabel('Voltage [V]',color='tab:red')
    ax[1].legend(loc='upper left'); axv.legend(loc='upper right'); ax[1].grid(alpha=.25)
    ax[2].plot(t,log['disturbance'],color='tab:purple',label='axial thrust disturbance')
    ax[2].set_ylabel('Load [N]'); ax[2].set_xlabel('Time [s]'); ax[2].legend(); ax[2].grid(alpha=.25)
    fig.suptitle('Cho et al. (2007) Transfer-Function AMB - Robust H-infinity Control')
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def plot_frequency_results(controller: HInfinityController,path: Path) -> None:
    d=controller.nominal_frequency_data(); w=d['omega']; floor=1e-14
    db=lambda x: 20*np.log10(np.maximum(np.abs(x),floor))
    fig,ax=plt.subplots(2,1,figsize=(10,8),sharex=True)
    ax[0].semilogx(w,db(d['L']),label=r'$L=GK$')
    ax[0].semilogx(w,db(d['S']),label=r'$S$')
    ax[0].semilogx(w,db(d['T']),label=r'$T$')
    ax[0].axhline(0,color='k',ls=':',lw=1); ax[0].set_ylabel('Magnitude [dB]')
    ax[0].set_title('Nominal loop, sensitivity and complementary sensitivity')
    ax[0].legend(ncol=3); ax[0].grid(True,which='both',alpha=.25)
    ax[1].semilogx(w,db(d['WpS']),label=r'$W_pS$')
    ax[1].semilogx(w,db(d['WuKS']),label=r'$W_uKS$')
    ax[1].semilogx(w,db(d['WtT']),label=r'$W_tT$')
    ax[1].axhline(0,color='k',ls=':',label='unit bound')
    ax[1].axhline(20*np.log10(controller.gamma),color='tab:red',ls='--',label=r'robust $\gamma$')
    ax[1].set_ylabel('Weighted magnitude [dB]'); ax[1].set_xlabel('Frequency [rad/s]')
    ax[1].set_title('Mixed-sensitivity performance channels')
    ax[1].legend(ncol=3); ax[1].grid(True,which='both',alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main(argv=None) -> Dict[str,float]:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='cho2007_hinf_results')
    parser.add_argument('--duration',type=float,default=3.0)
    parser.add_argument('--seed',type=int,default=21)
    args,_=parser.parse_known_args(argv) # safe inside Jupyter (-f kernel.json)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    pp,sp,hp=AMBPlantParameters(),SensorParameters(),HInfinityParameters()
    tf_check=validate_transfer_function(pp)
    plant=ThrustAMBPlant(pp); sensor=SensorFilter(sp,hp.sample_time,pp.initial_displacement,args.seed)
    controller=HInfinityController(hp,pp,synthesize=True)
    log=SimulationEnvironment(plant,sensor,controller,args.duration).run()
    rightmost=max(np.max(np.real(controller.closed_loop_poles(controller.theta,v)))
                  for v in controller.vertices)
    report={**metrics(log,pp),**tf_check,'robust_mixed_sensitivity_gamma':controller.gamma,
            'worst_vertex_rightmost_pole_real':float(rightmost),
            'controller_Kp':float(controller.theta[0]),'controller_Ki':float(controller.theta[1]),
            'controller_Kd':float(controller.theta[2]),
            'controller_derivative_rolloff_rad_s':float(controller.theta[3])}
    plot_results(log,pp,out/'cho2007_hinf_time_response.png')
    plot_frequency_results(controller,out/'cho2007_hinf_frequency_response.png')
    np.savez_compressed(out/'simulation_data.npz',**log)
    with open(out/'metrics.json','w',encoding='utf-8') as f: json.dump(report,f,indent=2)
    with open(out/'parameters.json','w',encoding='utf-8') as f:
        json.dump({'plant':asdict(pp),'sensor':asdict(sp),'hinfinity':asdict(hp)},f,indent=2)
    with open(out/'synthesized_controller.json','w',encoding='utf-8') as f:
        json.dump({'theta_Kp_Ki_Kd_wd':controller.theta.tolist(),
                   'continuous_numerator':controller.k_num.tolist(),
                   'continuous_denominator':controller.k_den.tolist(),
                   'Ad':controller.Ad.tolist(),'Bd':controller.Bd.tolist(),
                   'Cd':controller.Cd.tolist(),'Dd':controller.Dd.tolist(),
                   'sample_time_s':hp.sample_time,'robust_gamma':controller.gamma},f,indent=2)
    print('\nCho et al. transfer-function Robust H-infinity validation')
    print('-'*62)
    for key,value in report.items():
        shown='not reached' if isinstance(value,float) and np.isnan(value) else f'{value:.6g}'
        print(f'{key:43s}: {shown}')
    print(f'Results directory                          : {out.resolve()}')
    return report


if __name__=='__main__':
    main()
