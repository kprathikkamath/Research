#!/usr/bin/env python3
"""Linear active-disturbance-rejection control of a thrust AMB.

The bias-linearized plant requested by the user is

                    k_i K_amp
    G(s) = --------------------------------,
           (m s**2 - k_x) (tau s + 1)

with the state realization

    x_dot = v
    m v_dot = k_x*x + k_i*i_c + F_d
    tau i_c_dot = -i_c + K_amp*V_sp.

``k_x`` is stored as the positive magnitude of the destabilizing stiffness in
``m*x_ddot-k_x*x``.  If a source reports a signed negative coefficient, enter
its magnitude.  Position ``x`` is displacement from the nominal air gap.

The controller is a practical second-order LADRC.  It intentionally uses only
the nominal input gain b0; negative stiffness, amplifier lag/model mismatch,
and axial load are absorbed into a generalized disturbance estimated by a
third-order linear extended-state observer (LESO).

Dependencies: numpy, scipy, matplotlib
Run in a terminal: python Cho2007_ADRC_Thrust_AMB.py
Run in Jupyter:   %run Cho2007_ADRC_Thrust_AMB.py
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
from scipy import signal


@dataclass
class AMBPlantParameters:
    """Replace these placeholder values with identified experimental values."""

    rotor_mass: float = 0.42
    negative_stiffness_magnitude: float = 120.0
    current_force_gain: float = 3.8
    amplifier_gain: float = 0.45
    amplifier_time_constant: float = 0.006
    nominal_air_gap: float = 0.014
    bias_current: float = 2.20

    control_current_min: float = -1.80
    control_current_max: float = 1.80
    voltage_setpoint_limit: float = 10.0
    dc_bus_voltage: float = 48.0
    pwm_duty_limit: float = 0.95
    amplifier_efficiency: float = 0.97

    minimum_air_gap: float = 0.010
    maximum_air_gap: float = 0.018
    initial_displacement: float = 0.25e-3
    initial_velocity: float = 0.0

    def effective_voltage_limit(self) -> float:
        return min(self.voltage_setpoint_limit,
                   self.dc_bus_voltage * self.pwm_duty_limit)

    def analytical_tf(self) -> Tuple[np.ndarray, np.ndarray]:
        m = self.rotor_mass
        kx = self.negative_stiffness_magnitude
        tau = self.amplifier_time_constant
        return (np.array([self.current_force_gain * self.amplifier_gain]),
                np.array([m * tau, m, -kx * tau, -kx]))


@dataclass
class SensorParameters:
    sensitivity: float = 800.0
    bias_voltage: float = 0.0008
    noise_variance: float = 4.0e-7
    lowpass_cutoff_hz: float = 100.0


@dataclass
class ADRCParameters:
    sample_time: float = 0.001
    control_bandwidth_rad_s: float = 45.0
    observer_bandwidth_rad_s: float = 180.0
    tracking_differentiator_bandwidth_rad_s: float = 30.0
    damping_ratio: float = 1.0
    b0_scale: float = 1.0
    computational_delay_samples: int = 0
    maximum_voltage_slew_rate: float = 2500.0  # V/s


@dataclass
class SimulationParameters:
    duration: float = 3.0
    integration_substeps: int = 4
    disturbance_onset: float = 0.5
    disturbance_settling_band: float = 0.10e-3
    disturbance_settling_hold_time: float = 0.15


class ThrustAMBPlant:
    """Saturated RK4 realization of the exact third-order transfer model."""

    def __init__(self, p: AMBPlantParameters):
        self.p = p
        self.state = np.array([p.initial_displacement, p.initial_velocity, 0.0])
        self.voltage_applied = 0.0
        self.duty_ratio = 0.0

    def limited_voltage(self, demand: float) -> float:
        command = float(np.clip(demand, -self.p.effective_voltage_limit(),
                                self.p.effective_voltage_limit()))
        self.duty_ratio = command / self.p.dc_bus_voltage
        return self.p.amplifier_efficiency * command

    def derivative(self, state: np.ndarray, voltage_demand: float,
                   load_force: float, apply_limits: bool = True) -> np.ndarray:
        p = self.p
        x, velocity, incremental_current = state
        voltage = self.limited_voltage(voltage_demand) if apply_limits else voltage_demand
        current_target = p.amplifier_gain * voltage
        if apply_limits:
            current_target = float(np.clip(current_target, p.control_current_min,
                                           p.control_current_max))
        current_dot = (current_target - incremental_current) / p.amplifier_time_constant
        if apply_limits and ((incremental_current <= p.control_current_min and current_dot < 0)
                             or (incremental_current >= p.control_current_max and current_dot > 0)):
            current_dot = 0.0
        acceleration = (p.negative_stiffness_magnitude * x
                        + p.current_force_gain * incremental_current
                        + load_force) / p.rotor_mass
        return np.array([velocity, acceleration, current_dot])

    def step(self, voltage_demand: float, load_force: float, dt: float,
             substeps: int = 4) -> None:
        h = dt / substeps
        for _ in range(substeps):
            s = self.state
            k1 = self.derivative(s, voltage_demand, load_force)
            k2 = self.derivative(s + 0.5*h*k1, voltage_demand, load_force)
            k3 = self.derivative(s + 0.5*h*k2, voltage_demand, load_force)
            k4 = self.derivative(s + h*k3, voltage_demand, load_force)
            self.state = s + h*(k1 + 2*k2 + 2*k3 + k4)/6.0
        self.state[2] = np.clip(self.state[2], self.p.control_current_min,
                                self.p.control_current_max)
        self.voltage_applied = self.limited_voltage(voltage_demand)


class SensorFilter:
    """Position sensitivity, offset, Gaussian noise, and 2nd-order Butterworth."""

    def __init__(self, p: SensorParameters, sample_time: float,
                 initial_displacement: float, seed: int = 21):
        if p.lowpass_cutoff_hz >= 0.5/sample_time:
            raise ValueError("Sensor cutoff must be below the Nyquist frequency.")
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.sos = signal.butter(2, p.lowpass_cutoff_hz, fs=1/sample_time,
                                 output="sos")
        initial_voltage = p.sensitivity*initial_displacement + p.bias_voltage
        self.zi = signal.sosfilt_zi(self.sos) * initial_voltage

    def measure(self, displacement: float) -> float:
        voltage = (self.p.sensitivity*displacement + self.p.bias_voltage
                   + self.rng.normal(0.0, np.sqrt(self.p.noise_variance)))
        filtered, self.zi = signal.sosfilt(self.sos, [voltage], zi=self.zi)
        # Dividing by sensitivity leaves residual sensor bias visible to ADRC.
        return float(filtered[0] / self.p.sensitivity)


class ActiveDisturbanceRejectionController:
    """Tracking differentiator + third-order LESO + disturbance-canceling PD.

    The controller assumes only the canonical second-order relation

        x_ddot = b0*u + f_total.

    For the LESO, e_o=y-z1 and

        z1_dot = z2 + beta1*e_o
        z2_dot = z3 + b0*u + beta2*e_o
        z3_dot =             beta3*e_o.

    Choosing beta=[3*wo, 3*wo**2, wo**3] places all nominal observer poles at
    -wo.  The control law

        u = (wc**2*(v1-z1) + 2*zeta*wc*(v2-z2) - z3) / b0

    actively cancels the estimated generalized disturbance z3.  The observer
    is driven by the *actually limited/delayed* command from the preceding
    interval, which avoids observer windup when the amplifier saturates.
    """

    def __init__(self, c: ADRCParameters, plant: AMBPlantParameters):
        self.c = c
        self.voltage_limit = plant.effective_voltage_limit()
        self.b0 = (c.b0_scale * plant.current_force_gain * plant.amplifier_gain
                   / plant.rotor_mass)
        if self.b0 <= 0:
            raise ValueError("The nominal ADRC input gain b0 must be positive.")
        wo, wc = c.observer_bandwidth_rad_s, c.control_bandwidth_rad_s
        self.beta = np.array([3*wo, 3*wo**2, wo**3])
        self.kp = wc**2
        self.kd = 2*c.damping_ratio*wc
        self.z = np.zeros(3)
        self.td = np.zeros(2)
        self.previous_voltage = 0.0

    @staticmethod
    def _rk4(state: np.ndarray, dt: float,
             rhs: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        k1 = rhs(state)
        k2 = rhs(state + 0.5*dt*k1)
        k3 = rhs(state + 0.5*dt*k2)
        k4 = rhs(state + dt*k3)
        return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

    def update(self, reference: float, measurement: float,
               applied_voltage_previous: float) -> Tuple[float, Dict[str, float]]:
        dt = self.c.sample_time
        wt = self.c.tracking_differentiator_bandwidth_rad_s

        # Critically damped TD: v1 follows the requested position while v2 is
        # its smooth derivative, preventing reference derivative kick.
        def td_rhs(q: np.ndarray) -> np.ndarray:
            return np.array([q[1], -2*wt*q[1] - wt**2*(q[0]-reference)])
        self.td = self._rk4(self.td, dt, td_rhs)

        # Measurement and held input are constant over one digital interval.
        def eso_rhs(z: np.ndarray) -> np.ndarray:
            observer_error = measurement - z[0]
            return np.array([z[1] + self.beta[0]*observer_error,
                             z[2] + self.b0*applied_voltage_previous
                             + self.beta[1]*observer_error,
                             self.beta[2]*observer_error])
        self.z = self._rk4(self.z, dt, eso_rhs)

        virtual_acceleration = (self.kp*(self.td[0]-self.z[0])
                                + self.kd*(self.td[1]-self.z[1]))
        unlimited = (virtual_acceleration - self.z[2]) / self.b0
        max_change = self.c.maximum_voltage_slew_rate * dt
        slew_limited = np.clip(unlimited, self.previous_voltage-max_change,
                               self.previous_voltage+max_change)
        command = float(np.clip(slew_limited, -self.voltage_limit,
                                self.voltage_limit))
        self.previous_voltage = command
        return command, {"td_position": self.td[0], "td_velocity": self.td[1],
                         "z1": self.z[0], "z2": self.z[1], "z3": self.z[2],
                         "unlimited_voltage": unlimited}


def default_reference(_: float) -> float:
    return 0.0


def default_disturbance(t: float) -> float:
    """Step, sinusoidal surge, and finite-duration pulse in newtons."""
    step = 0.55 if t >= 0.5 else 0.0
    sine = 0.10*np.sin(2*np.pi*1.2*(t-1.5)) if t >= 1.5 else 0.0
    pulse = -0.30 if 2.15 <= t < 2.22 else 0.0
    return step + sine + pulse


class SimulationEnvironment:
    def __init__(self, plant_parameters: AMBPlantParameters,
                 sensor_parameters: SensorParameters,
                 controller_parameters: ADRCParameters,
                 simulation_parameters: SimulationParameters,
                 seed: int = 21,
                 reference: Callable[[float], float] = default_reference,
                 disturbance: Callable[[float], float] = default_disturbance):
        self.pp, self.sp, self.cp, self.sim = (plant_parameters, sensor_parameters,
                                               controller_parameters, simulation_parameters)
        self.plant = ThrustAMBPlant(self.pp)
        self.sensor = SensorFilter(self.sp, self.cp.sample_time,
                                   self.pp.initial_displacement, seed)
        self.controller = ActiveDisturbanceRejectionController(self.cp, self.pp)
        self.reference, self.disturbance = reference, disturbance

    def run(self) -> Dict[str, np.ndarray]:
        dt = self.cp.sample_time
        time = np.arange(0.0, self.sim.duration + 0.5*dt, dt)
        names = ("position", "velocity", "measurement", "reference", "gap",
                 "incremental_current", "total_current", "voltage_demand",
                 "voltage_applied", "load_force", "true_total_disturbance",
                 "estimated_total_disturbance", "z1", "z2", "td_position",
                 "td_velocity", "unlimited_voltage")
        log = {name: np.zeros(time.size) for name in names}
        delay: Deque[float] = deque([0.0]*(self.cp.computational_delay_samples+1),
                                    maxlen=self.cp.computational_delay_samples+1)
        applied_command_previous = 0.0

        for k, t in enumerate(time):
            x, velocity, current = self.plant.state
            measured = self.sensor.measure(x)
            reference = self.reference(float(t))
            demand, observer = self.controller.update(reference, measured,
                                                       applied_command_previous)
            delay.append(demand)
            delayed_demand = delay[0]
            load = self.disturbance(float(t))

            # The exact generalized disturbance relative to x_ddot=b0*u+f is
            # reconstructed from the simulated physical states for validation.
            acceleration = (self.pp.negative_stiffness_magnitude*x
                            + self.pp.current_force_gain*current + load) / self.pp.rotor_mass
            true_total = acceleration - self.controller.b0*applied_command_previous

            log["position"][k], log["velocity"][k] = x, velocity
            log["measurement"][k], log["reference"][k] = measured, reference
            log["gap"][k] = self.pp.nominal_air_gap + x
            log["incremental_current"][k] = current
            log["total_current"][k] = self.pp.bias_current + current
            log["voltage_demand"][k] = demand
            log["voltage_applied"][k] = self.plant.voltage_applied
            log["load_force"][k] = load
            log["true_total_disturbance"][k] = true_total
            log["estimated_total_disturbance"][k] = observer["z3"]
            for key in ("z1", "z2", "td_position", "td_velocity", "unlimited_voltage"):
                log[key][k] = observer[key]

            self.plant.step(delayed_demand, load, dt,
                            self.sim.integration_substeps)
            applied_command_previous = delayed_demand

        log["time"] = time
        return log

    def metrics(self, d: Dict[str, np.ndarray]) -> Dict[str, Optional[float]]:
        error = d["reference"] - d["position"]
        settling = disturbance_rejection_settling_time(
            d["time"], error, self.sim.disturbance_onset,
            self.sim.disturbance_settling_band,
            self.sim.disturbance_settling_hold_time)
        return {
            "position_RMSE_um": float(1e6*np.sqrt(np.mean(error**2))),
            "peak_gap_deviation_um": float(1e6*np.max(np.abs(error))),
            "maximum_incremental_current_A": float(np.max(np.abs(d["incremental_current"]))),
            "maximum_total_current_A": float(np.max(np.abs(d["total_current"]))),
            "maximum_voltage_demand_V": float(np.max(np.abs(d["voltage_demand"]))),
            "disturbance_estimation_RMSE_m_per_s2": float(np.sqrt(np.mean(
                (d["true_total_disturbance"]-d["estimated_total_disturbance"])**2))),
            "disturbance_rejection_settling_time_s": settling,
            "minimum_gap_mm": float(1e3*np.min(d["gap"])),
            "maximum_gap_mm": float(1e3*np.max(d["gap"])),
            "safety_gap_violation": bool(np.any(d["gap"] < self.pp.minimum_air_gap)
                                         or np.any(d["gap"] > self.pp.maximum_air_gap)),
        }


def disturbance_rejection_settling_time(time: np.ndarray, error: np.ndarray,
                                         onset: float, band: float,
                                         hold_time: float) -> Optional[float]:
    dt = float(time[1]-time[0])
    hold = max(1, int(np.ceil(hold_time/dt)))
    start = int(np.searchsorted(time, onset))
    inside = np.abs(error) <= band
    for k in range(start, len(time)-hold+1):
        if np.all(inside[k:k+hold]):
            return float(time[k]-onset)
    return None


def validate_transfer_function(p: AMBPlantParameters) -> float:
    """Numerically validate the unsaturated RK4 realization against scipy TF."""
    dt, duration, amplitude = 2e-5, 0.12, 0.05
    time = np.arange(0.0, duration+0.5*dt, dt)
    num, den = p.analytical_tf()
    _, analytical = signal.step(signal.TransferFunction(num, den), T=time)
    analytical *= amplitude
    state = np.zeros(3)
    numerical = np.zeros_like(time)

    def rhs(q: np.ndarray) -> np.ndarray:
        x, v, current = q
        return np.array([v,
            (p.negative_stiffness_magnitude*x+p.current_force_gain*current)/p.rotor_mass,
            (p.amplifier_gain*amplitude-current)/p.amplifier_time_constant])

    for k in range(time.size-1):
        numerical[k] = state[0]
        k1 = rhs(state); k2 = rhs(state+0.5*dt*k1)
        k3 = rhs(state+0.5*dt*k2); k4 = rhs(state+dt*k3)
        state += dt*(k1+2*k2+2*k3+k4)/6
    numerical[-1] = state[0]
    scale = max(np.max(np.abs(analytical)), np.finfo(float).eps)
    return float(100*np.sqrt(np.mean((numerical-analytical)**2))/scale)


def plot_results(d: Dict[str, np.ndarray], p: AMBPlantParameters,
                 output: Path) -> None:
    t = d["time"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True,
                             constrained_layout=True)
    axes[0].plot(t, 1e3*d["gap"], label="actual air gap", lw=1.5)
    axes[0].plot(t, 1e3*(p.nominal_air_gap+d["reference"]), "--", label="reference")
    axes[0].axhline(1e3*p.minimum_air_gap, color="r", ls=":", label="safety bounds")
    axes[0].axhline(1e3*p.maximum_air_gap, color="r", ls=":")
    axes[0].set_ylabel("Gap [mm]"); axes[0].legend(ncol=3); axes[0].grid(True)

    axes[1].plot(t, d["incremental_current"], label="incremental current")
    axes[1].axhline(p.control_current_max, color="tab:blue", ls=":")
    axes[1].axhline(p.control_current_min, color="tab:blue", ls=":")
    axes[1].set_ylabel("Current [A]"); axes[1].grid(True)
    voltage_axis = axes[1].twinx()
    voltage_axis.plot(t, d["voltage_demand"], color="tab:orange", alpha=.8,
                      label="voltage demand")
    voltage_axis.axhline(p.effective_voltage_limit(), color="tab:orange", ls=":")
    voltage_axis.axhline(-p.effective_voltage_limit(), color="tab:orange", ls=":")
    voltage_axis.set_ylabel("Voltage [V]")
    lines = axes[1].lines[:1] + voltage_axis.lines[:1]
    axes[1].legend(lines, [line.get_label() for line in lines], loc="upper right")

    axes[2].plot(t, d["true_total_disturbance"], label="true generalized disturbance")
    axes[2].plot(t, d["estimated_total_disturbance"], "--", label="LESO estimate z3")
    axes[2].set_ylabel(r"Disturbance [m/s$^2$]")
    axes[2].legend(); axes[2].grid(True)

    axes[3].plot(t, d["load_force"], color="tab:red", label="external axial load")
    axes[3].set_ylabel("Load [N]"); axes[3].set_xlabel("Time [s]")
    axes[3].legend(); axes[3].grid(True)
    fig.suptitle("Cho (2007) thrust AMB — linear ADRC validation", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cho-model thrust AMB LADRC simulation")
    parser.add_argument("--output", type=Path, default=Path("cho2007_adrc_results"))
    parser.add_argument("--seed", type=int, default=21)
    # parse_known_args is deliberate: Jupyter injects a '-f kernel.json' pair.
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    plant = AMBPlantParameters()
    sensor = SensorParameters()
    controller = ADRCParameters()
    simulation = SimulationParameters()

    tf_error = validate_transfer_function(plant)
    environment = SimulationEnvironment(plant, sensor, controller, simulation,
                                        seed=args.seed)
    data = environment.run()
    metrics = environment.metrics(data)
    metrics["transfer_function_validation_normalized_RMSE_percent"] = tf_error

    np.savez_compressed(args.output/"simulation_data.npz", **data)
    plot_results(data, plant, args.output/"adrc_validation.png")
    with (args.output/"metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    parameters = {"plant": asdict(plant), "sensor": asdict(sensor),
                  "controller": asdict(controller), "simulation": asdict(simulation)}
    with (args.output/"parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(parameters, handle, indent=2)

    print("\nCho (2007) thrust AMB — LADRC validation")
    print("-"*51)
    for key, value in metrics.items():
        print(f"{key:52s}: {value}")
    print(f"\nResults written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
