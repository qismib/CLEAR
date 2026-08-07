"""CLEAR-pulse photon-population experiment, following:
    D. T. McClure et al., "Rapid Driven Reset of a Qubit Readout Resonator",
    Phys. Rev. Applied 5, 011001 (2016).

Sequence, Fig. 1(b):
    (RX) -> M1 -> t_relax -> RX90 - t_R - RX90 -> t_buffer -> M2

Timing conventions from the paper: t_relax runs from the end of M1 to the
first X90; the two X90 pulses are separated by exactly t_R; t_buffer separates
the Ramsey from M2. M1 is either the 5-segment CLEAR pulse of Fig. 1(a) or the
square baseline (whose length matches CLEAR minus its ring-down segments, so
that square + t_relax = ringdown_duration reproduces the "square + delay"
comparison of Fig. 3(c)).

The Ramsey detuning is applied as a phase ramp on the second X90, not as a
drive-frequency offset, so the RX() preparation stays on resonance.

All parameters come from clear_config.SETUP; the analysis lives in
clear_analysis. This file is the hardware layer only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
    create_platform,
)

from clear_config import ClearPulseParameters, Setup, get_setup
from clear_analysis import (
    RamseyModel,
    StateReference,
    excited_fraction_from_reference,
    fit_pair,
    plot_ramsey,
    save_figure,
    summarize,
)


# ---------------------------------------------------------------------------
# 1. Pulse-sequence building (unchanged behaviour)
# ---------------------------------------------------------------------------


def rectangular_iq_pulse(duration: float, amplitude: complex) -> Pulse:
    """Rectangular pulse from a complex amplitude."""
    magnitude = abs(amplitude)
    if magnitude > 1:
        raise ValueError(
            f"segment amplitude {magnitude:.3f} exceeds 1; rescale into [-1, 1]."
        )
    return Pulse(
        duration=duration,
        amplitude=float(magnitude),
        envelope=Rectangular(),
        relative_phase=float(np.angle(amplitude)),
    )


def clear_drive_sequence(
    platform,
    qubit: int,
    params: ClearPulseParameters,
    rectangular: bool = False,
    match_clear_length: bool = True,
) -> PulseSequence:
    """M1: the 5-segment CLEAR pulse, or the square baseline at the same
    steady-state amplitude."""
    probe_channel = platform.qubits[qubit].probe
    sequence = PulseSequence()

    if rectangular:
        duration = params.duration(rectangular=True, match_clear_length=match_clear_length)
        if duration > 0:
            sequence.append((probe_channel, rectangular_iq_pulse(duration, params.steady)))
        return sequence

    for duration, amplitude in zip(params.durations, params.amplitudes):
        if duration > 0:
            sequence.append((probe_channel, rectangular_iq_pulse(duration, amplitude)))
    return sequence


def _delay_on_all(channels: Iterable, duration: float) -> tuple[PulseSequence, list[Delay]]:
    """Same delay on several channels at once.

    Returns the Delay objects too: a duration Sweeper must receive all of
    them, since sweeping one alone would desynchronize the channels.
    """
    delays = [Delay(duration=duration) for _ in channels]
    return PulseSequence(list(zip(channels, delays))), delays


def build_ramsey_sequence(
    platform,
    qubit: int,
    clear: ClearPulseParameters,
    t_relax: float,
    t_ramsey: float,
    t_buffer: float,
    prepare_excited: bool = False,
    rectangular: bool = False,
    match_clear_length: bool = True,
    with_ramsey: bool = True,
) -> tuple[PulseSequence, list[Delay], Pulse | None, int]:
    """Builds Fig. 1(b).

    `with_ramsey=False` drops the two X90 pulses and the t_R delay, leaving
    (RX) -> M1 -> t_relax -> M2. That is the reference sequence used by
    `run_excited_fraction`: it reads out the qubit at exactly the moment the
    Ramsey would have begun, which is what the T1 correction needs.

    M2 is the platform's calibrated native MZ (the paper uses a 10 us square
    pulse; change the readout duration in the platform to match).

    Returns the sequence, the Ramsey delays (all of them, for the Sweeper),
    the second X90 carrying the detuning phase, and the M2 acquisition id.
    """

    native = platform.natives.single_qubit[qubit]
    drive_channel = platform.qubits[qubit].drive
    acquisition_channel = platform.qubits[qubit].acquisition
    channels = (drive_channel, platform.qubits[qubit].probe, acquisition_channel)

    sequence = PulseSequence()

    if prepare_excited:
        sequence |= native.RX()

    sequence |= clear_drive_sequence(
        platform, qubit, clear, rectangular=rectangular, match_clear_length=match_clear_length
    )

    if t_relax > 0:
        relax_sequence, _ = _delay_on_all(channels, t_relax)
        sequence |= relax_sequence

    ramsey_delays: list[Delay] = []
    second_x90: Pulse | None = None

    if with_ramsey:
        sequence |= native.RX90()
        ramsey_sequence, ramsey_delays = _delay_on_all(channels, t_ramsey)
        sequence |= ramsey_sequence

        second_x90_sequence = native.RX90()
        second_x90 = [
            pulse
            for pulse in second_x90_sequence.channel(drive_channel)
            if isinstance(pulse, Pulse)
        ][-1]
        sequence |= second_x90_sequence

    buffer_sequence, _ = _delay_on_all(channels, t_buffer)
    sequence |= buffer_sequence

    m2 = native.MZ()
    m2_readout = list(m2.channel(acquisition_channel))[-1]
    sequence |= m2

    return sequence, ramsey_delays, second_x90, m2_readout.id


def build_state_reference_sequence(platform, qubit: int, prepare_excited: bool):
    """(RX) -> M2 with no M1 and no Ramsey: the |0>/|1> calibration points."""
    native = platform.natives.single_qubit[qubit]
    sequence = PulseSequence()
    if prepare_excited:
        sequence |= native.RX()
    m2 = native.MZ()
    m2_readout = list(m2.channel(platform.qubits[qubit].acquisition))[-1]
    sequence |= m2
    return sequence, m2_readout.id


# ---------------------------------------------------------------------------
# 2. Acquisition
# ---------------------------------------------------------------------------


def _probe_update(platform, setup: Setup, frequency_hz: float | None = None) -> list[dict]:
    """Pins the probe channel to the midpoint of the two state-dependent
    resonator frequencies, which is what makes the steady-state photon number
    independent of the prepared state.

    `frequency_hz` overrides the midpoint. Only CLEAR_midpoint.py should use
    the override: it is the experiment that *measures* the midpoint, so it has
    to be able to sit deliberately off it.
    """
    frequency = setup.readout_frequency_hz if frequency_hz is None else float(frequency_hz)
    return [{platform.qubits[setup.qubit].probe: {"frequency": frequency}}]


def _execute(platform, setup: Setup, sequence, sweepers, updates):
    return platform.execute(
        [sequence],
        [sweepers] if sweepers else [],
        nshots=setup.nshots,
        relaxation_time=setup.relaxation_time_ns,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLESHOT,
        updates=updates,
    )


def run_state_reference(
    platform, setup: Setup, frequency_hz: float | None = None
) -> StateReference:
    """Measures the bare |0> and |1> IQ points, so every trace can be turned
    into an excited-state probability instead of an arbitrary quadrature.

    The blobs move with the probe frequency, so this must be re-measured
    whenever `frequency_hz` changes.
    """
    updates = _probe_update(platform, setup, frequency_hz)
    points = {}
    for excited in (False, True):
        sequence, m2_id = build_state_reference_sequence(platform, setup.qubit, excited)
        points[excited] = _execute(platform, setup, sequence, None, updates)[m2_id]
    return StateReference.from_acquisitions(points[False], points[True])


def run_excited_fraction(
    platform,
    setup: Setup,
    clear: ClearPulseParameters,
    t_relax_values: np.ndarray,
    reference: StateReference,
    rectangular: bool = False,
    match_clear_length: bool = True,
    frequency_hz: float | None = None,
) -> dict[float, float]:
    """Measures the surviving excited fraction p at the start of the Ramsey,
    for each t_relax, by running the identical sequence with the X90 pulses
    removed.

    This is preferable to exp(-t/T1): the readout drive induces extra
    transitions, so the decay during M1 is generally faster than the bare T1,
    and the RX pulse has its own infidelity. Both are absorbed here.
    """
    updates = _probe_update(platform, setup, frequency_hz)
    fractions = {}
    for t_relax in np.atleast_1d(t_relax_values).astype(float):
        sequence, _, _, m2_id = build_ramsey_sequence(
            platform,
            setup.qubit,
            clear,
            t_relax=t_relax,
            t_ramsey=0.0,
            t_buffer=setup.t_buffer_ns,
            prepare_excited=True,
            rectangular=rectangular,
            match_clear_length=match_clear_length,
            with_ramsey=False,
        )
        result = _execute(platform, setup, sequence, None, updates)[m2_id]
        fractions[float(t_relax)] = excited_fraction_from_reference(result, reference)
    return fractions


def run_ramsey_scan(
    platform,
    setup: Setup,
    clear: ClearPulseParameters,
    t_relax_values: np.ndarray,
    t_ramsey_values: np.ndarray,
    prepare_excited: bool,
    rectangular: bool = False,
    match_clear_length: bool = True,
    frequency_hz: float | None = None,
) -> dict[float, np.ndarray]:
    """Ramsey-after-M1 traces, one per t_relax. Returns raw acquisitions."""
    updates = _probe_update(platform, setup, frequency_hz)
    t_ramsey_values = np.asarray(t_ramsey_values, dtype=float)
    traces = {}

    for t_relax in np.atleast_1d(t_relax_values).astype(float):
        sequence, ramsey_delays, second_x90, m2_id = build_ramsey_sequence(
            platform,
            setup.qubit,
            clear,
            t_relax=t_relax,
            t_ramsey=float(t_ramsey_values[0]),
            t_buffer=setup.t_buffer_ns,
            prepare_excited=prepare_excited,
            rectangular=rectangular,
            match_clear_length=match_clear_length,
        )

        sweepers = [
            Sweeper(
                parameter=Parameter.duration,
                values=t_ramsey_values,
                pulses=ramsey_delays,
            )
        ]
        if setup.ramsey_detuning_hz:
            # phi(t_R) = 2*pi*Delta*t_R, Delta in Hz and t_R in ns.
            phases = (2 * np.pi * setup.ramsey_detuning_hz * 1e-9 * t_ramsey_values) % (2 * np.pi)
            sweepers.append(
                Sweeper(
                    parameter=Parameter.relative_phase,
                    values=phases,
                    pulses=[second_x90],
                )
            )

        traces[float(t_relax)] = _execute(platform, setup, sequence, sweepers, updates)[m2_id]

    return traces


def run_experiment(
    setup: Setup,
    t_relax_values: np.ndarray,
    t_ramsey_values: np.ndarray,
    rectangular: bool = False,
    match_clear_length: bool = True,
    output_name: str = "clear_ramsey",
) -> Path:
    """One connection, one saved file: state reference, excited-fraction
    calibration, and the ground/excited Ramsey scans.

    Acquiring all of it in a single platform session matters -- the reference
    points and the T1 correction are only valid for the readout conditions
    they were taken under.
    """

    clear = setup.clear_pulse()
    t_relax_values = np.atleast_1d(np.asarray(t_relax_values, dtype=float))
    t_ramsey_values = np.asarray(t_ramsey_values, dtype=float)

    platform = create_platform(setup.platform)
    platform.connect()
    try:
        reference = run_state_reference(platform, setup)
        fractions = run_excited_fraction(
            platform, setup, clear, t_relax_values, reference, rectangular, match_clear_length
        )
        traces = {
            state: run_ramsey_scan(
                platform,
                setup,
                clear,
                t_relax_values,
                t_ramsey_values,
                prepare_excited=(state == "excited"),
                rectangular=rectangular,
                match_clear_length=match_clear_length,
            )
            for state in ("ground", "excited")
        }
    finally:
        platform.disconnect()

    payload = {
        "t_relax": t_relax_values,
        "t_ramsey": t_ramsey_values,
        "reference_iq": np.array([reference.iq_ground, reference.iq_excited]),
        "p_excited": np.array([fractions[float(t)] for t in t_relax_values]),
        "rectangular": np.array(rectangular),
    }
    for state, per_relax in traces.items():
        for t_relax, iq in per_relax.items():
            payload[f"{state}_t_relax_{t_relax:g}"] = iq

    setup.data_dir.mkdir(parents=True, exist_ok=True)
    path = setup.data_dir / f"{output_name}_{'square' if rectangular else 'clear'}.npz"
    np.savez(path, **payload)
    return path


# ---------------------------------------------------------------------------
# 3. Analysis of a saved run
# ---------------------------------------------------------------------------


def analyze_run(
    npz_path: Path,
    setup: Setup,
    use_measured_p: bool = True,
    plot_dir: Path | None = None,
) -> dict:
    """Fits every ground/excited pair in a saved run and returns
    {t_relax: {"ground": fit, "excited": fit, "p_excited": p}}.

    `use_measured_p=False` falls back to exp(-t/T1) from the config, for data
    taken before the calibration sequence existed.
    """

    data = np.load(npz_path, allow_pickle=False)
    model = RamseyModel.from_setup(setup)
    reference = StateReference(*data["reference_iq"])
    rectangular = bool(data["rectangular"])
    t_ramsey = data["t_ramsey"]
    plot_dir = plot_dir or npz_path.parent

    results = {}
    for i, t_relax in enumerate(data["t_relax"]):
        traces = {
            state: reference.project(data[f"{state}_t_relax_{t_relax:g}"])
            for state in ("ground", "excited")
        }

        if use_measured_p and "p_excited" in data:
            p = float(data["p_excited"][i])
        else:
            p = setup.excited_fraction(t_relax, rectangular)

        fits = fit_pair(t_ramsey, traces["ground"], traces["excited"], model, p_excited=p)
        fits["p_excited"] = p
        results[float(t_relax)] = fits

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, (state, color) in zip(axes, (("ground", "tab:blue"), ("excited", "tab:red"))):
            plot_ramsey(t_ramsey, traces[state], fits[state], f"{state}, $t_{{relax}}$ = {t_relax:g} ns", ax, color)
        save_figure(fig, Path(plot_dir) / f"ramsey_fit_t_relax_{t_relax:g}.png")

        print(f"t_relax = {t_relax:6.0f} ns   (p_e = {p:.3f})")
        for state in ("ground", "excited"):
            print("  " + summarize(state, fits[state]))

    return results


def plot_n0_decay(results: dict, setup: Setup, output_path: Path) -> None:
    """n0 versus t_relax for both preparations, with the expected exp(-kappa t)
    envelope for reference."""
    fig, ax = plt.subplots(figsize=(6, 4))
    t_relax = np.array(sorted(results))

    for state, color in (("ground", "tab:blue"), ("excited", "tab:red")):
        n0 = np.array([results[t][state]["n0"] for t in t_relax])
        err = np.array([results[t][state]["n0_err"] for t in t_relax])
        ax.errorbar(t_relax, n0, err, marker="o", ls="none", color=color, label=state, capsize=3)

    if t_relax.size > 1:
        t_fine = np.linspace(t_relax.min(), t_relax.max(), 300)
        n0_first = results[float(t_relax[0])]["ground"]["n0"]
        ax.plot(
            t_fine,
            n0_first * np.exp(-setup.kappa_rad_ns * (t_fine - t_relax[0])),
            "k--",
            lw=1,
            label=r"$e^{-\kappa t}$",
        )

    ax.set_xlabel(r"$t_{relax}$ (ns)")
    ax.set_ylabel(r"residual photons $n_0$")
    #ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, output_path)


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    setup = get_setup(platform="2q_chip_thesis", qubit=1)

    print(f"probe at midpoint      {setup.f_probe_mhz:.6f} MHz")
    print(f"kappa/2pi = {setup.kappa_mhz:.4f} MHz   chi/2pi = {setup.chi_mhz:.4f} MHz")
    print(f"n_target = {setup.n_target:g}  ->  amplitude {setup.amplitude_for_n(setup.n_target):.4f}")
    print(f"M1 exposure {setup.exposure_ns(0.0):.0f} ns, T1 = {setup.t1_ns:.0f} ns")

    t_relax = np.arange(0, 1200, 50, dtype=float)
    t_ramsey = np.arange(0.0, 600.0, 6.0)

    for rectangular in (True, False):
        name = "square" if rectangular else "clear"
        npz_path = setup.data_dir / f"run_{'square' if rectangular else 'clear'}.npz"

        if not npz_path.exists():
            npz_path = run_experiment(
                setup, t_relax, t_ramsey, rectangular=rectangular, output_name="run"
            )

        print(f"\n=== {name} M1 ===")
        results = analyze_run(npz_path, setup)
        plot_n0_decay(results, setup, setup.data_dir / f"n0_vs_t_relax_{name}.png")