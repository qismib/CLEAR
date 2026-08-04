""" CLEAR-pulse photon-population exp, following:
    D. T. McClure et al., "Rapid Driven Reset of a Qubit Readout Resonator", Phys. Rev. Applied 5, 011001 (2016).

Sequence:
    qubit preparation -> CLEAR M1 -> t_relax -> Ramsey -> t_buffer -> M2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


@dataclass(frozen=True)
class ClearPulseParameters:
    """Parameters for the CLEAR pulse.
    Absolute value of amplitudes <= 1.
    """

    ringup_1: complex
    ringup_2: complex
    steady: complex
    ringdown_1: complex
    ringdown_2: complex
    t_kick: float = 150.0
    t_steady: float = 300.0

    @property
    def amplitudes(self) -> tuple[complex, complex, complex, complex, complex]:
        return (
            self.ringup_1,
            self.ringup_2,
            self.steady,
            self.ringdown_1,
            self.ringdown_2,
        )

    @property
    def durations(self) -> tuple[float, float, float, float, float]:
        return (self.t_kick, self.t_kick, self.t_steady, self.t_kick, self.t_kick)


def rectangular_iq_pulse(duration: float, amplitude: complex) -> Pulse:
    """Creates a rectangular pulse from a complex amplitude."""

    magnitude = abs(amplitude)
    if magnitude > 1:
        raise ValueError(
            f"segment amplitude {magnitude} greater than 1. "
            "Rescale amplitude to be within [-1, 1] for the platform."
        )

    return Pulse(
        duration=duration,
        amplitude=float(magnitude),
        envelope=Rectangular(),
        relative_phase=float(np.angle(amplitude)),
    )


def clear_drive_sequence(platform, qubit: int, params: ClearPulseParameters) -> PulseSequence:
    """Creates the sequence of 5 rectangular pulses for CLEAR."""

    probe_channel = platform.qubits[qubit].probe
    sequence = PulseSequence()

    for duration, amplitude in zip(params.durations, params.amplitudes):
        if duration <= 0:
            continue
        sequence.append((probe_channel, rectangular_iq_pulse(duration, amplitude)))

    return sequence


def delay_sequence(channel, duration: float) -> PulseSequence:
    """Creates a delay sequence for a specific channel."""

    if duration <= 0:
        return PulseSequence()
    return PulseSequence([(channel, Delay(duration=duration))])


def build_clear_ramsey_sequence(
    platform,
    qubit: int,
    clear: ClearPulseParameters,
    t_relax: float,
    t_ramsey: float,
    t_buffer: float = 400.0,
    prepare_excited: bool = False,
) -> tuple[PulseSequence, Delay, int]:
    """Creates the sequence for the Ramsey-after-CLEAR test.

    returns:
        complete sequence of pulses.
        ramsey_delay.
        id of the acquisition pulse M2
    """

    native = platform.natives.single_qubit[qubit]
    drive_channel = platform.qubits[qubit].drive
    probe_channel = platform.qubits[qubit].probe
    acquisition_channel = platform.qubits[qubit].acquisition

    sequence = PulseSequence()

    if prepare_excited:
        sequence |= native.RX()

    # M1: CLEAR measurement/reset drive.
    
    sequence |= clear_drive_sequence(platform, qubit, clear)

    # Wait after M1 before probing residual photons with Ramsey.
    sequence |= delay_sequence(probe_channel, t_relax)

    # Ramsey: RX90 - wait t_R - RX90.
    sequence |= native.RX90()
    ramsey_delay = Delay(duration=t_ramsey)
    sequence |= PulseSequence([(drive_channel, ramsey_delay)])
    sequence |= native.RX90()

    #   Avoid corrupting the final measurement with photons coming from M1/Ramsey.
    sequence |= delay_sequence(drive_channel, t_buffer)

    # M2: standard square readout, already calibrated in the platform.
    m2 = native.MZ()
    m2_readout = list(m2.channel(acquisition_channel))[-1]
    sequence |= m2

    return sequence, ramsey_delay, m2_readout.id


def run_clear_ramsey_scan(
    platform_name: str,
    qubit: int,
    clear: ClearPulseParameters,
    t_relax_values: Iterable[float],
    t_ramsey_values: Iterable[float],
    output: Path,
    nshots: int = 1024,
    relaxation_time: float = 100_000,
    ramsey_detuning: float | None = 10_000_000,
    prepare_excited: bool = False,
) -> None:
    """ performs Ramsey-after-CLEAR scans and saves the integrated results."""

    output.mkdir(parents=True, exist_ok=True)
    t_relax_values = np.asarray(list(t_relax_values), dtype=float)
    t_ramsey_values = np.asarray(list(t_ramsey_values), dtype=float)
    platform = create_platform(platform_name)

    all_results = {}
    platform.connect()
    try:
        updates = []
        if ramsey_detuning is not None:
            drive_channel = platform.qubits[qubit].drive
            drive_frequency = platform.config(drive_channel).frequency
            updates.append({drive_channel: {"frequency": drive_frequency + ramsey_detuning}})

        for t_relax in t_relax_values:
            sequence, ramsey_delay, m2_id = build_clear_ramsey_sequence(
                platform=platform,
                qubit=qubit,
                clear=clear,
                t_relax=float(t_relax),
                t_ramsey=float(t_ramsey_values[0]),
                prepare_excited=prepare_excited,
            )

            sweeper = Sweeper(
                parameter=Parameter.duration,
                values=t_ramsey_values,
                pulses=[ramsey_delay],
            )

            results = platform.execute(
                [sequence],
                [[sweeper]],
                nshots=nshots,
                relaxation_time=relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.SINGLESHOT,
                updates=updates,
            )

            all_results[f"t_relax_{t_relax:g}"] = results[m2_id]

    finally:
        platform.disconnect()

    np.savez(
        output / ("clear_ramsey_excited.npz" if prepare_excited else "clear_ramsey_ground.npz"),
        t_relax=t_relax_values,
        t_ramsey=t_ramsey_values,
        **all_results,
    )


if __name__ == "__main__":
    # substitute values and amplitudes of CLEAR
    PLATFORM = "YOUR_PLATFORM_NAME"
    QUBIT = 0

   
    clear_parameters = ClearPulseParameters(
        ringup_1=0.10 + 0.0j,
        ringup_2=0.08 + 0.0j,
        steady=0.05 + 0.0j,
        ringdown_1=-0.08 + 0.0j,
        ringdown_2=-0.10 + 0.0j,
        t_kick=150.0,
        t_steady=600.0,
    )

    run_clear_ramsey_scan(
        platform_name=PLATFORM,
        qubit=QUBIT,
        clear=clear_parameters,
        t_relax_values=np.array([0.0, 40.0, 80.0, 160.0, 320.0]),
        t_ramsey_values=np.arange(0.0, 600.0 + 1.0, 8.0),
        output=Path("output/clear_test/q0"),
        nshots=1024,
        relaxation_time=100_000,
        ramsey_detuning=10_000_000,
        prepare_excited=False,
    )
