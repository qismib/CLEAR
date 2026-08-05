""" Reproduces Fig. 3(c) of:
    D. T. McClure et al., "Rapid Driven Reset of a Qubit Readout Resonator",
    Phys. Rev. Applied 5, 011001 (2016).

Fig. 3(c): residual cavity population n0 versus normalized drive power
P_norm, for a square (+delay) measurement pulse and for the CLEAR pulse,
with the qubit prepared in the ground or excited state.

This script:
    1. Loads the CLEAR segment amplitude *ratios* (each segment relative to
       the steady-state readout amplitude) exported by the Julia notebook
       `amplitudes_CLEAR.ipynb` (see its "6. Esportazione dei parametri
       CLEAR per qibolab" section) into `parametri/clear_amplitudes.json`.
    2. For a scan of normalized drive powers P_norm, builds and runs, on
       hardware via qibolab, a Ramsey-after-M1 sequence (t_relax = 0) for
       both pulse shapes (square+delay, CLEAR) and both qubit states
       (ground, excited) -- reusing the building blocks of CLEAR_qibolab.py.
    3. Fits Eq. (1) (see CLEAR_qibolab.ramsey_signal / fit_ramsey_eq1) to
       every trace to extract n0, and plots n0 vs P_norm exactly as in
       Fig. 3(c).

Sequence, for each P_norm / pulse type / qubit state (t_relax = 0 for both,
as in the paper: "For the CLEAR pulse, the Ramsey experiment begins
immediately at the end of the pulse, while for the square pulse, a delay of
approximately 300 ns is inserted to match the total length of the CLEAR
pulse's two ring-down segments."):

    qubit preparation -> M1 (square+delay OR CLEAR) -> Ramsey (t_relax=0)
        -> t_buffer -> M2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np

from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
    create_platform,
)

from CLEAR_qibolab import (
    ClearPulseParameters,
    clear_drive_sequence,
    delay_sequence,
    rectangular_iq_pulse,
    ramsey_signal,
    fit_ramsey_eq1,
    normalized_amplitude,
    mhz_to_rad_per_ns,
)


# ---------------------------------------------------------------------------
# 1. CLEAR amplitude ratios exported by amplitudes_CLEAR.ipynb
# ---------------------------------------------------------------------------


def load_clear_ratios(json_path: Path) -> dict:
    """Loads the CLEAR segment amplitude ratios (each segment relative to
    the steady-state readout amplitude) exported by `amplitudes_CLEAR.ipynb`.

    Since the CLEAR kick amplitudes are linear in the readout drive
    amplitude A in the linear resonator model, `ringup_i / A` and
    `ringdown_i / A` do not depend on A (nor on the arbitrary units A is
    expressed in inside the Julia model) -- only on kappa, chi, detuning and
    t_kick. They can therefore be reused directly with the hardware
    (qibolab-normalized) readout amplitude.

    Expected JSON structure (written by the notebook):
        {
          "t_kick_us": ...,
          "t_readout_us": ...,
          "ratios": {
            "ringup_1": [re, im],
            "ringup_2": [re, im],
            "steady":   [re, im],   # == [1.0, 0.0] by construction
            "ringdown_1": [re, im],
            "ringdown_2": [re, im]
          }
        }
    """

    with open(json_path) as f:
        data = json.load(f)

    ratios = {key: complex(re, im) for key, (re, im) in data["ratios"].items()}

    return {
        "ratios": ratios,
        "t_kick_ns": data["t_kick_us"] * 1e3,
        "t_readout_ns": data["t_readout_us"] * 1e3,
    }


def clear_params_from_ratios(
    ratios: dict,
    readout_amplitude: complex,
    t_kick_ns: float,
    t_readout_ns: float,
) -> ClearPulseParameters:
    """Builds ClearPulseParameters for a given hardware readout amplitude,
    scaling every CLEAR segment by the ratios from the linear model."""

    return ClearPulseParameters(
        ringup_1=ratios["ringup_1"] * readout_amplitude,
        ringup_2=ratios["ringup_2"] * readout_amplitude,
        steady=ratios["steady"] * readout_amplitude,
        ringdown_1=ratios["ringdown_1"] * readout_amplitude,
        ringdown_2=ratios["ringdown_2"] * readout_amplitude,
        t_kick=t_kick_ns,
        t_steady=t_readout_ns,
    )


# ---------------------------------------------------------------------------
# 2. Sequence building: square+delay vs CLEAR, followed by Ramsey (t_relax=0)
# ---------------------------------------------------------------------------


def square_drive_sequence(
    platform, qubit: int, amplitude: complex, duration: float
) -> PulseSequence:
    """Square (rectangular) M1 pulse with the given complex amplitude and
    duration, on the probe channel."""

    probe_channel = platform.qubits[qubit].probe
    sequence = PulseSequence()
    if duration > 0:
        sequence.append((probe_channel, rectangular_iq_pulse(duration, amplitude)))
    return sequence


def build_fig3c_sequence(
    platform,
    qubit: int,
    pulse_type: Literal["square", "clear"],
    readout_amplitude: complex,
    ratios: dict,
    t_kick_ns: float,
    t_readout_ns: float,
    t_ramsey: float,
    t_buffer: float = 400.0,
    prepare_excited: bool = False,
) -> tuple[PulseSequence, Delay, int]:
    """Builds the M1 -> Ramsey(t_relax=0) -> M2 sequence of Fig. 3(c), for
    either a square pulse (+ ring-down-matching delay) or a CLEAR pulse.

    returns:
        complete sequence of pulses.
        ramsey_delay (to be swept over t_ramsey_values).
        id of the acquisition pulse M2.
    """

    native = platform.natives.single_qubit[qubit]
    drive_channel = platform.qubits[qubit].drive
    probe_channel = platform.qubits[qubit].probe
    acquisition_channel = platform.qubits[qubit].acquisition

    # Length of CLEAR's two ring-down segments, used both to size the CLEAR
    # pulse and to size the delay appended after the square pulse.
    t_ringdown_total = 2 * t_kick_ns

    sequence = PulseSequence()

    if prepare_excited:
        sequence |= native.RX()

    # --- M1: CLEAR or square + matching delay ---------------------------
    if pulse_type == "clear":
        clear = clear_params_from_ratios(
            ratios, readout_amplitude, t_kick_ns, t_readout_ns
        )
        sequence |= clear_drive_sequence(platform, qubit, clear)
        # Ramsey starts immediately at the end of the CLEAR pulse (t_relax=0).
    elif pulse_type == "square":
        # Same steady-state amplitude, and same total length as the CLEAR
        # pulse up to (but not including) the ring-down segments.
        t_square = 2 * t_kick_ns + t_readout_ns
        sequence |= square_drive_sequence(platform, qubit, readout_amplitude, t_square)
        # Delay equivalent to the CLEAR ring-down segments, to match the
        # total M1 length ("a delay of approximately 300 ns is inserted to
        # match the total length of the CLEAR pulse's two ring-down
        # segments").
        sequence |= delay_sequence(probe_channel, t_ringdown_total)
    else:
        raise ValueError(f"Unknown pulse_type {pulse_type!r}")

    # --- Ramsey: RX90 - wait t_R - RX90, immediately after M1 -----------
    sequence |= native.RX90()
    ramsey_delay = Delay(duration=t_ramsey)
    sequence |= PulseSequence([(drive_channel, ramsey_delay)])
    sequence |= native.RX90()

    #   Avoid corrupting M2 with lingering photons from M1/Ramsey.
    sequence |= delay_sequence(drive_channel, t_buffer)

    # --- M2: standard square readout, already calibrated in the platform.
    m2 = native.MZ()
    m2_readout = list(m2.channel(acquisition_channel))[-1]
    sequence |= m2

    return sequence, ramsey_delay, m2_readout.id


# ---------------------------------------------------------------------------
# 3. Power scan acquisition
# ---------------------------------------------------------------------------


def run_fig3c_scan(
    platform_name: str,
    qubit: int,
    clear_json_path: Path,
    pnorm_values: Iterable[float],
    t_ramsey_values: Iterable[float],
    amplitude_1ph: float,
    output: Path,
    nshots: int = 1024,
    relaxation_time: float = 100_000,
    ramsey_detuning: float | None = 10_000_000,
) -> Path:
    """Runs the Fig. 3(c) power scan: for each P_norm, both pulse types
    (square+delay, CLEAR) and both qubit states (ground, excited), acquires
    a Ramsey trace at t_relax=0 and saves everything to a single .npz file.

    `amplitude_1ph` is the (real, hardware-normalized, in [-1, 1]) amplitude
    of the steady-state readout segment that yields n=1 photon (P_norm=1),
    obtained from an independent calibration (a standard Ramsey/Stark-shift
    measurement, as in the paper). Since P_norm = P/P_1ph is a *power*
    (i.e. proportional to |amplitude|^2, like a photon number), the
    amplitude at a given P_norm is amplitude_1ph * sqrt(P_norm); the CLEAR
    ratios (linear in the field amplitude) are then applied on top of that.

    Returns the path of the saved .npz file.
    """

    output.mkdir(parents=True, exist_ok=True)
    pnorm_values = np.asarray(list(pnorm_values), dtype=float)
    t_ramsey_values = np.asarray(list(t_ramsey_values), dtype=float)

    clear_data = load_clear_ratios(clear_json_path)
    ratios = clear_data["ratios"]
    t_kick_ns = clear_data["t_kick_ns"]
    t_readout_ns = clear_data["t_readout_ns"]

    platform = create_platform(platform_name)

    all_results = {}
    platform.connect()
    try:
        updates = []
        if ramsey_detuning is not None:
            drive_channel = platform.qubits[qubit].drive
            drive_frequency = platform.config(drive_channel).frequency
            updates.append({drive_channel: {"frequency": drive_frequency + ramsey_detuning}})

        for pulse_type in ("square", "clear"):
            for prepare_excited in (False, True):
                for pnorm in pnorm_values:
                    readout_amplitude = amplitude_1ph * np.sqrt(pnorm)

                    sequence, ramsey_delay, m2_id = build_fig3c_sequence(
                        platform=platform,
                        qubit=qubit,
                        pulse_type=pulse_type,
                        readout_amplitude=readout_amplitude,
                        ratios=ratios,
                        t_kick_ns=t_kick_ns,
                        t_readout_ns=t_readout_ns,
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

                    state = "excited" if prepare_excited else "ground"
                    key = f"{pulse_type}_{state}_pnorm_{pnorm:g}"
                    all_results[key] = results[m2_id]

    finally:
        platform.disconnect()

    npz_path = output / "clear_fig3c.npz"
    np.savez(npz_path, pnorm=pnorm_values, t_ramsey=t_ramsey_values, **all_results)

    return npz_path


# ---------------------------------------------------------------------------
# 4. Analysis: Eq. (1) fit for every trace, Fig. 3(c)-style plot
# ---------------------------------------------------------------------------


_FIG3C_STYLES = {
    ("square", "ground"): dict(
        color="tab:blue", marker="o", linestyle="--",
        label="Square + delay (ground)",
    ),
    ("square", "excited"): dict(
        color="tab:red", marker="o", linestyle="--",
        label="Square + delay (excited)",
    ),
    ("clear", "ground"): dict(
        color="tab:blue", marker="s", linestyle="-",
        label="CLEAR (ground)",
    ),
    ("clear", "excited"): dict(
        color="tab:red", marker="s", linestyle="-",
        label="CLEAR (excited)",
    ),
}


def analyze_fig3c(
    npz_path: Path,
    kappa_mhz: float,
    chi_mhz: float,
    ramsey_detuning_mhz: float,
    t2_echo_ns: float,
    output_path: Path | None = None,
) -> dict:
    """Fits Eq. (1) to every trace saved by `run_fig3c_scan` and reproduces
    Fig. 3(c): n0 vs P_norm, for square+delay and CLEAR pulses, ground and
    excited qubit states.

    Returns a dict {(pulse_type, state): n0_array}, in the same order as
    `pnorm_values`.
    """

    kappa = mhz_to_rad_per_ns(kappa_mhz)
    chi = mhz_to_rad_per_ns(chi_mhz)
    detuning = mhz_to_rad_per_ns(ramsey_detuning_mhz)
    gamma2 = 1.0 / t2_echo_ns

    data = np.load(npz_path)
    pnorm_values = data["pnorm"]
    t_ramsey = data["t_ramsey"]

    n0_curves = {}
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for (pulse_type, state), style in _FIG3C_STYLES.items():
        n0_values = np.full(pnorm_values.shape, np.nan)
        for i, pnorm in enumerate(pnorm_values):
            key = f"{pulse_type}_{state}_pnorm_{pnorm:g}"
            if key not in data:
                continue
            amplitude = normalized_amplitude(data[key])
            fit_result = fit_ramsey_eq1(t_ramsey, amplitude, kappa, chi, detuning, gamma2)
            n0_values[i] = fit_result["n0"]

        n0_curves[(pulse_type, state)] = n0_values
        ax.plot(pnorm_values, n0_values, **style)

    ax.set_xlabel(r"Normalized drive power $P_{\mathrm{norm}}$")
    ax.set_ylabel(r"Residual population $n_0$")
    ax.set_title("Fig. 3(c): residual population vs drive power")
    ax.legend(loc="best")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200)

    for (pulse_type, state), n0_values in n0_curves.items():
        print(f"{pulse_type:6s} {state:8s} n0 = {np.round(n0_values, 3)}")

    return n0_curves


def _demo_synthetic_fig3c(
    ratios: dict,
    t_kick_ns: float,
    t_readout_ns: float,
    output_path: Path | None = None,
) -> dict:
    """Demonstrates the Fig. 3(c) pipeline (power scan -> Eq. 1 fit -> plot)
    with synthetic data, for use before real hardware data is available.

    For each pulse type and qubit state, synthetic Ramsey traces are
    generated directly from Eq. (1), with an n0(P_norm) curve chosen to
    mimic the qualitative behavior of Fig. 3(c) (roughly linear growth of
    n0 with P_norm for the square pulse, and a much flatter, near-zero
    curve for the CLEAR pulse in the linear-response regime), plus shot
    noise. The same `analyze_fig3c`-style fit is then run on top of it.
    """

    kappa = mhz_to_rad_per_ns(1.1)
    chi = mhz_to_rad_per_ns(0.5)
    detuning = mhz_to_rad_per_ns(10.0)
    gamma2 = 1.0 / 10_000.0

    pnorm_values = np.linspace(0.0, 10.0, 11)
    t_ramsey = np.arange(0.0, 600.0 + 1.0, 8.0)

    # Illustrative "true" n0(P_norm) curves, only for the synthetic demo.
    n0_true_curves = {
        ("square", "ground"): 0.18 * pnorm_values,
        ("square", "excited"): 0.15 * pnorm_values,
        ("clear", "ground"): 0.01 * pnorm_values,
        ("clear", "excited"): 0.01 * pnorm_values,
    }

    rng = np.random.default_rng(0)
    all_data = {}
    for (pulse_type, state), n0_curve in n0_true_curves.items():
        for pnorm, n0_true in zip(pnorm_values, n0_curve):
            amplitude = ramsey_signal(
                t_ramsey, n0_true, 0.3, kappa, chi, detuning, gamma2
            ) + 0.03 * rng.standard_normal(t_ramsey.size)
            key = f"{pulse_type}_{state}_pnorm_{pnorm:g}"
            all_data[key] = amplitude[None, :]  # fake single "shot" axis

    npz_path = Path("clear_fig3c_demo.npz")
    np.savez(npz_path, pnorm=pnorm_values, t_ramsey=t_ramsey, **all_data)

    print("Demo con dati sintetici (nessun dato sperimentale ancora disponibile):")
    return analyze_fig3c(
        npz_path=npz_path,
        kappa_mhz=1.1,
        chi_mhz=0.5,
        ramsey_detuning_mhz=10.0,
        t2_echo_ns=10_000.0,
        output_path=output_path,
    )


if __name__ == "__main__":
    PLATFORM = "YOUR_PLATFORM_NAME"
    QUBIT = 0

    output = Path("output/clear_test/q0")
    clear_json_path = Path("parametri/clear_amplitudes.json")
    npz_path = output / "clear_fig3c.npz"

    if npz_path.exists():
        analyze_fig3c(
            npz_path=npz_path,
            kappa_mhz=1.1,
            chi_mhz=0.5,
            ramsey_detuning_mhz=10.0,
            t2_echo_ns=10_000.0,
            output_path=output / "fig3c_n0_vs_pnorm.png",
        )
    elif clear_json_path.exists():
        # Real amplitude ratios are available, but the scan has not been run
        # yet: substitute amplitude_1ph with your calibrated value (the
        # readout amplitude, in [-1, 1] hardware units, giving n=1) and run
        # the scan on hardware.
        npz_path = run_fig3c_scan(
            platform_name=PLATFORM,
            qubit=QUBIT,
            clear_json_path=clear_json_path,
            pnorm_values=np.linspace(0.0, 10.0, 11),
            t_ramsey_values=np.arange(0.0, 600.0 + 1.0, 8.0),
            amplitude_1ph=0.1,
            output=output,
        )
        analyze_fig3c(
            npz_path=npz_path,
            kappa_mhz=1.1,
            chi_mhz=0.5,
            ramsey_detuning_mhz=10.0,
            t2_echo_ns=10_000.0,
            output_path=output / "fig3c_n0_vs_pnorm.png",
        )
    else:
        # Neither the ratios from amplitudes_CLEAR.ipynb nor real data are
        # available yet: run the pipeline on synthetic data to illustrate
        # the Fig. 3(c) plot and the Eq. (1) fit.
        _demo_synthetic_fig3c(
            ratios={}, t_kick_ns=150.0, t_readout_ns=600.0,
            output_path=output / "fig3c_n0_vs_pnorm_demo.png",
        )