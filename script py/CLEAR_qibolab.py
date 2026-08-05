""" CLEAR-pulse photon-population exp, following:
    D. T. McClure et al., "Rapid Driven Reset of a Qubit Readout Resonator", Phys. Rev. Applied 5, 011001 (2016).

Sequence (Fig. 1(b) of the paper):
    qubit preparation -> M1 -> t_relax -> Ramsey (RX90 - t_R - RX90) -> t_buffer -> M2.

M1 is built by `clear_drive_sequence`:
    - rectangular=False (default): the full 5-segment CLEAR pulse of
      Fig. 1(a) (ring-up x2, steady-state readout, ring-down x2).
    - rectangular=True: only the steady-state segment is kept, i.e. a plain
      square pulse of duration `t_steady` and amplitude `steady` -- the
      paper's square-pulse baseline used for comparison in Figs. 2(b)/2(c)
      and 3(c). The rest of the sequence (preparation, t_relax, Ramsey,
      t_buffer, M2) is identical in both cases, so switching `rectangular`
      reproduces the same experiment with the two different M1 pulse
      shapes compared in the paper, with parameters that can differ from
      the ones used there (kappa, chi, powers, etc., set via
      parametri/risonatore.toml and the calibration constants below).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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

DATA_FOLDER = "../qibocal/data/clear_test"

# ---------------------------------------------------------------------------
# Paths to the calibration files produced by stark_analysis.py and used by
# amplitudes_CLEAR.ipynb (see its Section 1 "Parameters" and Section 6
# "Esportazione dei parametri CLEAR per qibolab"):
#   - data/stark_fit_results.json  -> {"k": ..., "q": ...}, saved by
#     stark_analysis.py as `FOLDER + "stark_fit_results.json"`, with
#     FOLDER = "data/" *relative to the directory stark_analysis.py is run
#     from*. stark_analysis.py and CLEAR_qibolab.py live in the same
#     directory (e.g. ".../CLEAR/script py/"), so here we mirror that same
#     convention: "data/" relative to this script's own directory.
#   - parametri/risonatore.toml    -> resonator/pulse parameters, incl.
#     n_target (Section 1 of amplitudes_CLEAR.ipynb).
# If your local layout differs (e.g. "data/" or "parametri/" live one level
# up, next to "script py/" rather than inside it), adjust PROJECT_DIR below
# accordingly (e.g. `.parent.parent` instead of `.parent`).
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
RESONATOR_TOML_PATH = PROJECT_DIR / "parametri" / "risonatore.toml"
STARK_FIT_JSON_PATH = PROJECT_DIR / "data" / "stark_fit_results.json"


def _parse_simple_toml(toml_path: Path) -> dict:
    """Minimal fallback parser for flat `key = value` TOML files (no
    tables/sections), used only if neither `tomllib` (Python >= 3.11) nor
    `tomli` (`pip install tomli`) is available. Sufficient for
    parametri/risonatore.toml, which has no nested tables.
    """

    result: dict = {}
    with open(toml_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not value:
                raise ValueError(
                    f"Empty value for key {key!r} in {toml_path} "
                    "(fallback TOML parser cannot handle this; either fill "
                    "in a value or install tomli: `pip install tomli`)."
                )
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                result[key] = value[1:-1]
            elif value.lower() in ("true", "false"):
                result[key] = value.lower() == "true"
            else:
                try:
                    result[key] = int(value)
                except ValueError:
                    result[key] = float(value)
    return result


def load_resonator_toml(toml_path: Path = RESONATOR_TOML_PATH) -> dict:
    """Loads parametri/risonatore.toml (same file used by
    amplitudes_CLEAR.ipynb via TOML.parsefile)."""

    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # requires `pip install tomli`
        except ModuleNotFoundError:
            return _parse_simple_toml(toml_path)

    with open(toml_path, "rb") as f:
        return tomllib.load(f)


def load_stark_fit_k(json_path: Path = STARK_FIT_JSON_PATH) -> float:
    """Loads the k coefficient (angular coefficient of the linear fit
    n_photons = k * amp^2 + q) produced and saved by stark_analysis.py, so
    that it is not re-typed by hand and always matches the latest Stark
    shift calibration."""

    with open(json_path) as f:
        data = json.load(f)
    return float(data["k"])

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


def clear_drive_sequence(platform, qubit: int, params: ClearPulseParameters, rectangular=False) -> PulseSequence:
    """Creates the M1 measurement/reset drive sequence.

    If `rectangular` is False (default), builds the full 5-segment CLEAR
    pulse (two ring-up kicks, the steady-state readout segment, two
    ring-down kicks), exactly as in Fig. 1(a) of McClure et al. (2016).

    If `rectangular` is True, only the steady-state segment (index 2, the
    central readout amplitude) is emitted, i.e. a plain rectangular/square
    M1 pulse of duration `params.t_steady` and amplitude `params.steady` is
    used instead of CLEAR -- this reproduces the paper's "square pulse"
    baseline (Figs. 2 and 3), driven with the same overall Ramsey-after-M1
    sequence structure as the CLEAR case (see build_clear_ramsey_sequence).
    """

    probe_channel = platform.qubits[qubit].probe
    sequence = PulseSequence()
    segment_idx = 0
    for duration, amplitude in zip(params.durations, params.amplitudes):
        if duration <= 0:
            continue
        if not rectangular:
            sequence.append((probe_channel, rectangular_iq_pulse(duration, amplitude)))
        elif segment_idx == 2:
            sequence.append((probe_channel, rectangular_iq_pulse(duration, amplitude)))
        segment_idx += 1
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
    rectangular : bool = False
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
    sequence |= clear_drive_sequence(platform, qubit, clear, rectangular=rectangular)

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
    rectangular: bool = False
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
                rectangular=rectangular
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


# ---------------------------------------------------------------------------
# Ramsey trace analysis: Eq. (1) model, fit and Fig. 2(a)-style plot.
#
# Eq. (1) of McClure et al. (2016):
#   S(t_R) = 1/2 [1 - Im{exp[-(Gamma2 + i*Delta)*t_R + i*(phi0 - 2*n0*chi*tau)]}]
#   tau(t_R) = (1 - exp(-(kappa + 2i*chi)*t_R)) / (kappa + 2i*chi)
#
# Delta = Ramsey detuning, Gamma2 = 1/T2_echo, phi0 = initial phase,
# n0 = cavity population at the start of the Ramsey delay.
# kappa, chi, Delta, Gamma2 are kept fixed; the only free fit parameters
# are n0 and phi0, as in the paper.
#
# This script uses nanoseconds (consistent with `t_kick`, `t_ramsey_values`,
# etc. above), so kappa/chi/Delta/Gamma2 are expected in rad/ns.
# ---------------------------------------------------------------------------


def mhz_to_rad_per_ns(f_mhz: float) -> float:
    """Converts a frequency in MHz to an angular frequency in rad/ns."""
    return 2 * np.pi * f_mhz * 1e-3


def ramsey_tau(t_r: np.ndarray, kappa: float, chi: float) -> np.ndarray:
    """tau(t_R), as defined below Eq. (1). kappa, chi in rad/ns, t_r in ns."""
    lam = kappa + 2j * chi
    return (1 - np.exp(-lam * t_r)) / lam


def ramsey_signal(
    t_r: np.ndarray,
    n0: float,
    phi0: float,
    kappa: float,
    chi: float,
    detuning: float,
    gamma2: float,
) -> np.ndarray:
    """Eq. (1): normalized Ramsey amplitude S(t_R).

    kappa, chi, detuning, gamma2 in rad/ns; t_r in ns.
    """
    tau = ramsey_tau(np.asarray(t_r, dtype=float), kappa, chi)
    expo = -(gamma2 + 1j * detuning) * t_r + 1j * (phi0 - 2 * n0 * chi * tau)
    return 0.5 * (1 - np.imag(np.exp(expo)))


def fit_ramsey_eq1(
    t_r: np.ndarray,
    amplitude: np.ndarray,
    kappa: float,
    chi: float,
    detuning: float,
    gamma2: float,
    p0: tuple[float, float] = (0.5, 0.0),
) -> dict:
    """Fits n0 and phi0 of Eq. (1) to a measured (or simulated) Ramsey trace.

    kappa, chi, detuning and gamma2 are held fixed, as in the paper
    ("the only free parameters are n0 and phi0").
    """

    def model(t_r, n0, phi0):
        return ramsey_signal(t_r, n0, phi0, kappa, chi, detuning, gamma2)

    popt, pcov = curve_fit(model, t_r, amplitude, p0=p0)
    perr = np.sqrt(np.diag(pcov))
    return {
        "n0": float(popt[0]),
        "phi0": float(popt[1]),
        "n0_err": float(perr[0]),
        "phi0_err": float(perr[1]),
        "popt": popt,
        "pcov": pcov,
    }


def normalized_amplitude(iq: np.ndarray) -> np.ndarray:
    """Averages single-shot IQ points over shots and rescales the resulting
    trace to [0, 1], to match the 'Normalized amplitude' of Fig. 2(a).

    `iq` is expected to have shape (nshots, len(t_ramsey)) with a complex
    dtype, as returned for a single t_relax by `run_clear_ramsey_scan`
    (AveragingMode.SINGLESHOT). Some qibolab versions instead save the I/Q
    components as a real array of shape (nshots, len(t_ramsey), 2); that
    case is converted to a complex (nshots, len(t_ramsey)) array first.
    """

    iq = np.asarray(iq)

    if not np.iscomplexobj(iq) and iq.ndim == 3 and iq.shape[-1] == 2:
        # (nshots, len(t_ramsey), 2) -> complex (nshots, len(t_ramsey))
        iq = iq[..., 0] + 1j * iq[..., 1]

    trace = np.real(iq) if np.iscomplexobj(iq) else iq

    if trace.ndim != 2:
        raise ValueError(
            f"normalized_amplitude expected a 2D (nshots, len(t_ramsey)) "
            f"array after IQ handling, got shape {trace.shape}. Check the "
            "shape of the raw data saved by run_clear_ramsey_scan."
        )

    trace = np.mean(trace, axis=0)
    trace = trace - trace.min()
    if trace.max() > 0:
        trace = trace / trace.max()
    return trace


def load_ramsey_trace(
    npz_path: Path, t_relax: float
) -> tuple[np.ndarray, np.ndarray]:
    """Loads a Ramsey-after-CLEAR trace saved by `run_clear_ramsey_scan`, for
    a given t_relax, and returns (t_ramsey, normalized_amplitude)."""

    data = np.load(npz_path)
    t_ramsey = data["t_ramsey"]
    key = f"t_relax_{t_relax:g}"
    amplitude = normalized_amplitude(data[key])
    return t_ramsey, amplitude


def plot_ramsey_fig2a(
    t_r_data: np.ndarray,
    amplitude_data: np.ndarray,
    fit_result: dict,
    kappa: float,
    chi: float,
    detuning: float,
    gamma2: float,
    title: str = "Fig. 2(a): Ramsey experiment and fit",
    output_path: Path | None = None,
):
    """Reproduces the style of Fig. 2(a): red circles for the data, solid
    black curve for the Eq. (1) fit."""

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(t_r_data, amplitude_data, color="red", s=20, label="Data")

    t_r_fit = np.linspace(t_r_data.min(), t_r_data.max(), 400)
    s_fit = ramsey_signal(
        t_r_fit, fit_result["n0"], fit_result["phi0"], kappa, chi, detuning, gamma2
    )
    ax.plot(t_r_fit, s_fit, color="black", linewidth=2, label="Fit of Eq. (1)")

    ax.set_xlabel(r"Ramsey delay $t_R$ (ns)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200)

    return fig, ax


def analyze_ramsey_fig2a(
    npz_path: Path,
    t_relax: float,
    kappa_mhz: float,
    chi_mhz: float,
    ramsey_detuning_mhz: float,
    t2_echo_ns: float,
    output_path: Path | None = None,
) -> dict:
    """End-to-end analysis: loads a saved Ramsey-after-CLEAR trace, fits
    Eq. (1) to extract n0 and phi0, plots the Fig. 2(a)-style figure, and
    returns the fit result (including n0).
    """

    kappa = mhz_to_rad_per_ns(kappa_mhz)
    chi = mhz_to_rad_per_ns(chi_mhz)
    detuning = mhz_to_rad_per_ns(ramsey_detuning_mhz)
    gamma2 = 1.0 / t2_echo_ns

    t_r_data, amplitude_data = load_ramsey_trace(npz_path, t_relax)

    fit_result = fit_ramsey_eq1(
        t_r_data, amplitude_data, kappa, chi, detuning, gamma2
    )

    plot_ramsey_fig2a(
        t_r_data, amplitude_data, fit_result, kappa, chi, detuning, gamma2,
        output_path=output_path,
    )

    print(f"n0   = {fit_result['n0']:.4f} +/- {fit_result['n0_err']:.4f}")
    print(f"phi0 = {fit_result['phi0']:.4f} +/- {fit_result['phi0_err']:.4f}")

    return fit_result


def _demo_synthetic_fit(output_path: Path | None = None) -> dict:
    """Demonstrates the Fig. 2(a) plot and Eq. (1) fit with synthetic data,
    for use before real hardware data (clear_ramsey_*.npz) is available.

    Synthetic single-shot data is generated from Eq. (1) itself for known
    "true" n0, phi0, plus shot noise, then fitted back -- exactly the
    pipeline `analyze_ramsey_fig2a` runs on real data via `load_ramsey_trace`.
    """

    kappa = mhz_to_rad_per_ns(1.1)          # resonator linewidth, kappa/2pi = 1.1 MHz
    chi = mhz_to_rad_per_ns(0.5)            # dispersive shift, chi/2pi = 0.5 MHz (example)
    detuning = mhz_to_rad_per_ns(10.0)      # Ramsey detuning, 10 MHz (as in the paper)
    gamma2 = 1.0 / 10_000.0                 # 1/T2_echo, T2_echo = 10 us = 10000 ns (example)

    n0_true, phi0_true = 0.9, 0.3
    t_r_data = np.arange(0.0, 600.0 + 1.0, 8.0)  # ns, as in run_clear_ramsey_scan

    rng = np.random.default_rng(0)
    amplitude_data = ramsey_signal(
        t_r_data, n0_true, phi0_true, kappa, chi, detuning, gamma2
    ) + 0.03 * rng.standard_normal(t_r_data.size)

    fit_result = fit_ramsey_eq1(t_r_data, amplitude_data, kappa, chi, detuning, gamma2)

    plot_ramsey_fig2a(
        t_r_data, amplitude_data, fit_result, kappa, chi, detuning, gamma2,
        title="Fig. 2(a): Ramsey experiment and fit (synthetic demo data)",
        output_path=output_path,
    )

    print("Demo con dati sintetici (nessun dato sperimentale ancora disponibile):")
    print(f"n0_true  = {n0_true}, phi0_true = {phi0_true}")
    print(f"n0_fit   = {fit_result['n0']:.4f} +/- {fit_result['n0_err']:.4f}")
    print(f"phi0_fit = {fit_result['phi0']:.4f} +/- {fit_result['phi0_err']:.4f}")

    return fit_result


if __name__ == "__main__":
    # substitute values and amplitudes of CLEAR
    PLATFORM = "sqps_thesis"
    QUBIT = 0

    # n_target: dal file di calibrazione parametri/risonatore.toml (lo
    # stesso letto in Sezione 1 di amplitudes_CLEAR.ipynb).
    resonator_params = load_resonator_toml()
    n_target = resonator_params["n_target"]

    # k: coefficiente angolare del fit lineare n_photons = k*amp^2 + q,
    # prodotto da stark_analysis.py e salvato in data/stark_fit_results.json
    # -- stessa quantita' letta in Sezione 1 del notebook, cosi' che
    # A_readout usi sempre l'ultima calibrazione di Stark shift.
    k = 5.194e4#load_stark_fit_k()

    A_readout = float(np.sqrt(n_target / k))

    clear_parameters = ClearPulseParameters(
        ringup_1=0.10 + 0.0j,
        ringup_2=0.08 + 0.0j,
        steady=A_readout + 0.0j,
        ringdown_1=-0.08 + 0.0j,
        ringdown_2=-0.10 + 0.0j,
        t_kick=150.0,
        t_steady=600.0,
    )

    run_clear_ramsey_scan(
        platform_name=PLATFORM,
        qubit=QUBIT,
        clear=clear_parameters,
        t_relax_values=np.array([100]),
        t_ramsey_values=np.arange(0.0, 600.0 + 1.0, 8.0),
        output=Path(DATA_FOLDER + ""),
        nshots=1024,
        relaxation_time=100_000,
        ramsey_detuning=10_000_000,
        prepare_excited=False,
        rectangular=True
    )

    # --- Fig. 2(a): Ramsey trace and Eq. (1) fit, extracting n0 ---------
    # Substitute kappa_mhz, chi_mhz and t2_echo_ns with the values from your
    # calibration once available.
    npz_path = Path(DATA_FOLDER + "/clear_ramsey_ground.npz")
    if npz_path.exists():
        analyze_ramsey_fig2a(
            npz_path=npz_path,
            t_relax=0.0,
            kappa_mhz=10.7,
            chi_mhz=0.5,
            ramsey_detuning_mhz=10.0,
            t2_echo_ns=10_000.0,
            output_path=Path(DATA_FOLDER + "/fig2a_ramsey_fit.png"),
        )
    else:
        # No experimental data saved yet: run the pipeline on synthetic data
        # to illustrate the plot and the fit.
        _demo_synthetic_fit(
            output_path=Path(DATA_FOLDER + "/fig2a_ramsey_fit_demo.png")
        )