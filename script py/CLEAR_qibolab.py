""" CLEAR-pulse photon-population exp, following:
    D. T. McClure et al., "Rapid Driven Reset of a Qubit Readout Resonator", Phys. Rev. Applied 5, 011001 (2016).
 
Sequence (Fig. 1(b) of the paper):
    qubit preparation -> M1 -> t_relax -> Ramsey (RX90 - t_R - RX90) -> t_buffer -> M2.
 
Timing conventions taken from the paper:
    - t_relax is measured from the *end* of M1 to the *start* of the first
      X90 pulse ("The time trelax between the end of M1 and the start of
      the Ramsey experiment").
    - the two X90 pulses are separated by exactly t_R (t_gate = 8 ns,
      t_R = 0 to 600 ns).
    - t_buffer = 400 ns between the end of the Ramsey experiment and M2
      ("The time tbuffer between the Ramsey experiment and the second
      measurement pulse (M2) is set to 400 ns").
    - M2 is a plain square readout pulse, t_M2 = 10 us in the paper; here
      the platform-calibrated native MZ is used instead (see notes in
      build_clear_ramsey_sequence).
    - the Ramsey detuning is Delta/2pi = 10 MHz, applied here as a
      temporary offset of the drive-channel frequency.
 
M1 is built by `clear_drive_sequence`:
    - rectangular=False (default): the full 5-segment CLEAR pulse of
      Fig. 1(a) (ring-up x2, steady-state readout, ring-down x2).
    - rectangular=True: the paper's square-pulse baseline. By default its
      duration is 2*t_kick + t_steady, i.e. the CLEAR pulse *without* its
      two ring-down segments, so that combining it with t_relax = 2*t_kick
      reproduces the "square + delay" comparison of Fig. 3(c) ("for the
      square pulse, a delay of approximately 300 ns is inserted to match
      the total length of the CLEAR pulse's two ring-down segments").
      Set match_clear_length=False to get a bare square pulse of duration
      t_steady instead.
The rest of the sequence (preparation, t_relax, Ramsey, t_buffer, M2) is
identical in both cases, so switching `rectangular` reproduces the same
experiment with the two M1 pulse shapes compared in the paper, with
parameters that can differ from the ones used there (kappa, chi, powers,
etc., set via parametri/risonatore.toml and the calibration constants
below).
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
CLEAR_AMPLITUDES_JSON_PATH = PROJECT_DIR / "parametri" / "clear_amplitudes.json"
 
 
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


def load_clear_segment_amplitudes(
    readout_amplitude: complex,
    json_path: Path = CLEAR_AMPLITUDES_JSON_PATH,
) -> dict[str, complex]:
    """Load CLEAR segment amplitudes exported by amplitudes_CLEAR.ipynb.

    The JSON stores each segment as a ratio with respect to the steady
    readout amplitude; here the ratios are scaled by `readout_amplitude`.
    """

    with open(json_path) as f:
        data = json.load(f)

    ratios = data["ratios"]
    return {
        name: complex(*ratios[name]) * readout_amplitude
        for name in ("ringup_1", "ringup_2", "steady", "ringdown_1", "ringdown_2")
    }


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
 
    @property
    def ringdown_duration(self) -> float:
        """Total length of the two ring-down segments (300 ns in the paper),
        i.e. the delay to append to the square pulse for the Fig. 3(c)
        comparison."""
        return 2 * self.t_kick
 
 
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
 
 
def clear_drive_sequence(
    platform,
    qubit: int,
    params: ClearPulseParameters,
    rectangular: bool = False,
    match_clear_length: bool = True,
) -> PulseSequence:
    """Creates the M1 measurement/reset drive sequence.
 
    If `rectangular` is False (default), builds the full 5-segment CLEAR
    pulse (two ring-up kicks, the steady-state readout segment, two
    ring-down kicks), exactly as in Fig. 1(a) of McClure et al. (2016).
 
    If `rectangular` is True, a plain square M1 pulse at the steady-state
    amplitude `params.steady` is used instead:
      - match_clear_length=True (default): duration 2*t_kick + t_steady,
        i.e. the CLEAR pulse minus its ring-down segments. Together with
        t_relax = params.ringdown_duration this gives the paper's
        "square + delay" trace of Fig. 3(c), whose total M1 length equals
        that of the CLEAR pulse.
      - match_clear_length=False: duration t_steady only (bare square
        pulse).
    """
 
    probe_channel = platform.qubits[qubit].probe
    sequence = PulseSequence()
 
    if rectangular:
        duration = (
            2 * params.t_kick + params.t_steady
            if match_clear_length
            else params.t_steady
        )
        if duration > 0:
            sequence.append((probe_channel, rectangular_iq_pulse(duration, params.steady)))
        return sequence
 
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
 
 
def _delay_on_all(channels: Iterable, duration: float) -> tuple[PulseSequence, list[Delay]]:
    """Creates the same delay simultaneously on several channels.
 
    Returns the sub-sequence and the list of the `Delay` objects, so that
    they can all be handed to a duration `Sweeper` together: sweeping only
    one of them would desynchronize the channels, because the delays that
    `|=` inserts to align channels are compiled once, with the duration the
    sequence was built with.
    """
 
    delays = [Delay(duration=duration) for _ in channels]
    return PulseSequence(list(zip(channels, delays))), delays
 
 
def build_clear_ramsey_sequence(
    platform,
    qubit: int,
    clear: ClearPulseParameters,
    t_relax: float,
    t_ramsey: float,
    t_buffer: float = 400.0,
    prepare_excited: bool = False,
    rectangular: bool = False,
    match_clear_length: bool = True,
) -> tuple[PulseSequence, list[Delay], Pulse, int]:
    """Creates the sequence of Fig. 1(b):
 
        (RX) -> M1 -> t_relax -> RX90 - t_R - RX90 -> t_buffer -> M2
 
    Note on M2: the paper uses a square readout pulse of t_M2 = 10 us; here
    the platform's calibrated native MZ is used, whose duration/frequency
    come from the platform parameters. Change the readout duration there if
    you want to match the paper's 10 us exactly.
 
    returns:
        complete sequence of pulses.
        list of the Ramsey delays (one per channel) -- pass *all* of them to
        the duration Sweeper.
        the second X90 pulse, whose relative phase carries the Ramsey
        detuning (see run_clear_ramsey_scan).
        id of the acquisition pulse M2.
    """
 
    native = platform.natives.single_qubit[qubit]
    drive_channel = platform.qubits[qubit].drive
    probe_channel = platform.qubits[qubit].probe
    acquisition_channel = platform.qubits[qubit].acquisition
    channels = (drive_channel, probe_channel, acquisition_channel)
 
    sequence = PulseSequence()
 
    # Qubit preparation: ground state (nothing) or excited state (X pulse).
    if prepare_excited:
        sequence |= native.RX()
 
    # M1: CLEAR pulse, or square baseline. `|=` synchronizes all channels
    # first, so M1 starts after the preparation pulse.
    sequence |= clear_drive_sequence(
        platform, qubit, clear, rectangular=rectangular,
        match_clear_length=match_clear_length,
    )
 
    # t_relax, measured from the end of M1 to the start of the Ramsey.
    if t_relax > 0:
        relax_sequence, _ = _delay_on_all(channels, t_relax)
        sequence |= relax_sequence
 
    # Ramsey: RX90 - t_R - RX90. The delay is replicated on every channel so
    # that sweeping t_R keeps drive, probe and acquisition aligned.
    sequence |= native.RX90()
    ramsey_sequence, ramsey_delays = _delay_on_all(channels, t_ramsey)
    sequence |= ramsey_sequence
 
    # The second X90 carries the Ramsey detuning as a relative phase (see
    # run_clear_ramsey_scan): the drive channel must stay *on resonance*,
    # otherwise the preparation RX() below would also be detuned.
    second_x90_sequence = native.RX90()
    second_x90 = [
        pulse
        for pulse in second_x90_sequence.channel(drive_channel)
        if isinstance(pulse, Pulse)
    ][-1]
    sequence |= second_x90_sequence
 
    # t_buffer, so that lingering photons from M1/Ramsey do not corrupt M2.
    buffer_sequence, _ = _delay_on_all(channels, t_buffer)
    sequence |= buffer_sequence
 
    # M2: standard square readout, already calibrated in the platform.
    m2 = native.MZ()
    m2_readout = list(m2.channel(acquisition_channel))[-1]
    sequence |= m2
 
    return sequence, ramsey_delays, second_x90, m2_readout.id
 
 
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
    readout_frequency: float | None = None,
    prepare_excited: bool = False,
    rectangular: bool = False,
    match_clear_length: bool = True,
    t_buffer: float = 400.0,
) -> None:
    """Performs Ramsey-after-M1 scans and saves the integrated results.
 
    `readout_frequency` (Hz) overrides the probe-channel frequency for the
    whole sequence. It must be set to the *midpoint* between the two
    qubit-state-dependent resonator frequencies,
 
        f_probe = (f_res_g + f_res_e) / 2 = f_dressed + chi/2pi,
 
    as the paper does ("the measurement tone is applied at the midpoint of
    the two frequencies"). This is what makes the steady-state photon
    number independent of the qubit state in the linear regime; driving at
    the platform's default readout frequency (typically the ground-state
    resonance, from resonator_spectroscopy) populates the cavity for one
    qubit state only. Note the probe frequency is a *channel* config in
    qibolab, so M1 and M2 necessarily share it; the midpoint is also where
    the readout signal is largest, so M2 does not suffer from this.
 
    The Ramsey detuning is *not* applied by shifting the drive-channel
    frequency: that would also detune the RX() preparation pulse, spoiling
    the excited-state preparation. It is applied instead as a phase ramp
    phi = 2*pi*Delta*t_R on the second X90, swept in parallel with t_R,
    which is equivalent to a detuned Ramsey and leaves RX() on resonance.
    """
 
    output.mkdir(parents=True, exist_ok=True)
    t_relax_values = np.asarray(list(t_relax_values), dtype=float)
    t_ramsey_values = np.asarray(list(t_ramsey_values), dtype=float)
    platform = create_platform(platform_name)
 
    all_results = {}
    platform.connect()
    try:
        updates = []
        if readout_frequency is not None:
            probe_channel = platform.qubits[qubit].probe
            updates.append({probe_channel: {"frequency": readout_frequency}})
 
        for t_relax in t_relax_values:
            sequence, ramsey_delays, second_x90, m2_id = build_clear_ramsey_sequence(
                platform=platform,
                qubit=qubit,
                clear=clear,
                t_relax=float(t_relax),
                t_ramsey=float(t_ramsey_values[0]),
                t_buffer=t_buffer,
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
 
            if ramsey_detuning:
                # phi(t_R) = 2*pi*Delta*t_R, with Delta in Hz and t_R in ns.
                phases = (
                    2 * np.pi * ramsey_detuning * 1e-9 * t_ramsey_values
                ) % (2 * np.pi)
                sweepers.append(
                    Sweeper(
                        parameter=Parameter.relative_phase,
                        values=phases,
                        pulses=[second_x90],
                    )
                )
 
            results = platform.execute(
                [sequence],
                [sweepers],
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
    ramsey_detuning = 6e6
    # n_target: dal file di calibrazione parametri/risonatore.toml (lo
    # stesso letto in Sezione 1 di amplitudes_CLEAR.ipynb).
    resonator_params = load_resonator_toml()
    n_target = resonator_params["n_target"]

    # k: coefficiente angolare del fit lineare n_photons = k*amp^2 + q,
    # prodotto da stark_analysis.py e salvato in data/stark_fit_results.json
    # -- stessa quantita' letta in Sezione 1 del notebook, cosi' che
    # A_readout usi sempre l'ultima calibrazione di Stark shift.
    k = 3.49e4  # load_stark_fit_k()

    A_readout = float(np.sqrt(n_target / k))
    clear_amplitudes = load_clear_segment_amplitudes(A_readout + 0.0j)

    clear_parameters = ClearPulseParameters(
        ringup_1=clear_amplitudes["ringup_1"],
        ringup_2=clear_amplitudes["ringup_2"],
        steady=clear_amplitudes["steady"],
        ringdown_1=clear_amplitudes["ringdown_1"],
        ringdown_2=clear_amplitudes["ringdown_2"],
        t_kick=150.0,
        t_steady=3000.0,
    )

    t_relax = np.arange(0, 1600, 20)
    res ={}
    for excited in [False]:
        #run_clear_ramsey_scan(
        #    platform_name=PLATFORM,
        #    qubit=QUBIT,
        #    clear=clear_parameters,
        #    t_relax_values=t_relax,
        #    t_ramsey_values=np.arange(0.0, 600, 8.0),
        #    output=Path(DATA_FOLDER + ""),
        #    nshots=1024*10,
        #    relaxation_time=200_000,
        #    ramsey_detuning=ramsey_detuning,
        #    readout_frequency = None,
        #    prepare_excited=excited,
        #    rectangular=True,
        #)

        # --- Fig. 2(a): Ramsey trace and Eq. (1) fit, extracting n0 ---------
        # Substitute kappa_mhz, chi_mhz and t2_echo_ns with the values from your
        # calibration once available.
        if excited:
            npz_path = Path(DATA_FOLDER + "/clear_ramsey_excited.npz")
        else:
            npz_path = Path(DATA_FOLDER + "/clear_ramsey_ground.npz")

        if npz_path.exists():
            n0s = []
            n0_errs = []
            for t in t_relax:
                fit_result = analyze_ramsey_fig2a(
                    npz_path=npz_path,
                    t_relax=t,
                    kappa_mhz=0.7,
                    chi_mhz=(7269.555-7267.185)/2,
                    ramsey_detuning_mhz=ramsey_detuning/1e6,
                    t2_echo_ns=61956000.0,
                    output_path=Path(DATA_FOLDER + f"/fig2a_ramsey_fit_relax_{t}_ns.png"),
                )
                n0s.append(fit_result['n0'])
                n0_errs.append(fit_result['n0_err'])
                plt.close()
            if excited:
                res["e"] = {"t": t_relax, "n0": n0s, "n0_err": n0_errs}
            else:
                res["g"] = {"t": t_relax, "n0": n0s, "n0_err": n0_errs}

        else:
            # No experimental data saved yet: run the pipeline on synthetic data
            # to illustrate the plot and the fit.
            _demo_synthetic_fit(
                output_path=Path(DATA_FOLDER + "/fig2a_ramsey_fit_demo.png")
            )
    def decay_model(t, offset, amplitude, tau):
        return offset + amplitude * np.exp(-np.asarray(t, dtype=float) / tau)

    plt.figure()
    for key in ["g", "e"]:
        if key in res:
            t_data = np.asarray(res[key]["t"], dtype=float)
            n0_data = np.asarray(res[key]["n0"], dtype=float)
            n0_err_data = np.asarray(res[key]["n0_err"], dtype=float)

            plt.errorbar(t_data, n0_data, n0_err_data, marker="o", label=key)

            mask = np.isfinite(t_data) & np.isfinite(n0_data)
            if np.count_nonzero(mask) >= 3:
                p0 = (
                    float(n0_data[mask][-1]),
                    float(n0_data[mask][0] - n0_data[mask][-1]),
                    max(float(t_data[mask][-1] - t_data[mask][0]), 1.0),
                )
                sigma = n0_err_data[mask]
                sigma = np.where(sigma > 0, sigma, 1.0)

                try:
                    popt, pcov = curve_fit(
                        decay_model,
                        t_data[mask],
                        n0_data[mask],
                        p0=p0,
                        sigma=sigma,
                        bounds=([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf]),
                        maxfev=10_000,
                    )
                    t_fit = np.linspace(t_data[mask].min(), t_data[mask].max(), 400)
                    plt.plot(
                        t_fit,
                        decay_model(t_fit, *popt),
                        "-",
                        label=f"{key} fit, tau={popt[2]:.1f} ns",
                    )
                    # Extract tau and its uncertainty from the covariance matrix
                    tau = float(popt[2])
                    try:
                        tau_err = float(np.sqrt(np.diag(pcov))[2])
                    except Exception:
                        tau_err = float('nan')

                    # k = 1 / tau, propagate uncertainty: dk = dtau / tau^2
                    k_val = 1.0 / tau if tau != 0 else float('inf')
                    k_err = (tau_err / (tau * tau)) if (tau_err == tau_err and tau != 0) else float('nan')

                    print(f"{key}: tau = {tau:.3f} +/- {tau_err:.3f} ns, k = {k_val:.6e} +/- {k_err:.6e} 1/ns")

                    # store results in res if present
                    if key in res:
                        res[key]["tau"] = tau
                        res[key]["tau_err"] = tau_err
                        res[key]["k"] = k_val
                        res[key]["k_err"] = k_err
                except (RuntimeError, ValueError) as exc:
                    print(f"{key}: fit n0(t_relax) non riuscito: {exc}")
    plt.legend()
    plt.grid()
    plt.savefig("test_decay.png")
