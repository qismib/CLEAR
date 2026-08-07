"""Fig. 3(a),(b) of McClure et al. (2016): the IQ-plane trajectory of the
cavity field during a square and a CLEAR measurement pulse, with the qubit
prepared in the ground or the excited state.

    (RX) -> M1, acquired in scope mode for the whole pulse plus a tail

Nothing is measured on the qubit here: the observable is the transmitted
readout tone itself, sampled during M1. The acquisition therefore runs in
`AcquisitionType.RAW` (the scope trace, one complex sample per ADC period)
and starts simultaneously with the first pulse segment, so that the raw trace
*is* the cavity response alpha(t), up to a delay, a complex gain and an
offset of the acquisition chain.

Those three nuisances are removed here, not on the instrument:

  delay   the trace is cross-correlated with the calculated response of the
          square pulse, whose edges are sharp, and the resulting offset is
          applied to every trace of the run;
  gain    a single complex factor c and offset b are fitted once, by least
          squares, on the square-pulse traces (y = c*alpha + b), then applied
          to the CLEAR ones. This is the "multiplied by an overall amplitude
          factor to match the data" of the paper, except that we divide it
          out of the data instead, so that the plot axes are sqrt(photons)
          and the dashed target circle sits at sqrt(n) by construction;
  IF      the scope trace is at the intermediate frequency, so it is
          digitally demodulated. The IF is taken from the data (FFT peak plus
          a phase-slope refinement over the steady-state window), because the
          hardware value is only known for platforms with a configured LO.

The theory curves are the response of the linear resonator model to the
piecewise-constant envelope that was actually played -- the segment
amplitudes are read back from the saved run, not recomputed -- with

    alpha' = -(kappa/2 + i*Delta_s) alpha - i eps(t),
    Delta_g = +chi,  Delta_e = -chi   (drive at the midpoint),

which is the Python transcription of impulsi_readout.jl. As in the paper, the
calculated curve is the mixture of the two state-conditioned trajectories
weighted by the residual excited population, since the measured trace is an
average over shots and the signal is linear in alpha.

Times in ns, rates in rad/ns, amplitudes in hardware units.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clear_config import ClearPulseParameters, Setup, get_setup
from clear_analysis import save_figure

# qibolab and the sequence builders are imported inside run_iq_trajectories,
# so that the analysis can be re-run without the hardware stack.


PULSE_TYPES = ("square", "clear")
STATES = ("ground", "excited")

STYLES = {
    "ground": dict(color="tab:blue", marker="o", label="ground"),
    "excited": dict(color="tab:red", marker="s", label="excited"),
}

TITLES = {"square": "square pulse", "clear": "CLEAR pulse"}

MARKER_BIN_NS = 24.0
"""Spacing of the markers along the trajectory, as in Fig. 3(a),(b)."""

TAIL_NS = 1500.0
"""Acquisition kept running after the end of M1, to show the free decay of
the square pulse next to the driven reset of CLEAR."""

SCOPE_LIMIT_NS = 16384.0
"""Typical maximum scope-acquisition length; only used to warn."""


# ---------------------------------------------------------------------------
# 1. The pulse, as a list of segments
# ---------------------------------------------------------------------------


def pulse_segments(
    params: ClearPulseParameters, rectangular: bool = False, match_clear_length: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """(amplitudes, durations) of M1, for the theory and for the record kept
    in the npz. Identical to what `clear_drive_sequence` plays."""
    if rectangular:
        return (
            np.array([params.steady], dtype=complex),
            np.array([params.duration(True, match_clear_length)], dtype=float),
        )
    return np.array(params.amplitudes, dtype=complex), np.array(params.durations, dtype=float)


def segment_edges(durations: np.ndarray) -> np.ndarray:
    """Times at which the envelope changes, including 0 and the pulse end."""
    return np.concatenate([[0.0], np.cumsum(np.asarray(durations, dtype=float))])


# ---------------------------------------------------------------------------
# 2. Linear-resonator model (port of impulsi_readout.jl)
# ---------------------------------------------------------------------------


def propagate_constant_drive(alpha0: complex, eps: complex, kappa: float, delta: float, tau):
    """alpha after a time tau of constant drive eps, exactly."""
    lam = kappa / 2.0 + 1j * delta
    decay = np.exp(-lam * np.asarray(tau, dtype=float))
    return decay * alpha0 - 1j * eps * (1.0 - decay) / lam


def resonator_response(
    times: np.ndarray, amplitudes: np.ndarray, durations: np.ndarray, kappa: float, delta: float
) -> np.ndarray:
    """alpha(t) for a piecewise-constant envelope, exact within each segment.

    Drive amplitudes are in the arbitrary units of `amplitudes`; after the
    last segment the field decays freely, which is what makes the square
    pulse's tail comparable to CLEAR's ring-down.
    """
    times = np.asarray(times, dtype=float)
    alpha = np.zeros(times.shape, dtype=complex)
    edges = segment_edges(durations)

    state = 0.0 + 0.0j
    for eps, t_start, t_stop in zip(amplitudes, edges[:-1], edges[1:]):
        inside = (times >= t_start) & (times < t_stop)
        alpha[inside] = propagate_constant_drive(state, eps, kappa, delta, times[inside] - t_start)
        state = propagate_constant_drive(state, eps, kappa, delta, t_stop - t_start)

    after = times >= edges[-1]
    alpha[after] = propagate_constant_drive(state, 0.0, kappa, delta, times[after] - edges[-1])
    return alpha


def state_detuning(setup: Setup, state: str) -> float:
    """Detuning of the state-dependent resonator from the drive, which sits at
    the midpoint: +chi for |g> (the higher of the two frequencies), -chi for
    |e>. Same convention as `clear_kick_amplitudes` in the Julia model, where
    Delta_s = detuning -/+ chi with chi = pi*(f_e - f_g) < 0."""
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    return setup.chi_rad_ns if state == "ground" else -setup.chi_rad_ns


def photon_normalization(setup: Setup, amplitudes: np.ndarray) -> float:
    """Factor turning the model alpha into sqrt(photons).

    The steady segment drives the cavity to |alpha_ss| = |A|/|kappa/2 + i chi|
    in model units and to sqrt(k*|A|^2) photons according to the Stark
    calibration, and at the midpoint both qubit states give the same
    |alpha_ss|, so one factor serves both.
    """
    amp_steady = np.max(np.abs(amplitudes))
    if amp_steady == 0:
        return 1.0
    lam = abs(setup.kappa_rad_ns / 2.0 + 1j * setup.chi_rad_ns)
    alpha_model = amp_steady / lam
    return float(np.sqrt(setup.k_stark) * amp_steady / alpha_model)


def trajectory(
    setup: Setup, times: np.ndarray, amplitudes: np.ndarray, durations: np.ndarray, state: str
) -> np.ndarray:
    """Calculated cavity field in sqrt(photon) units, so that |alpha|^2 is the
    photon number and the steady state sits on the circle of radius
    sqrt(n)."""
    alpha = resonator_response(
        times, amplitudes, durations, setup.kappa_rad_ns, state_detuning(setup, state)
    )
    return alpha * photon_normalization(setup, amplitudes)


def mixed_trajectory(
    setup: Setup,
    times: np.ndarray,
    amplitudes: np.ndarray,
    durations: np.ndarray,
    p_excited: float,
) -> np.ndarray:
    """Shot-averaged field for a preparation that leaves a fraction p_excited
    of the ensemble in |e>.

    The average of a linear observable over a mixed state is the mixture of
    the two conditioned trajectories, which is why the paper's calculated
    curves are "adjusted to reflect the independently observed 20% thermal
    population of the qubit excited state". Qubit transitions *during* M1 are
    neglected: they would move a trajectory from one branch to the other
    partway through, and for T1 >> M1 the correction is second order.
    """
    p = float(np.clip(p_excited, 0.0, 1.0))
    alpha_g = trajectory(setup, times, amplitudes, durations, "ground")
    if p == 0.0:
        return alpha_g
    alpha_e = trajectory(setup, times, amplitudes, durations, "excited")
    return (1.0 - p) * alpha_g + p * alpha_e


# ---------------------------------------------------------------------------
# 3. Acquisition
# ---------------------------------------------------------------------------


def build_trajectory_sequence(
    platform,
    qubit: int,
    params: ClearPulseParameters,
    acquisition_ns: float,
    prepare_excited: bool = False,
    rectangular: bool = False,
    match_clear_length: bool = True,
):
    """(RX) -> M1, with a scope acquisition starting with the first segment.

    The acquisition is appended to the same block as the drive segments, so
    the two channels start together; the `|=` before it aligns that block
    after the RX preparation.
    """
    from qibolab import Acquisition, PulseSequence

    from CLEAR_qibolab2 import clear_drive_sequence

    native = platform.natives.single_qubit[qubit]
    sequence = PulseSequence()

    if prepare_excited:
        sequence |= native.RX()

    block = clear_drive_sequence(
        platform, qubit, params, rectangular=rectangular, match_clear_length=match_clear_length
    )
    acquisition = Acquisition(duration=float(acquisition_ns))
    block.append((platform.qubits[qubit].acquisition, acquisition))
    sequence |= block

    return sequence, acquisition.id


def probe_if_frequency(platform, setup: Setup) -> float | None:
    """Probe IF = probe RF - LO, when the platform declares an LO for the
    probe channel. Only stored as a cross-check: the demodulation frequency
    used in the analysis is measured on the trace itself."""
    try:
        channel = platform.channels[platform.qubits[setup.qubit].probe]
        if getattr(channel, "lo", None) is None:
            return None
        return float(setup.readout_frequency_hz - platform.config(channel.lo).frequency)
    except (KeyError, AttributeError):
        return None


def run_iq_trajectories(
    setup: Setup,
    n: float | None = None,
    tail_ns: float = TAIL_NS,
    nshots: int | None = None,
    output_name: str = "fig3ab",
) -> Path:
    """Acquires the scope traces of Fig. 3(a),(b) in one platform session.

    One trace per (pulse type, prepared state) at the same steady-state
    amplitude, all with the same acquisition window -- long enough for the
    CLEAR pulse plus a tail, so the square pulse's free decay is recorded over
    the same span -- plus a background trace with the probe silent.
    """

    from qibolab import AcquisitionType, AveragingMode, create_platform

    from CLEAR_qibolab2 import _probe_update

    params = setup.clear_pulse(n)
    acquisition_ns = float(params.duration() + tail_ns)
    if acquisition_ns > SCOPE_LIMIT_NS:
        warnings.warn(
            f"acquisition window {acquisition_ns:.0f} ns exceeds the usual scope limit "
            f"of {SCOPE_LIMIT_NS:.0f} ns; shorten t_steady or tail_ns.",
            stacklevel=2,
        )

    platform = create_platform(setup.platform)
    platform.connect()
    payload: dict[str, np.ndarray] = {}
    try:
        updates = _probe_update(platform, setup)

        def acquire(sequence, acquisition_id):
            result = platform.execute(
                [sequence],
                [],
                nshots=setup.nshots if nshots is None else int(nshots),
                relaxation_time=setup.relaxation_time_ns,
                acquisition_type=AcquisitionType.RAW,
                averaging_mode=AveragingMode.CYCLIC,
                updates=updates,
            )
            return np.asarray(result[acquisition_id])

        for pulse_type in PULSE_TYPES:
            rectangular = pulse_type == "square"
            amplitudes, durations = pulse_segments(params, rectangular)
            payload[f"{pulse_type}_amplitudes"] = amplitudes
            payload[f"{pulse_type}_durations"] = durations
            for state in STATES:
                sequence, acquisition_id = build_trajectory_sequence(
                    platform,
                    setup.qubit,
                    params,
                    acquisition_ns,
                    prepare_excited=(state == "excited"),
                    rectangular=rectangular,
                )
                payload[f"{pulse_type}_{state}"] = acquire(sequence, acquisition_id)

        # Background: the same window with no probe segments at all, i.e. the
        # offset of the acquisition chain, subtracted from every trace.
        silent = setup.clear_pulse(0.0)
        sequence, acquisition_id = build_trajectory_sequence(
            platform, setup.qubit, silent, acquisition_ns, rectangular=True
        )
        payload["background"] = acquire(sequence, acquisition_id)

        payload["sampling_rate_gsps"] = np.array(float(platform.sampling_rate))
        if_hz = probe_if_frequency(platform, setup)
        payload["if_hz_hardware"] = np.array(np.nan if if_hz is None else if_hz)
    finally:
        platform.disconnect()

    payload["acquisition_ns"] = np.array(acquisition_ns)
    payload["n_steady"] = np.array(setup.n_target if n is None else float(n))

    setup.data_dir.mkdir(parents=True, exist_ok=True)
    path = setup.data_dir / f"{output_name}.npz"
    np.savez(path, **payload)
    return path


# ---------------------------------------------------------------------------
# 4. From scope samples to a trajectory
# ---------------------------------------------------------------------------


def to_iq(raw: np.ndarray) -> np.ndarray:
    """Scope samples -> complex trace. Accepts (..., 2) I/Q pairs or an
    already complex array, and averages any leading shot axis."""
    raw = np.asarray(raw)
    if not np.iscomplexobj(raw) and raw.shape[-1] == 2:
        raw = raw[..., 0] + 1j * raw[..., 1]
    while raw.ndim > 1:
        raw = raw.mean(axis=0)
    return raw


def sample_times(trace: np.ndarray, sampling_rate_gsps: float) -> np.ndarray:
    """Time axis in ns, starting at the first acquired sample."""
    return np.arange(trace.size, dtype=float) / float(sampling_rate_gsps)


def estimate_if_frequency(t: np.ndarray, z: np.ndarray, window: slice | None = None) -> float:
    """Demodulation frequency in GHz, from the trace itself.

    Coarse FFT peak over `window` (a stretch where the drive is on and the
    field is in steady state), then refined on the phase slope of the
    coarsely demodulated trace: the FFT resolution over ~1 us is ~1 MHz,
    which would still wind the trajectory by a radian over the record.
    """
    if window is None:
        window = slice(0, z.size)
    segment = z[window]
    if segment.size < 16:
        raise ValueError("steady-state window too short to estimate the IF")
    dt = float(np.mean(np.diff(t)))

    spectrum = np.fft.fft(segment)
    freqs = np.fft.fftfreq(segment.size, d=dt)
    coarse = float(freqs[np.argmax(np.abs(spectrum))])

    phase = np.unwrap(np.angle(segment * np.exp(-2j * np.pi * coarse * t[window])))
    slope = np.polyfit(t[window], phase, 1)[0]
    return coarse + slope / (2.0 * np.pi)


def demodulate(t: np.ndarray, z: np.ndarray, if_ghz: float) -> np.ndarray:
    """Bring the scope trace down to baseband. A residual global phase is
    irrelevant: it is absorbed by the complex gain of the calibration."""
    return z * np.exp(-2j * np.pi * if_ghz * t)


def bin_trace(t: np.ndarray, z: np.ndarray, bin_ns: float) -> tuple[np.ndarray, np.ndarray]:
    """Box-average into bins of `bin_ns`. This is the low-pass filter that
    removes the image at 2*IF and most of the noise; the bin is short compared
    with 1/kappa, so the trajectory is not distorted."""
    dt = float(np.mean(np.diff(t)))
    width = max(int(round(bin_ns / dt)), 1)
    n_bins = z.size // width
    if n_bins == 0:
        return t.copy(), z.copy()
    t_binned = t[: n_bins * width].reshape(n_bins, width).mean(axis=1)
    z_binned = z[: n_bins * width].reshape(n_bins, width).mean(axis=1)
    return t_binned, z_binned


def estimate_time_offset(
    t: np.ndarray,
    z: np.ndarray,
    t_model: np.ndarray,
    alpha_model: np.ndarray,
    max_offset_ns: float = 600.0,
    step_ns: float = 1.0,
) -> float:
    """Delay between the start of the acquisition window and the arrival of
    the pulse, from the correlation of the two magnitudes.

    Estimated on the square pulse, whose ring-up and free decay give the trace
    an unambiguous shape, and then applied to every trace of the run: the
    cable delay and the time of flight do not depend on the envelope.
    """
    data = np.abs(z)
    data = data - data.mean()
    offsets = np.arange(0.0, max_offset_ns + step_ns, step_ns)

    best_offset, best_score = 0.0, -np.inf
    for offset in offsets:
        model = np.interp(t - offset, t_model, np.abs(alpha_model), left=0.0, right=0.0)
        model = model - model.mean()
        norm = np.linalg.norm(data) * np.linalg.norm(model)
        if norm == 0:
            continue
        score = float(np.dot(data, model) / norm)
        if score > best_score:
            best_offset, best_score = float(offset), score
    return best_offset


def calibrate_gain(alpha_model: np.ndarray, y: np.ndarray) -> tuple[complex, complex]:
    """Least-squares c and b in y = c*alpha + b.

    Fitted on the square-pulse traces only, and reused for CLEAR: both are
    acquired in the same session at the same probe frequency, so the response
    of the acquisition chain is common, and letting CLEAR set its own gain
    would hide exactly the discrepancy we want to see.
    """
    design = np.column_stack([alpha_model, np.ones_like(alpha_model)])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    return complex(solution[0]), complex(solution[1])


# ---------------------------------------------------------------------------
# 5. Analysis
# ---------------------------------------------------------------------------


def _steady_window(t: np.ndarray, durations: np.ndarray, offset: float = 0.0) -> slice:
    """Samples in the settled part of the steady-state segment, used for the
    IF fit.

    Only the second half of the segment is taken: the IF is refined on the
    phase slope, so any residual ring-up would be read as a frequency error.
    A time offset that has not been determined yet is tolerated because the
    window is well inside the segment on either side.
    """
    edges = segment_edges(durations)
    start, stop = (edges[2], edges[3]) if edges.size >= 5 else (edges[0], edges[-1])
    start, stop = start + 0.5 * (stop - start) + offset, stop - 0.05 * (stop - start) + offset
    inside = np.flatnonzero((t >= start) & (t <= stop))
    if inside.size < 16:
        return slice(0, t.size)
    return slice(int(inside[0]), int(inside[-1]) + 1)


def analyze_iq_trajectories(
    npz_path: Path,
    setup: Setup,
    p_thermal: float = 0.0,
    p_excited: float = 1.0,
    bin_ns: float = MARKER_BIN_NS,
    if_ghz: float | None = None,
    time_offset_ns: float | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Demodulates, aligns and calibrates every trace of a run, and draws
    Fig. 3(a),(b) plus the photon number versus time.

    `p_thermal` is the residual excited population of a nominally ground
    preparation and `p_excited` the surviving excited fraction after an RX
    (measurable with `run_excited_fraction` from CLEAR_qibolab2). They only
    enter the calculated curves, never the data.
    """

    data = np.load(npz_path)
    output_dir = Path(output_dir or npz_path.parent)
    sampling_rate = float(data["sampling_rate_gsps"])
    n_steady = float(data["n_steady"])
    weights = {"ground": p_thermal, "excited": p_excited}

    background = to_iq(data["background"]) if "background" in data else 0.0

    raw = {}
    for pulse_type in PULSE_TYPES:
        for state in STATES:
            key = f"{pulse_type}_{state}"
            if key in data:
                raw[(pulse_type, state)] = to_iq(data[key]) - background
    if not raw:
        raise ValueError(f"{npz_path} contains no trajectory traces")

    any_trace = next(iter(raw.values()))
    t = sample_times(any_trace, sampling_rate)
    durations = {pt: data[f"{pt}_durations"] for pt in PULSE_TYPES if f"{pt}_durations" in data}
    amplitudes = {pt: data[f"{pt}_amplitudes"] for pt in PULSE_TYPES if f"{pt}_amplitudes" in data}

    # -- IF, from the CLEAR steady segment if it was acquired ---------------
    if if_ghz is None:
        reference = ("clear", "ground") if ("clear", "ground") in raw else next(iter(raw))
        window = _steady_window(t, durations[reference[0]])
        if_ghz = estimate_if_frequency(t, raw[reference], window)
    if_hardware = float(data["if_hz_hardware"]) if "if_hz_hardware" in data else np.nan
    if np.isfinite(if_hardware) and abs(if_hardware * 1e-9 - if_ghz) > 1e-3:
        warnings.warn(
            f"IF measured on the trace ({if_ghz * 1e3:.2f} MHz) differs from the "
            f"platform's probe IF ({if_hardware * 1e-6:.2f} MHz) by more than 1 MHz.",
            stacklevel=2,
        )

    binned = {}
    for key, z in raw.items():
        binned[key] = bin_trace(t, demodulate(t, z, if_ghz), bin_ns)

    # -- delay, from the square pulse ---------------------------------------
    t_fine = np.arange(0.0, t[-1] + 1.0, 1.0)
    if time_offset_ns is None:
        anchor = ("square", "ground") if ("square", "ground") in binned else next(iter(binned))
        model = mixed_trajectory(
            setup, t_fine, amplitudes[anchor[0]], durations[anchor[0]], weights[anchor[1]]
        )
        time_offset_ns = estimate_time_offset(*binned[anchor], t_fine, model)

    # -- gain and offset, from the square pulse -----------------------------
    models = {
        key: mixed_trajectory(
            setup,
            binned[key][0] - time_offset_ns,
            amplitudes[key[0]],
            durations[key[0]],
            weights[key[1]],
        )
        for key in binned
    }
    calibration_keys = [key for key in binned if key[0] == "square"] or list(binned)
    gain, offset = calibrate_gain(
        np.concatenate([models[key] for key in calibration_keys]),
        np.concatenate([binned[key][1] for key in calibration_keys]),
    )

    trajectories = {
        key: {
            "t": binned[key][0] - time_offset_ns,
            "alpha": (binned[key][1] - offset) / gain,
            "alpha_model": models[key],
        }
        for key in binned
    }

    results = {
        "trajectories": trajectories,
        "amplitudes": amplitudes,
        "durations": durations,
        "if_ghz": float(if_ghz),
        "time_offset_ns": float(time_offset_ns),
        "gain": gain,
        "offset": offset,
        "n_steady": n_steady,
    }

    print(f"IF = {if_ghz * 1e3:.3f} MHz   delay = {time_offset_ns:.0f} ns")
    print(f"gain |c| = {abs(gain):.4g}, arg c = {np.angle(gain):+.3f} rad")
    for key in sorted(trajectories):
        alpha = trajectories[key]["alpha"]
        n_end = abs(alpha[_end_index(trajectories[key]["t"], durations[key[0]])]) ** 2
        residual = np.abs(alpha - trajectories[key]["alpha_model"])
        print(
            f"  {key[0]:>6s} {key[1]:<8s} "
            f"max n = {np.max(np.abs(alpha) ** 2):5.2f}   "
            f"n at the end of M1 = {n_end:5.2f}   "
            f"rms |alpha - model| = {np.sqrt(np.mean(residual ** 2)):.3f}"
        )

    plot_iq_planes(results, output_dir / "fig3ab_iq_trajectories.png")
    plot_photon_number(results, output_dir / "fig3ab_photons_vs_time.png")
    return results


def _end_index(t: np.ndarray, durations: np.ndarray) -> int:
    """Index of the first sample at or after the end of the pulse."""
    end = segment_edges(durations)[-1]
    inside = np.flatnonzero(t >= end)
    return int(inside[0]) if inside.size else int(t.size - 1)


# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------


def plot_iq_planes(results: dict, output_path: Path) -> None:
    """Fig. 3(a),(b): markers for the measured trajectory, solid curves for
    the calculated one, a dashed circle at the target population and a cross
    at the origin."""

    trajectories = results["trajectories"]
    pulse_types = [pt for pt in PULSE_TYPES if any(key[0] == pt for key in trajectories)]
    fig, axes = plt.subplots(1, len(pulse_types), figsize=(5.2 * len(pulse_types), 5.0))
    axes = np.atleast_1d(axes)
    radius = np.sqrt(results["n_steady"])

    for ax, pulse_type, panel in zip(axes, pulse_types, "abcdef"):
        for state in STATES:
            key = (pulse_type, state)
            if key not in trajectories:
                continue
            alpha = trajectories[key]["alpha"]
            model = trajectories[key]["alpha_model"]
            style = STYLES[state]
            ax.plot(
                np.real(alpha),
                np.imag(alpha),
                ls="none",
                ms=4,
                mfc="none",
                color=style["color"],
                marker=style["marker"],
                label=f"{style['label']} (data)",
            )
            ax.plot(
                np.real(model), np.imag(model), color=style["color"], lw=1.4, alpha=0.9,
                label=f"{style['label']} (model)",
            )
            _add_arrows(ax, model, style["color"])

        angle = np.linspace(0, 2 * np.pi, 361)
        ax.plot(radius * np.cos(angle), radius * np.sin(angle), "k--", lw=0.9,
                label=rf"$n = {results['n_steady']:.3g}$")
        ax.plot(0, 0, "k+", ms=10, mew=1.5)
        ax.set_aspect("equal")
        ax.set_xlabel(r"Re $\alpha$ ($\sqrt{\mathrm{photons}}$)")
        ax.set_ylabel(r"Im $\alpha$ ($\sqrt{\mathrm{photons}}$)")
        ax.set_title(f"({panel}) {TITLES[pulse_type]}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    # Same limits everywhere: the point of the figure is that one pulse stays
    # compact where the other does not.
    extent = 1.1 * max(
        [radius] + [np.max(np.abs(traj["alpha"])) for traj in trajectories.values()]
    )
    for ax in axes:
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)

    save_figure(fig, output_path)


def _add_arrows(ax, curve: np.ndarray, color: str, count: int = 3) -> None:
    """A few arrows along a calculated trajectory, to show its direction."""
    step = max(curve.size // (count + 1), 1)
    for i in range(step, curve.size - 1, step):
        start, end = curve[i], curve[i + 1]
        if abs(end - start) == 0:
            continue
        ax.annotate(
            "",
            xy=(np.real(end), np.imag(end)),
            xytext=(np.real(start), np.imag(start)),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=12),
        )


def plot_photon_number(results: dict, output_path: Path) -> None:
    """|alpha|^2 versus time for both pulses: the same data as the IQ plane,
    read as the ring-up and ring-down the CLEAR pulse is meant to shorten."""

    trajectories = results["trajectories"]
    pulse_types = [pt for pt in PULSE_TYPES if any(key[0] == pt for key in trajectories)]
    fig, axes = plt.subplots(
        1, len(pulse_types), figsize=(5.6 * len(pulse_types), 4.0), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, pulse_type in zip(axes, pulse_types):
        for state in STATES:
            key = (pulse_type, state)
            if key not in trajectories:
                continue
            style = STYLES[state]
            traj = trajectories[key]
            ax.plot(traj["t"], np.abs(traj["alpha"]) ** 2, ls="none", ms=4, mfc="none",
                    marker=style["marker"], color=style["color"], label=style["label"])
            ax.plot(traj["t"], np.abs(traj["alpha_model"]) ** 2, color=style["color"], lw=1.2)

        for edge in segment_edges(results["durations"][pulse_type]):
            ax.axvline(edge, color="0.7", lw=0.8, ls=":")
        ax.axhline(results["n_steady"], color="k", ls="--", lw=0.9)
        ax.set_xlabel(r"$t$ (ns)")
        ax.set_title(TITLES[pulse_type])
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel(r"photon number $|\alpha|^2$")

    save_figure(fig, output_path)


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    setup = get_setup(platform="2q_chip_thesis", qubit=1)

    print(f"probe at midpoint      {setup.f_probe_mhz:.6f} MHz")
    print(f"kappa/2pi = {setup.kappa_mhz:.4f} MHz   chi/2pi = {setup.chi_mhz:.4f} MHz")
    print(f"n_target = {setup.n_target:g}  ->  amplitude {setup.amplitude_for_n(setup.n_target):.4f}")

    npz_path = setup.data_dir / "fig3ab.npz"
    if not npz_path.exists():
        npz_path = run_iq_trajectories(setup, output_name="fig3ab")

    analyze_iq_trajectories(
        npz_path,
        setup,
        # Independently measured populations; both default to the ideal case.
        p_thermal=0.0,
        p_excited=1.0,
    )