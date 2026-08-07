"""Calibration of the probe frequency, from the g/e symmetry of the residual
cavity population.

Rationale
---------
In the linear dispersive regime the steady-state population is

    n_bar = |eps|^2 / [(kappa/2)^2 + Delta^2]

with Delta the detuning of the probe from the *state-dependent* resonator
frequency. Driving at the midpoint of f_res_g and f_res_e gives
Delta_g = -chi and Delta_e = +chi, so |Delta| -- and therefore n_bar -- is the
same for both qubit states. That symmetry is what lets a single Stark
calibration serve both preparations.

Locating the midpoint by fitting the two resonator spectroscopies separately
is fragile here: with kappa/2pi = 0.75 MHz and 2*chi/2pi = 1.07 MHz the two
Lorentzians overlap heavily, and T1 decay plus preparation error distort the
excited one. So instead of inferring the midpoint from lineshapes, this
experiment measures the thing that actually has to be symmetric, on the same
sequence used for the physics:

    for each probe frequency f:
        rectangular M1, t_relax = 0, Ramsey -> n0_g(f) and n0_e(f)
    find f where  Delta_n0(f) = n0_e(f) - n0_g(f)  crosses zero.

Two properties make this a good criterion:

  * It is a *ratio at fixed f*, so the Stark slope k, the readout contrast and
    every other amplitude scale factor cancel. This calibration therefore does
    not depend on the photon-number axis being right -- which is convenient,
    since the photon-number axis is not right until this calibration is done.
  * Delta_n0 is steep and monotonic near the crossing. With the parameters
    above, moving 134 kHz off the midpoint already changes n_bar by roughly a
    factor of two between the two states.

The zero crossing is located by a weighted straight-line fit through the
points that bracket it, not by taking the argmin of |Delta_n0| on the grid:
the grid spacing would otherwise set the resolution.

Use the rectangular pulse, not CLEAR. CLEAR's ring-up/ring-down ratios are a
solution of the linear-resonator model for an assumed drive detuning, so they
are not valid off the midpoint -- and a residual population that CLEAR has
deliberately nulled carries no information about the symmetry anyway.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clear_config import Setup, get_setup
from clear_analysis import RamseyModel, StateReference, fit_pair, save_figure

# qibolab and the sequence builders are imported inside run_midpoint_scan, so
# that the analysis can be re-run on a machine with no hardware stack.


def default_frequencies_mhz(setup: Setup, span_mhz: float | None = None, n: int = 13) -> np.ndarray:
    """Grid centred on the nominal midpoint.

    The default span is +/- chi: wide enough to bracket the crossing with room
    to spare, narrow enough that the readout contrast (and hence the Ramsey
    fringe) has not collapsed at the edges.
    """
    span = setup.chi_mhz if span_mhz is None else float(span_mhz)
    return setup.f_probe_mhz + np.linspace(-span, span, n)


def predicted_delta_n0(
    f_mhz: np.ndarray, setup: Setup, n_mid: float, f_mid_mhz: float | None = None
) -> np.ndarray:
    """n0_e - n0_g expected from the Lorentzian model, normalized so that both
    equal `n_mid` at the midpoint. Plotted alongside the data as a sanity
    check on the shape, not fitted."""
    f_mid = setup.f_probe_mhz if f_mid_mhz is None else f_mid_mhz
    half_kappa_sq = (setup.kappa_mhz / 2.0) ** 2
    delta = np.asarray(f_mhz, dtype=float) - f_mid
    lorentzian = lambda d: 1.0 / (half_kappa_sq + d**2)
    scale = n_mid / lorentzian(setup.chi_mhz)
    return scale * (lorentzian(delta - setup.chi_mhz) - lorentzian(delta + setup.chi_mhz))


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def run_midpoint_scan(
    setup: Setup,
    frequencies_mhz: np.ndarray | None = None,
    t_ramsey_values: np.ndarray | None = None,
    n_photons: float | None = None,
    output_name: str = "midpoint_calibration",
) -> Path:
    """Sweeps the probe frequency and, at each point, acquires everything
    needed for an unbiased n0 on both preparations.

    Per frequency this re-measures the |0>/|1> reference and the surviving
    excited fraction, because both depend on the probe frequency: the readout
    blobs move (so the projection axis changes) and the drive-induced
    transition rate changes with detuning. Reusing a single reference across
    the sweep would put a frequency-dependent contrast error straight into
    the quantity being nulled.
    """

    frequencies_mhz = (
        default_frequencies_mhz(setup) if frequencies_mhz is None else np.asarray(frequencies_mhz, float)
    )
    t_ramsey_values = (
        np.arange(0.0, 600.0, 6.0) if t_ramsey_values is None else np.asarray(t_ramsey_values, float)
    )
    from qibolab import create_platform
    from CLEAR_qibolab2 import run_excited_fraction, run_ramsey_scan, run_state_reference

    clear = setup.clear_pulse(n=n_photons)
    t_relax = 0.0

    platform = create_platform(setup.platform)
    platform.connect()
    payload: dict[str, np.ndarray] = {}
    references, fractions = [], []
    try:
        for f_mhz in frequencies_mhz:
            f_hz = float(f_mhz) * 1e6

            reference = run_state_reference(platform, setup, frequency_hz=f_hz)
            references.append([reference.iq_ground, reference.iq_excited])

            fractions.append(
                run_excited_fraction(
                    platform,
                    setup,
                    clear,
                    np.array([t_relax]),
                    reference,
                    rectangular=True,
                    frequency_hz=f_hz,
                )[t_relax]
            )

            for state in ("ground", "excited"):
                payload[f"{state}_f_{f_mhz:.6f}"] = run_ramsey_scan(
                    platform,
                    setup,
                    clear,
                    np.array([t_relax]),
                    t_ramsey_values,
                    prepare_excited=(state == "excited"),
                    rectangular=True,
                    frequency_hz=f_hz,
                )[t_relax]
    finally:
        platform.disconnect()

    payload["frequencies_mhz"] = frequencies_mhz
    payload["t_ramsey"] = t_ramsey_values
    payload["reference_iq"] = np.array(references)
    payload["p_excited"] = np.array(fractions)

    setup.data_dir.mkdir(parents=True, exist_ok=True)
    path = setup.data_dir / f"{output_name}.npz"
    np.savez(path, **payload)
    return path


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _fit_segment(
    x: np.ndarray, y: np.ndarray, yerr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted straight-line fit y = m*x + c, returning [m, c] and its 2x2
    covariance. Points with a non-finite value or a non-positive error are
    dropped."""
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if np.count_nonzero(good) < 3:
        raise RuntimeError("fewer than 3 usable points in the segment window")
    coeffs, cov = np.polyfit(x[good], y[good], 1, w=1.0 / yerr[good], cov=True)
    return coeffs, cov


def _segment_window(
    x: np.ndarray, delta: np.ndarray, window_mhz: float, min_points: int = 4
) -> np.ndarray:
    """Boolean mask selecting the points near the crossing.

    n0(f) is a Lorentzian, so it is only locally straight; the segments must
    be fitted over a window around the crossing rather than over the whole
    sweep, or the curvature in the wings pulls the intersection. The window is
    centred on the sign change of n0_e - n0_g (or, failing that, on its
    smallest absolute value) and defaults to +/- kappa/2, over which a
    Lorentzian is straight to a few percent.
    """
    finite = np.isfinite(delta)
    if not finite.any():
        raise RuntimeError("no usable points")

    sign_change = np.nonzero(np.diff(np.sign(delta[finite])))[0]
    if sign_change.size:
        i = int(sign_change[0])
        centre = 0.5 * (x[finite][i] + x[finite][i + 1])
    else:
        centre = float(x[finite][np.argmin(np.abs(delta[finite]))])

    mask = finite & (np.abs(x - centre) <= window_mhz)
    if np.count_nonzero(mask) < min_points:
        # Widen to the nearest `min_points` finite points instead of failing.
        order = np.argsort(np.abs(x - centre))
        mask = np.zeros_like(finite)
        mask[[i for i in order if finite[i]][:min_points]] = True
    return mask


def _segment_intersection(
    x: np.ndarray,
    y_ground: np.ndarray,
    err_ground: np.ndarray,
    y_excited: np.ndarray,
    err_excited: np.ndarray,
    window_mhz: float,
) -> dict:
    """Fits a straight segment to n0_g(f) and to n0_e(f) independently, then
    returns the frequency where the two segments meet.

    Near the midpoint the two branches are Lorentzians centred on opposite
    sides, so within the window they run with opposite slopes and cross
    steeply -- which is what makes the intersection well conditioned. The two
    fits are statistically independent, so their covariances simply add in the
    propagation below.

        u* = (c_e - c_g) / (m_g - m_e)

    with partial derivatives d/dc_e = 1/D, d/dc_g = -1/D, d/dm_g = -u*/D and
    d/dm_e = +u*/D, D = m_g - m_e.
    """
    mask = _segment_window(x, y_excited - y_ground, window_mhz)

    # Fit in a shifted coordinate so that the intercept is not extrapolated
    # from f = 0, which would make the covariance numerically useless.
    x0 = float(np.mean(x[mask]))
    u = x[mask] - x0

    (m_g, c_g), cov_g = _fit_segment(u, y_ground[mask], err_ground[mask])
    (m_e, c_e), cov_e = _fit_segment(u, y_excited[mask], err_excited[mask])

    denom = m_g - m_e
    if denom == 0:
        raise RuntimeError("the two segments are parallel; no intersection")

    u_star = (c_e - c_g) / denom
    var = (
        (cov_g[1, 1] + cov_e[1, 1]) / denom**2
        + (u_star / denom) ** 2 * (cov_g[0, 0] + cov_e[0, 0])
        + 2 * (u_star / denom**2) * (cov_g[0, 1] + cov_e[0, 1])
    )

    return {
        "f_mhz": float(x0 + u_star),
        "f_err_mhz": float(np.sqrt(max(var, 0.0))),
        "n0_at_crossing": float(c_g + m_g * u_star),
        "slope_ground": float(m_g),
        "slope_excited": float(m_e),
        "x0": x0,
        "line_ground": (float(m_g), float(c_g)),
        "line_excited": (float(m_e), float(c_e)),
        "mask": mask,
        "same_sign_slopes": bool(m_g * m_e > 0),
        "extrapolated": bool(u_star < u.min() or u_star > u.max()),
    }


def analyze_midpoint_scan(
    npz_path: Path,
    setup: Setup,
    output_path: Path | None = None,
    window_mhz: float | None = None,
) -> dict:
    """Fits n0 for both preparations at every probe frequency, fits a straight
    segment to each branch near the crossing, and returns the frequency where
    the two segments meet.

    `window_mhz` is the half-width of the segment window (default kappa/2).
    """

    data = np.load(npz_path)
    model = RamseyModel.from_setup(setup)
    frequencies = data["frequencies_mhz"]
    t_ramsey = data["t_ramsey"]

    n0 = {state: np.full(frequencies.shape, np.nan) for state in ("ground", "excited")}
    err = {state: np.full(frequencies.shape, np.nan) for state in ("ground", "excited")}
    contrast = np.full(frequencies.shape, np.nan)

    for i, f_mhz in enumerate(frequencies):
        reference = StateReference(*data["reference_iq"][i])
        contrast[i] = reference.contrast
        keys = {s: f"{s}_f_{f_mhz:.6f}" for s in ("ground", "excited")}
        if not all(k in data for k in keys.values()):
            continue
        traces = {s: reference.project(data[k]) for s, k in keys.items()}
        try:
            fits = fit_pair(
                t_ramsey, traces["ground"], traces["excited"], model,
                p_excited=float(data["p_excited"][i]),
            )
        except RuntimeError:
            print(f"  f = {f_mhz:.6f} MHz: fit failed, skipped")
            continue
        for state in ("ground", "excited"):
            n0[state][i] = fits[state]["n0"]
            err[state][i] = fits[state]["n0_err"]
        print(
            f"  f = {f_mhz:.6f} MHz   n0_g = {n0['ground'][i]:6.3f}   "
            f"n0_e = {n0['excited'][i]:6.3f}   "
            f"delta = {n0['excited'][i] - n0['ground'][i]:+6.3f}   "
            f"p_e = {data['p_excited'][i]:.3f}"
        )

    delta = n0["excited"] - n0["ground"]
    delta_err = np.hypot(err["excited"], err["ground"])

    crossing = _segment_intersection(
        frequencies, n0["ground"], err["ground"], n0["excited"], err["excited"],
        window_mhz=window_mhz if window_mhz is not None else setup.kappa_mhz / 2.0,
    )
    f_opt, f_opt_err = crossing["f_mhz"], crossing["f_err_mhz"]
    mask = crossing["mask"]

    # --- figure ------------------------------------------------------------
    fig, (ax_n0, ax_delta) = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)

    f_seg = np.linspace(frequencies[mask].min(), frequencies[mask].max(), 200)
    for state, color, line_key in (
        ("ground", "tab:blue", "line_ground"),
        ("excited", "tab:red", "line_excited"),
    ):
        ax_n0.errorbar(frequencies, n0[state], err[state], marker="o", ls="none",
                       color=color, capsize=3, label=f"$n_0$ {state}")
        ax_n0.errorbar(frequencies[mask], n0[state][mask], err[state][mask],
                       marker="o", ls="none", color=color, capsize=3,
                       markerfacecolor="white", markeredgewidth=1.4)
        m, c = crossing[line_key]
        ax_n0.plot(f_seg, m * (f_seg - crossing["x0"]) + c, color=color, lw=1.6)

    ax_n0.plot(f_opt, crossing["n0_at_crossing"], marker="*", ms=16, color="black",
               ls="none", zorder=5,
               label=f"crossing {f_opt:.4f} $\\pm$ {f_opt_err * 1e3:.0f} kHz")
    ax_n0.axvline(setup.f_probe_mhz, color="grey", ls="--", lw=0.8,
                  label=f"TOML midpoint {setup.f_probe_mhz:.4f}")
    ax_n0.set_ylabel(r"$n_0$ at $t_{relax}=0$")
    ax_n0.set_title("Probe midpoint from the crossing of the two segments")
    ax_n0.legend(fontsize=8)
    ax_n0.grid(alpha=0.3)

    # Difference panel: a residual diagnostic only. The number above comes
    # from the intersection of the two independent segments, not from this.
    ax_delta.errorbar(frequencies, delta, delta_err, marker="o", ls="none",
                      color="tab:purple", capsize=3, label=r"$n_0^e-n_0^g$")
    finite = np.isfinite(delta)
    if finite.any():
        n_mid = crossing["n0_at_crossing"]
        f_fine = np.linspace(frequencies.min(), frequencies.max(), 400)
        ax_delta.plot(f_fine, predicted_delta_n0(f_fine, setup, n_mid, f_opt),
                      "k:", lw=1, label="Lorentzian model")
    ax_delta.axhline(0.0, color="grey", lw=0.8)
    ax_delta.axvline(f_opt, color="black", lw=1.2)
    ax_delta.set_xlabel("probe frequency (MHz)")
    ax_delta.set_ylabel(r"$\Delta n_0$ (diagnostic)")
    ax_delta.legend(fontsize=8)
    ax_delta.grid(alpha=0.3)

    save_figure(fig, output_path or npz_path.with_name("midpoint_calibration.png"))

    shift_khz = (f_opt - setup.f_probe_mhz) * 1e3
    print(f"\nsegment slopes     ground {crossing['slope_ground']:+.3f}, "
          f"excited {crossing['slope_excited']:+.3f} photons/MHz "
          f"({np.count_nonzero(mask)} points each)")
    print(f"measured midpoint  {f_opt:.6f} +/- {f_opt_err * 1e3:.0f} kHz (1 sigma)")
    print(f"TOML midpoint      {setup.f_probe_mhz:.6f} MHz  ({shift_khz:+.0f} kHz away)")

    if crossing["same_sign_slopes"]:
        print("WARNING: the two segments have slopes of the same sign. Near the "
              "midpoint they should run opposite ways; the window is probably off "
              "the crossing, or one branch is not being fitted reliably.")
    if crossing["extrapolated"]:
        print("WARNING: the crossing lies outside the fitted window, so it is "
              "extrapolated. Re-run the sweep centred on the value above.")
    if abs(shift_khz) > 0.05 * setup.kappa_mhz * 1e3:
        print("Update f_res_g_MHz / f_res_e_MHz in risonatore.toml so that their "
              "midpoint matches, then re-run the Stark calibration at the new "
              "frequency: k was measured at the old one.")
    if np.nanmax(contrast) > 0 and np.nanmin(contrast) < 0.3 * np.nanmax(contrast):
        print("Note: readout contrast varies by more than 3x across the sweep; the "
              "outermost points carry little weight.")

    return {
        "frequencies_mhz": frequencies,
        "n0": n0,
        "n0_err": err,
        "delta": delta,
        "delta_err": delta_err,
        "f_midpoint_mhz": f_opt,
        "f_midpoint_err_mhz": f_opt_err,
        "crossing": crossing,
    }


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    setup = get_setup(platform="2q_chip_thesis", qubit=1)

    print(f"nominal midpoint {setup.f_probe_mhz:.6f} MHz, sweeping +/- {setup.chi_mhz:.3f} MHz")

    npz_path = setup.data_dir / "midpoint" / "midpoint_calibration.npz"
    if True:#not npz_path.exists():
        npz_path = run_midpoint_scan(setup, t_ramsey_values=np.arange(0.0, 600.0, 6.0), frequencies_mhz=np.linspace(7583.1175-0.1, 7583.1175+0.1, 5, endpoint=True))

    analyze_midpoint_scan(npz_path, setup)

