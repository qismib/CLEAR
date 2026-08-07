"""Ramsey-trace analysis for the CLEAR experiment.

Eq. (1) of McClure et al., Phys. Rev. Applied 5, 011001 (2016):

    S(t_R) = 1/2 [1 - Im{exp[-(G2 + i*D)*t_R + i*(phi0 - 2*n0*chi*tau(t_R))]}]
    tau(t_R) = (1 - exp(-(kappa + 2i*chi)*t_R)) / (kappa + 2i*chi)

Two deliberate departures from the paper's "only n0 and phi0 are free":

1. `scale` and `offset` are free. Eq. (1) returns a probability in [0, 1],
   while the data are an integrated quadrature in arbitrary units whose
   contrast depends on readout fidelity and on how much of the excited
   preparation survived M1. With scale and offset frozen, all of that has
   nowhere to go except into n0. `scale` is constrained positive: the sign of
   the fringe is carried by phi0, which is free, so allowing both would only
   add a degeneracy.

2. For an excited preparation, the model is a two-component mixture. What sets
   the sign of the fringe is the qubit state at the *first* X90, not at
   preparation, so trajectories that decayed during M1 contribute a fringe
   shifted by pi. Their photon number is the ground-state one, because the
   cavity re-equilibrates in ~1/kappa << M1:

       S_e = p*S(n0_e, phi0) + (1-p)*S(n0_g, phi0 + pi)

   with p the measured surviving excited fraction and n0_g taken from the
   independently fitted ground trace. Eq. (1) is nonlinear in n0 (n0 sits
   inside a complex exponential), so fitting a single-n0 model to this mixture
   does not return the mean -- it returns a biased n0.

All rates are in rad/ns and all times in ns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from clear_config import Setup, mhz_to_rad_per_ns  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# Turning raw acquisitions into a trace
# ---------------------------------------------------------------------------


def to_complex(iq: np.ndarray) -> np.ndarray:
    """Normalizes the several shapes qibolab returns into a complex array
    with the shot axis first."""
    iq = np.asarray(iq)
    if not np.iscomplexobj(iq) and iq.ndim >= 2 and iq.shape[-1] == 2:
        iq = iq[..., 0] + 1j * iq[..., 1]
    return np.atleast_2d(iq)


def shot_average(iq: np.ndarray) -> np.ndarray:
    """Mean IQ point per t_ramsey, as a complex trace."""
    return to_complex(iq).mean(axis=0)


@dataclass(frozen=True)
class StateReference:
    """Mean IQ points for a bare |0> and |1> preparation, used to turn raw
    quadratures into an excited-state probability.

    Projecting onto the line joining the two blobs removes the arbitrary
    rotation, gain and background of the acquisition chain, so `scale` and
    `offset` in the fit become small corrections rather than the dominant
    free parameters -- and, more importantly, it makes the excited fraction
    measurable on the same axis.
    """

    iq_ground: complex
    iq_excited: complex

    @classmethod
    def from_acquisitions(cls, iq_ground, iq_excited) -> "StateReference":
        return cls(complex(shot_average(iq_ground).mean()), complex(shot_average(iq_excited).mean()))

    def project(self, iq: np.ndarray) -> np.ndarray:
        """Raw acquisition -> excited-state probability (0 = |0>, 1 = |1>)."""
        axis = self.iq_excited - self.iq_ground
        if axis == 0:
            raise ValueError("ground and excited reference points coincide")
        return np.real((shot_average(iq) - self.iq_ground) * np.conj(axis)) / abs(axis) ** 2

    @property
    def contrast(self) -> float:
        return abs(self.iq_excited - self.iq_ground)


def excited_fraction_from_reference(iq_after_m1, reference: StateReference) -> float:
    """Surviving excited fraction p, from a sequence identical to the Ramsey
    one but with the two X90 pulses removed."""
    return float(np.mean(reference.project(iq_after_m1)))


# ---------------------------------------------------------------------------
# Eq. (1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RamseyModel:
    """Eq. (1) with the resonator and qubit parameters held fixed."""

    kappa: float
    chi: float
    detuning: float
    gamma2: float

    @classmethod
    def from_setup(cls, setup: Setup) -> "RamseyModel":
        return cls(
            kappa=setup.kappa_rad_ns,
            chi=setup.chi_rad_ns,
            detuning=setup.detuning_rad_ns,
            gamma2=setup.gamma2_rad_ns,
        )

    def tau(self, t_r: np.ndarray) -> np.ndarray:
        lam = self.kappa + 2j * self.chi
        return (1.0 - np.exp(-lam * t_r)) / lam

    def signal(self, t_r: np.ndarray, n0: float, phi0: float) -> np.ndarray:
        """S(t_R), in [0, 1]."""
        t_r = np.asarray(t_r, dtype=float)
        expo = -(self.gamma2 + 1j * self.detuning) * t_r + 1j * (
            phi0 - 2.0 * n0 * self.chi * self.tau(t_r)
        )
        return 0.5 * (1.0 - np.imag(np.exp(expo)))

    def mixture(
        self, t_r: np.ndarray, n0: float, phi0: float, p: float, n0_decayed: float
    ) -> np.ndarray:
        """Excited-preparation signal with a fraction (1-p) having decayed to
        |g> before the first X90, contributing an inverted fringe."""
        if p >= 1.0:
            return self.signal(t_r, n0, phi0)
        return p * self.signal(t_r, n0, phi0) + (1.0 - p) * self.signal(
            t_r, n0_decayed, phi0 + np.pi
        )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_ramsey(
    t_r: np.ndarray,
    y: np.ndarray,
    model: RamseyModel,
    p_excited: float = 1.0,
    n0_decayed: float = 0.0,
    sigma: np.ndarray | None = None,
) -> dict:
    """Fits n0, phi0, scale and offset of Eq. (1) to one trace.

    p_excited = 1 (the default, and the only sensible value for a ground
    preparation) reduces the model to plain Eq. (1) with a free contrast.
    For an excited preparation pass the measured surviving fraction and the
    n0 obtained from the corresponding ground trace.

    Returns n0, phi0, scale, offset with 1-sigma errors, plus the reduced
    chi-square and the fitted curve evaluator.
    """

    t_r = np.asarray(t_r, dtype=float)
    y = np.asarray(y, dtype=float)
    p_excited = float(np.clip(p_excited, 1e-6, 1.0))

    def model_fn(t, n0, phi0, scale, offset):
        return offset + scale * model.mixture(t, n0, phi0, p_excited, n0_decayed)

    span = float(np.ptp(y))
    guesses = [
        (n0, phi0, span, float(np.min(y)))
        for n0 in (0.2, 1.0, 3.0)
        for phi0 in np.linspace(-np.pi, np.pi, 8, endpoint=False)
    ]
    bounds = ([0.0, -4 * np.pi, 0.0, -np.inf], [np.inf, 4 * np.pi, np.inf, np.inf])

    best = None
    for p0 in guesses:
        try:
            popt, pcov = curve_fit(
                model_fn, t_r, y, p0=p0, sigma=sigma, bounds=bounds, maxfev=20_000
            )
        except (RuntimeError, ValueError):
            continue
        residual = float(np.sum((y - model_fn(t_r, *popt)) ** 2))
        if best is None or residual < best[0]:
            best = (residual, popt, pcov)

    if best is None:
        raise RuntimeError("Eq. (1) fit did not converge from any starting guess")

    residual, popt, pcov = best
    perr = np.sqrt(np.diag(pcov))
    dof = max(t_r.size - popt.size, 1)

    n0, phi0, scale, offset = popt
    return {
        "n0": float(n0),
        "n0_err": float(perr[0]),
        "phi0": float(np.arctan2(np.sin(phi0), np.cos(phi0))),
        "phi0_err": float(perr[1]),
        "scale": float(scale),
        "scale_err": float(perr[2]),
        "offset": float(offset),
        "offset_err": float(perr[3]),
        "chi2_red": residual / dof,
        "p_excited": p_excited,
        "n0_decayed": float(n0_decayed),
        "curve": lambda t: model_fn(np.asarray(t, dtype=float), *popt),
    }


def fit_pair(
    t_r: np.ndarray,
    y_ground: np.ndarray,
    y_excited: np.ndarray,
    model: RamseyModel,
    p_excited: float,
) -> dict[str, dict]:
    """Fits a ground/excited pair taken under the same conditions.

    The ground trace is fitted first with plain Eq. (1); its n0 is then held
    as the photon number of the decayed sub-ensemble in the excited fit. This
    is the whole T1 correction, and it needs the two traces together, which is
    why it lives in one function.
    """
    fit_g = fit_ramsey(t_r, y_ground, model)
    fit_e = fit_ramsey(t_r, y_excited, model, p_excited=p_excited, n0_decayed=fit_g["n0"])
    return {"ground": fit_g, "excited": fit_e}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ramsey(t_r, y, fit: dict, title: str = "", ax=None, color: str = "tab:red"):
    """Fig. 2(a) style: points for the data, solid line for the Eq. (1) fit."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(t_r, y, s=18, color=color, alpha=0.8, label="Data")
    t_fine = np.linspace(float(np.min(t_r)), float(np.max(t_r)), 600)
    label = f"Eq. (1): $n_0$ = {fit['n0']:.3f} $\\pm$ {fit['n0_err']:.3f}"
    if fit["p_excited"] < 1.0:
        label += f"\n($p_e$ = {fit['p_excited']:.2f} corrected)"
    ax.plot(t_fine, fit["curve"](t_fine), color="black", lw=1.8, label=label)
    ax.set_xlabel(r"Ramsey delay $t_R$ (ns)")
    ax.set_ylabel(r"$P_e$")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize(label: str, fit: dict) -> str:
    return (
        f"{label:>28s}:  n0 = {fit['n0']:7.4f} +/- {fit['n0_err']:.4f}   "
        f"phi0 = {fit['phi0']:+.3f}   contrast = {fit['scale']:.3f}   "
        f"chi2_red = {fit['chi2_red']:.3g}"
    )


# ---------------------------------------------------------------------------
# n0 vs t_relax: exponential-decay fit (Fig. 2(b) style)
# ---------------------------------------------------------------------------


def fit_relaxation(
    t_relax: np.ndarray, n0: np.ndarray, n0_err: np.ndarray | None = None
) -> dict:
    """Fits n0(t_relax) = n0_0 * exp(-(t_relax - t0)/T_cav) + offset.

    This is the fit McClure et al. use for Fig. 2(b): "the decay is
    exponential ... and the time constant Tcav extracted from the best-fit
    curve is consistent with the value of kappa obtained from frequency-domain
    measurements." T_cav is left free rather than fixed to 1/kappa, so it acts
    as an independent cross-check of the frequency-domain kappa. `offset`
    absorbs a small constant background (residual thermal population,
    fit-floor bias) instead of forcing the tail to zero.
    """
    t_relax = np.asarray(t_relax, dtype=float)
    n0 = np.asarray(n0, dtype=float)
    if t_relax.size < 4:
        raise RuntimeError("need at least 4 points to fit an exponential decay")
    t0 = float(t_relax[0])

    def model(t, n0_0, t_cav, offset):
        return n0_0 * np.exp(-(t - t0) / t_cav) + offset

    p0 = [max(float(n0[0] - np.min(n0)), 1e-3), 150.0, float(np.min(n0))]
    bounds = ([0.0, 1.0, -np.inf], [np.inf, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(
            model,
            t_relax,
            n0,
            p0=p0,
            sigma=n0_err,
            absolute_sigma=n0_err is not None,
            bounds=bounds,
            maxfev=20_000,
        )
    except RuntimeError as exc:
        raise RuntimeError("exponential fit of n0 vs t_relax did not converge") from exc

    perr = np.sqrt(np.diag(pcov))
    n0_0, t_cav, offset = popt
    t_cav_err = float(perr[1])

    # kappa = 1/T_cav is the emptying rate of the resonator (rad/ns, i.e. the
    # same convention as Setup.kappa_rad_ns). Propagated from T_cav's error
    # via d(1/T)/dT = -1/T^2.
    kappa_rad_ns = 1.0 / t_cav
    kappa_rad_ns_err = t_cav_err / t_cav**2
    # rad/ns -> MHz: f_MHz = kappa_rad_ns / (2*pi*1e-3), inverse of
    # clear_config.mhz_to_rad_per_ns.
    kappa_mhz = kappa_rad_ns / (2.0 * np.pi * 1e-3)
    kappa_mhz_err = kappa_rad_ns_err / (2.0 * np.pi * 1e-3)

    return {
        "n0_0": float(n0_0),
        "n0_0_err": float(perr[0]),
        "t_cav": float(t_cav),
        "t_cav_err": t_cav_err,
        "offset": float(offset),
        "offset_err": float(perr[2]),
        "kappa_rad_ns": float(kappa_rad_ns),
        "kappa_rad_ns_err": float(kappa_rad_ns_err),
        "kappa_mhz": float(kappa_mhz),
        "kappa_mhz_err": float(kappa_mhz_err),
        "t0": t0,
        "curve": lambda t: model(np.asarray(t, dtype=float), *popt),
    }


# ---------------------------------------------------------------------------
# Whole-run analysis (no hardware dependency: operates on a saved .npz)
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

    Pure post-processing: only reads `npz_path` and `setup`, no platform
    connection, so it works on data acquired in an earlier session (e.g. the
    files already saved under `setup.data_dir`).
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


def plot_n0_decay(
    results: dict, setup: Setup, output_path: Path, fit: bool = True
) -> dict[str, dict]:
    """n0 versus t_relax for both preparations (Fig. 2(b) of the paper).

    If `fit=True` (default), an independent exponential decay is fitted to
    each state's n0(t_relax) via `fit_relaxation` and drawn as a solid curve,
    with T_cav (and its uncertainty) annotated in the legend -- mirroring the
    "Tcav = 141 ns" / "Tcav = 143 ns" labels of Fig. 2(b). If fitting is
    disabled or fails for both states, falls back to the theoretical
    exp(-kappa*t) curve anchored to the first ground-state point.

    Returns the per-state fit results (empty dict if none succeeded), so
    T_cav can be inspected or logged by the caller.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    t_relax = np.array(sorted(results))
    fits: dict[str, dict] = {}

    for state, color in (("ground", "tab:blue"), ("excited", "tab:red")):
        n0 = np.array([results[t][state]["n0"] for t in t_relax])
        err = np.array([results[t][state]["n0_err"] for t in t_relax])
        ax.errorbar(t_relax, n0, err, marker="o", ls="none", color=color, label=state, capsize=3)

        if fit:
            try:
                fit_result = fit_relaxation(t_relax, n0, err)
            except RuntimeError as exc:
                print(f"  [warn] {state}: {exc}")
                continue
            fits[state] = fit_result
            t_fine = np.linspace(t_relax.min(), t_relax.max(), 300)
            ax.plot(
                t_fine,
                fit_result["curve"](t_fine),
                color=color,
                lw=1.5,
                label=(
                    rf"{state} fit: $T_{{cav}}$ = {fit_result['t_cav']:.0f} "
                    rf"$\pm$ {fit_result['t_cav_err']:.0f} ns"
                    "\n"
                    rf"   $\kappa/2\pi$ = {fit_result['kappa_mhz']:.3f} "
                    rf"$\pm$ {fit_result['kappa_mhz_err']:.3f} MHz"
                ),
            )

    if not fits and t_relax.size > 1:
        t_fine = np.linspace(t_relax.min(), t_relax.max(), 300)
        n0_first = results[float(t_relax[0])]["ground"]["n0"]
        ax.plot(
            t_fine,
            n0_first * np.exp(-setup.kappa_rad_ns * (t_fine - t_relax[0])),
            "k--",
            lw=1,
            label=r"$e^{-\kappa t}$ (theory, no fit)",
        )

    ax.set_xlabel(r"$t_{relax}$ (ns)")
    ax.set_ylabel(r"residual photons $n_0$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save_figure(fig, output_path)
    return fits