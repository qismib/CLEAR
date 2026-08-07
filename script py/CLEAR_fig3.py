"""Fig. 3(c) of McClure et al. (2016): residual cavity population n0 versus
normalized drive power P_norm, for a square (+delay) M1 and for CLEAR, with
the qubit prepared in the ground or the excited state.

Sequence, for every P_norm / pulse type / state:

    (RX) -> M1 -> t_relax -> RX90 - t_R - RX90 -> t_buffer -> M2

with t_relax = 0 for CLEAR and t_relax = 2*t_kick for the square pulse ("a
delay of approximately 300 ns is inserted to match the total length of the
CLEAR pulse's two ring-down segments"). Both are built by the same
`build_ramsey_sequence` as the t_relax scan, so there is one sequence
definition in the project rather than two that can drift apart.

P_norm = P/P_1ph is a power, so the amplitude at a given P_norm is
amplitude_1ph * sqrt(P_norm), with amplitude_1ph from the Stark calibration in
clear_config. The CLEAR ratios, being linear in the field, apply on top.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clear_config import Setup, get_setup
from clear_analysis import (
    RamseyModel,
    StateReference,
    fit_pair,
    save_figure,
    summarize,
)
# qibolab and the sequence builders are imported inside run_power_scan, so
# that analyze_power_scan can be re-run without the hardware stack.


PULSE_TYPES = ("clear", "square")

STYLES = {
    ("square", "ground"): dict(color="tab:blue", marker="o", ls="--", label="Square + delay, ground"),
    ("square", "excited"): dict(color="tab:red", marker="o", ls="--", label="Square + delay, excited"),
    ("clear", "ground"): dict(color="tab:blue", marker="s", ls="-", label="CLEAR, ground"),
    ("clear", "excited"): dict(color="tab:red", marker="s", ls="-", label="CLEAR, excited"),
}


def _t_relax_for(pulse_type: str, setup: Setup) -> float:
    """CLEAR's Ramsey starts immediately; the square pulse gets a delay equal
    to CLEAR's two ring-down segments, so both have the same total M1 length
    (and, importantly, the same T1 exposure)."""
    return 0.0 if pulse_type == "clear" else setup.clear_pulse().ringdown_duration


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def run_power_scan(
    setup: Setup,
    pnorm_values: np.ndarray,
    t_ramsey_values: np.ndarray,
    output_name: str = "fig3c",
) -> Path:
    """Acquires the whole Fig. 3(c) dataset in one platform session:
    |0>/|1> reference, the surviving excited fraction for each pulse type, and
    a Ramsey trace for every (P_norm, pulse type, state)."""

    from qibolab import create_platform
    from CLEAR_qibolab2 import run_excited_fraction, run_ramsey_scan, run_state_reference

    pnorm_values = np.asarray(pnorm_values, dtype=float)
    t_ramsey_values = np.asarray(t_ramsey_values, dtype=float)

    platform = create_platform(setup.platform)
    platform.connect()
    payload: dict[str, np.ndarray] = {}
    try:
        reference = run_state_reference(platform, setup)
        p_measured = {}

        for pulse_type in PULSE_TYPES:
            rectangular = pulse_type == "square"
            t_relax = _t_relax_for(pulse_type, setup)

            # p depends on the M1 length and power, so it is measured per
            # pulse type -- at the strongest power, where the drive-induced
            # contribution to the decay is largest.
            clear_max = setup.clear_pulse(n=float(np.max(pnorm_values)))
            p_measured[pulse_type] = run_excited_fraction(
                platform, setup, clear_max, np.array([t_relax]), reference, rectangular
            )[t_relax]

            for pnorm in pnorm_values:
                clear = setup.clear_pulse(n=float(pnorm))
                for state in ("ground", "excited"):
                    trace = run_ramsey_scan(
                        platform,
                        setup,
                        clear,
                        np.array([t_relax]),
                        t_ramsey_values,
                        prepare_excited=(state == "excited"),
                        rectangular=rectangular,
                    )[t_relax]
                    payload[f"{pulse_type}_{state}_pnorm_{pnorm:g}"] = trace

        payload["p_excited"] = np.array([p_measured[pt] for pt in PULSE_TYPES])
    finally:
        platform.disconnect()

    payload["pnorm"] = pnorm_values
    payload["t_ramsey"] = t_ramsey_values
    payload["reference_iq"] = np.array([reference.iq_ground, reference.iq_excited])

    setup.data_dir.mkdir(parents=True, exist_ok=True)
    path = setup.data_dir / f"{output_name}.npz"
    np.savez(path, **payload)
    return path


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_power_scan(npz_path: Path, setup: Setup, output_path: Path | None = None) -> dict:
    """Fits Eq. (1) to every trace and reproduces Fig. 3(c).

    Ground and excited are fitted as a pair at each P_norm, so the excited fit
    can hold the decayed sub-ensemble at the ground-state photon number.
    """

    data = np.load(npz_path)
    model = RamseyModel.from_setup(setup)
    reference = StateReference(*data["reference_iq"])
    pnorm_values = data["pnorm"]
    t_ramsey = data["t_ramsey"]

    curves: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for j, pulse_type in enumerate(PULSE_TYPES):
        p = float(data["p_excited"][j]) if "p_excited" in data else setup.excited_fraction(
            _t_relax_for(pulse_type, setup), pulse_type == "square"
        )
        n0 = {state: np.full(pnorm_values.shape, np.nan) for state in ("ground", "excited")}
        err = {state: np.full(pnorm_values.shape, np.nan) for state in ("ground", "excited")}

        for i, pnorm in enumerate(pnorm_values):
            keys = {s: f"{pulse_type}_{s}_pnorm_{pnorm:g}" for s in ("ground", "excited")}
            if not all(k in data for k in keys.values()):
                continue
            traces = {s: reference.project(data[k]) for s, k in keys.items()}
            try:
                fits = fit_pair(t_ramsey, traces["ground"], traces["excited"], model, p_excited=p)
            except RuntimeError:
                continue
            for state in ("ground", "excited"):
                n0[state][i] = fits[state]["n0"]
                err[state][i] = fits[state]["n0_err"]
                print(summarize(f"{pulse_type} {state} P={pnorm:g}", fits[state]))

        for state in ("ground", "excited"):
            curves[(pulse_type, state)] = {"n0": n0[state], "n0_err": err[state], "p_excited": p}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key, style in STYLES.items():
        if key not in curves:
            continue
        ax.errorbar(pnorm_values, curves[key]["n0"], curves[key]["n0_err"], capsize=3, **style)
    ax.set_xlabel(r"Normalized drive power $P_{\mathrm{norm}}$")
    ax.set_ylabel(r"Residual population $n_0$")
    ax.set_title("Fig. 3(c): residual population vs drive power")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    save_figure(fig, output_path or npz_path.with_name("fig3c_n0_vs_pnorm.png"))

    return curves


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    setup = get_setup(platform="2q_chip_thesis", qubit=1)
    t_ramsey = np.arange(0.0, 600.0, 6.0)

    npz_path = setup.data_dir / "fig3c.npz"
    if True:#not npz_path.exists():
        npz_path = run_power_scan(
            setup,
            pnorm_values=np.linspace(0.5, 50, 30),
            t_ramsey_values=t_ramsey,
        )

    analyze_power_scan(npz_path, setup)

    # from CLEAR_qibolab2 import *

    # setup = get_setup(platform="2q_chip_thesis", qubit=1)

    # print(f"probe at midpoint      {setup.f_probe_mhz:.6f} MHz")
    # print(f"kappa/2pi = {setup.kappa_mhz:.4f} MHz   chi/2pi = {setup.chi_mhz:.4f} MHz")
    # print(f"n_target = {setup.n_target:g}  ->  amplitude {setup.amplitude_for_n(setup.n_target):.4f}")
    # print(f"M1 exposure {setup.exposure_ns(0.0):.0f} ns, T1 = {setup.t1_ns:.0f} ns")

    # t_relax = np.arange(0, 1200, 50, dtype=float)

    # for rectangular in (True, False):
    #     name = "square" if rectangular else "clear"
    #     npz_path = setup.data_dir / f"run_{'square' if rectangular else 'clear'}.npz"

    #     npz_path = run_experiment(
    #         setup, t_relax, t_ramsey, rectangular=rectangular, output_name="run"
    #     )

    #     print(f"\n=== {name} M1 ===")
    #     results = analyze_run(npz_path, setup)
    #     plot_n0_decay(results, setup, setup.data_dir / f"n0_vs_t_relax_{name}.png")