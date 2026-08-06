"""Fig. 2(b)-style residual-population scan for CLEAR_qibolab.

This script measures the residual cavity population n0 as a function of the
wait time after M1, t_relax. It can use either a square M1 pulse
(`rectangular=True`) or the full 5-segment CLEAR pulse (`rectangular=False`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from qibolab import (
    AcquisitionType,
    AveragingMode,
    create_platform,
)

from CLEAR_qibolab import (
    ClearPulseParameters,
    build_clear_ramsey_sequence,
    clear_params_from_ratios,
    load_clear_ratios,
    ramsey_sweepers,
    readout_amplitude_for_n_target,
    rectangular_params,
)

from CLEAR_qibolab_analysis import (
    fit_ramsey_eq1,
    fixed_ramsey_parameters,
    normalized_amplitude,
    ramsey_signal,
)


_STATE_STYLES = {
    "ground": dict(color="tab:blue", marker="o", label="Ground"),
    "excited": dict(color="tab:red", marker="o", label="Excited"),
}

def build_fig2b_sequence(
    platform,
    qubit: int,
    clear: ClearPulseParameters,
    t_relax: float,
    t_ramsey: float,
    t_buffer: float = 400.0,
    prepare_excited: bool = False,
    rectangular: bool = True,
) -> tuple[Any, Any, Any, int]:
    """Build M1 -> t_relax -> Ramsey -> t_buffer -> M2.

    `rectangular=True` uses only the steady square M1 segment. `False` uses
    the full CLEAR pulse. The Ramsey tail itself is imported from
    CLEAR_qibolab.py, so Fig. 2 uses the same sequence definition as every
    other CLEAR script.
    """

    return build_clear_ramsey_sequence(
        platform=platform,
        qubit=qubit,
        clear=clear,
        t_relax=t_relax,
        t_ramsey=t_ramsey,
        t_buffer=t_buffer,
        prepare_excited=prepare_excited,
        rectangular=rectangular,
    )


def run_fig2b_scan(
    platform_name: str,
    qubit: int,
    clear: ClearPulseParameters,
    t_relax_values: Iterable[float],
    t_ramsey_values: Iterable[float],
    output: Path,
    nshots: int = 1024,
    relaxation_time: float = 100_000,
    ramsey_detuning: float | None = 10_000_000,
    rectangular: bool = True,
) -> Path:
    """Acquire Fig. 2(b)-style data for ground and excited preparations."""

    output.mkdir(parents=True, exist_ok=True)
    t_relax_values = np.asarray(list(t_relax_values), dtype=float)
    t_ramsey_values = np.asarray(list(t_ramsey_values), dtype=float)

    if t_relax_values.size == 0:
        raise ValueError("t_relax_values cannot be empty.")
    if t_ramsey_values.size == 0:
        raise ValueError("t_ramsey_values cannot be empty.")

    platform = create_platform(platform_name)
    all_results = {}

    platform.connect()
    try:
        for prepare_excited in (False, True):
            state = "excited" if prepare_excited else "ground"

            for t_relax in t_relax_values:
                (
                    sequence,
                    ramsey_delay,
                    second_rx90_pulse,
                    m2_id,
                ) = build_fig2b_sequence(
                    platform=platform,
                    qubit=qubit,
                    clear=clear,
                    t_relax=float(t_relax),
                    t_ramsey=float(t_ramsey_values[0]),
                    prepare_excited=prepare_excited,
                    rectangular=rectangular,
                )

                sweepers = ramsey_sweepers(
                    ramsey_delay=ramsey_delay,
                    second_rx90_pulse=second_rx90_pulse,
                    t_ramsey_values=t_ramsey_values,
                    ramsey_detuning=ramsey_detuning,
                )

                results = platform.execute(
                    [sequence],
                    [sweepers],
                    nshots=nshots,
                    relaxation_time=relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                    averaging_mode=AveragingMode.SINGLESHOT,
                )

                all_results[f"{state}_t_relax_{t_relax:g}"] = results[m2_id]

    finally:
        platform.disconnect()

    pulse_label = "square" if rectangular else "clear"
    npz_path = output / f"fig2b_{pulse_label}.npz"
    np.savez(
        npz_path,
        t_relax=t_relax_values,
        t_ramsey=t_ramsey_values,
        rectangular=rectangular,
        **all_results,
    )

    return npz_path


def exponential_decay(
    t: np.ndarray, offset: float, amplitude: float, tau: float
) -> np.ndarray:
    """Simple decay model used only to guide the eye in the Fig. 2(b) plot."""

    return offset + amplitude * np.exp(-np.asarray(t, dtype=float) / tau)


def fit_decay_curve(
    t_relax: np.ndarray, n0_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit n0(t_relax) to offset + amplitude * exp(-t_relax / tau)."""

    mask = np.isfinite(n0_values)
    if np.count_nonzero(mask) < 3:
        return None

    t_fit = np.asarray(t_relax[mask], dtype=float)
    y_fit = np.asarray(n0_values[mask], dtype=float)
    span = max(float(t_fit.max() - t_fit.min()), 1.0)
    p0 = (
        max(float(y_fit.min()), 0.0),
        max(float(y_fit.max() - y_fit.min()), 1e-6),
        span,
    )

    try:
        popt, pcov = curve_fit(
            exponential_decay,
            t_fit,
            y_fit,
            p0=p0,
            bounds=([0.0, 0.0, 1e-9], [np.inf, np.inf, np.inf]),
            maxfev=10_000,
        )
    except (RuntimeError, ValueError):
        return None

    return popt, pcov


def analyze_fig2b(
    npz_path: Path,
    kappa_mhz: float,
    chi_mhz: float,
    ramsey_detuning_mhz: float,
    t2_echo_ns: float,
    output_path: Path | None = None,
    title: str | None = None,
) -> dict:
    """Fit every Ramsey trace and plot n0 versus t_relax."""

    kappa, chi, detuning, gamma2 = fixed_ramsey_parameters(
        kappa_mhz, chi_mhz, ramsey_detuning_mhz, t2_echo_ns
    )

    n0_curves = {}
    decay_fits = {}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    with np.load(npz_path) as data:
        t_relax_values = data["t_relax"]
        t_ramsey = data["t_ramsey"]
        rectangular = bool(data["rectangular"]) if "rectangular" in data else True

        for state, style in _STATE_STYLES.items():
            n0_values = np.full(t_relax_values.shape, np.nan)

            for i, t_relax in enumerate(t_relax_values):
                key = f"{state}_t_relax_{t_relax:g}"
                if key not in data:
                    continue

                amplitude = normalized_amplitude(data[key])
                fit_result = fit_ramsey_eq1(
                    t_ramsey, amplitude, kappa, chi, detuning, gamma2
                )
                n0_values[i] = fit_result["n0"]

            n0_curves[state] = n0_values
            ax.plot(t_relax_values, n0_values, linestyle="", **style)

            decay_fit = fit_decay_curve(t_relax_values, n0_values)
            if decay_fit is not None:
                popt, pcov = decay_fit
                decay_fits[state] = {"popt": popt, "pcov": pcov}
                t_plot = np.linspace(t_relax_values.min(), t_relax_values.max(), 400)
                ax.plot(
                    t_plot,
                    exponential_decay(t_plot, *popt),
                    color=style["color"],
                    linestyle="-",
                    label=f"{style['label']} exp. fit",
                )

    pulse_label = "square" if rectangular else "CLEAR"
    ax.set_xlabel(r"Wait time after M1 $t_{\mathrm{relax}}$ (ns)")
    ax.set_ylabel(r"Residual population $n_0$")
    ax.set_title(title or f"Fig. 2(b): residual population after {pulse_label} M1")
    ax.legend(loc="best")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    for state, n0_values in n0_curves.items():
        print(f"{state:8s} n0 = {np.round(n0_values, 3)}")
        if state in decay_fits:
            tau = decay_fits[state]["popt"][2]
            print(f"{state:8s} tau = {tau:.3f} ns")

    return {"n0": n0_curves, "decay_fits": decay_fits}


def demo_synthetic_fig2b(output_path: Path | None = None) -> dict:
    """Run the Fig. 2(b) analysis on synthetic data."""

    output = output_path.parent if output_path is not None else Path(".")
    output.mkdir(parents=True, exist_ok=True)

    kappa, chi, detuning, gamma2 = fixed_ramsey_parameters(
        kappa_mhz=1.1,
        chi_mhz=0.5,
        ramsey_detuning_mhz=10.0,
        t2_echo_ns=10_000.0,
    )
    t_relax = np.arange(0.0, 900.0 + 1.0, 100.0)
    t_ramsey = np.arange(0.0, 600.0 + 1.0, 8.0)

    rng = np.random.default_rng(0)
    all_data = {}
    for state, n0_0 in {"ground": 1.6, "excited": 1.2}.items():
        n0_curve = 0.04 + n0_0 * np.exp(-t_relax / 250.0)
        for wait, n0 in zip(t_relax, n0_curve):
            amplitude = ramsey_signal(
                t_ramsey,
                n0=n0,
                phi0=0.2,
                kappa=kappa,
                chi=chi,
                detuning=detuning,
                gamma2=gamma2,
            ) + 0.03 * rng.standard_normal(t_ramsey.size)
            key = f"{state}_t_relax_{wait:g}"
            all_data[key] = amplitude[None, :]

    npz_path = output / "fig2b_demo.npz"
    np.savez(
        npz_path,
        t_relax=t_relax,
        t_ramsey=t_ramsey,
        rectangular=True,
        **all_data,
    )
    return analyze_fig2b(
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
    RECTANGULAR = True

    output = Path("output/clear_test")
    pulse_label = "square" if RECTANGULAR else "clear"
    npz_path = output / f"fig2b_{pulse_label}.npz"

    if npz_path.exists():
        analyze_fig2b(
            npz_path=npz_path,
            kappa_mhz=1.1,
            chi_mhz=0.5,
            ramsey_detuning_mhz=10.0,
            t2_echo_ns=10_000.0,
            output_path=output / f"fig2b_{pulse_label}.png",
        )
    else:
        clear_json_path = Path("parametri/clear_amplitudes.json")
        readout_amplitude = readout_amplitude_for_n_target()

        if clear_json_path.exists():
            clear_data = load_clear_ratios(clear_json_path)
            clear_parameters = clear_params_from_ratios(
                clear_data["ratios"],
                readout_amplitude=readout_amplitude,
                t_kick_ns=clear_data["t_kick_ns"],
                t_readout_ns=clear_data["t_readout_ns"],
            )
        elif RECTANGULAR:
            clear_parameters = rectangular_params(
                readout_amplitude=readout_amplitude + 0.0j,
                t_kick=150.0,
                t_steady=600.0,
            )
        else:
            raise FileNotFoundError(
                "parametri/clear_amplitudes.json is required for RECTANGULAR=False."
            )

        npz_path = run_fig2b_scan(
            platform_name=PLATFORM,
            qubit=QUBIT,
            clear=clear_parameters,
            t_relax_values=np.arange(0.0, 900.0 + 1.0, 100.0),
            t_ramsey_values=np.arange(0.0, 600.0 + 1.0, 8.0),
            output=output,
            rectangular=RECTANGULAR,
        )
        analyze_fig2b(
            npz_path=npz_path,
            kappa_mhz=1.1,
            chi_mhz=0.5,
            ramsey_detuning_mhz=10.0,
            t2_echo_ns=10_000.0,
            output_path=output / f"fig2b_{pulse_label}.png",
        )
