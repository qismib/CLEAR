"""Rifà i grafici n0 vs t_relax (Fig. 2(b) del paper) a partire dai dati GIA'
salvati in `setup.data_dir` (es. .../data/2Q_chip/fine_scans_clear/run_*.npz),
senza rilanciare l'esperimento e senza importare qibolab.

`CLEAR_qibolab2.py` importa `create_platform`, `Sweeper`, ecc. da qibolab a
livello di modulo, quindi anche solo importarlo richiede l'hardware layer
installato. Le funzioni di analisi vere e proprie (`analyze_run`,
`plot_n0_decay`, il fit di Eq. (1) e quello esponenziale di n0) vivono invece
in `clear_analysis.py`, che dipende solo da numpy/scipy/matplotlib: questo
script usa solo quelle, quindi puo' girare offline sui dati gia' acquisiti.

Uso:
    python CLEAR_plot_relax_fit.py

Per ogni pulse-shape ("square", "clear") cerca `run_<shape>.npz` in
`setup.data_dir` (stesso nome usato da `run_experiment` in
CLEAR_qibolab2.py); se il file non c'e' lo salta con un avviso invece di
fallire. Rigenera:
    - ramsey_fit_t_relax_<t>.png   (un pannello ground+excited per ogni t_relax)
    - n0_vs_t_relax_<shape>.png    (n0 vs t_relax, con il fit esponenziale)
e stampa T_cav (con errore) estratto dal fit per ciascuno stato, invece del
solo confronto teorico con kappa.
"""

from __future__ import annotations

from pathlib import Path

from clear_config import Setup, get_setup
from clear_analysis import analyze_run, plot_n0_decay


def refit_and_plot(setup: Setup, shape: str, use_measured_p: bool = True) -> dict:
    """Rifà il fit e il grafico per una singola pulse-shape ('square' o
    'clear'), leggendo run_<shape>.npz da setup.data_dir. Non fa nulla (e
    ritorna {}) se il file non esiste."""
    npz_path = setup.data_dir / f"run_{shape}.npz"
    if not npz_path.exists():
        print(f"[skip] {npz_path} non trovato: nessun dato da rifittare per '{shape}'.")
        return {}

    print(f"\n=== {shape} M1  (dati: {npz_path}) ===")
    results = analyze_run(npz_path, setup, use_measured_p=use_measured_p)

    output_path = setup.data_dir / f"n0_vs_t_relax_{shape}.png"
    fits = plot_n0_decay(results, setup, output_path)

    for state, fit in fits.items():
        delta_mhz = fit["kappa_mhz"] - setup.kappa_mhz
        print(
            f"  T_cav[{state:>8s}] = {fit['t_cav']:.1f} +/- {fit['t_cav_err']:.1f} ns"
            f"   (n0_0 = {fit['n0_0']:.3f}, offset = {fit['offset']:.3f})"
        )
        print(
            f"    kappa/2pi[{state:>8s}] = {fit['kappa_mhz']:.4f} +/- {fit['kappa_mhz_err']:.4f} MHz"
            f"   (nominale risonatore.toml: {setup.kappa_mhz:.4f} MHz, "
            f"differenza {delta_mhz:+.4f} MHz)"
        )
    print(f"  -> salvato {output_path}")
    return fits


def main() -> None:
    setup = get_setup(platform="2q_chip_thesis", qubit=1)
    print(f"data_dir = {setup.data_dir}")
    print(f"kappa/2pi = {setup.kappa_mhz:.4f} MHz  ->  1/kappa = {1/setup.kappa_rad_ns:.1f} ns (teorico)")

    for shape in ("square", "clear"):
        refit_and_plot(setup, shape)


if __name__ == "__main__":
    main()