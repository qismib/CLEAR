"""Self-test for the analysis: generates synthetic traces with known n0 and a
known excited-state loss, then checks that the fit recovers them.

Run with `python test_clear_analysis.py`. Requires no hardware.

The point of the second test is the one that matters physically: fitting an
excited trace with the uncorrected single-n0 model returns a biased n0, and
the mixture model does not. If a change to clear_analysis silently breaks the
correction, this is where it shows up.
"""

from __future__ import annotations

import numpy as np

from clear_config import load_setup
from clear_analysis import RamseyModel, StateReference, fit_pair, fit_ramsey


def synthetic_trace(model, n0, phi0, rng, noise=0.02, p=1.0, n0_decayed=0.0,
                    scale=0.85, offset=0.06, t=None):
    t = np.arange(0.0, 600.0, 6.0) if t is None else t
    y = offset + scale * model.mixture(t, n0, phi0, p, n0_decayed)
    return t, y + noise * rng.standard_normal(t.size)


def main() -> int:
    setup = load_setup()
    model = RamseyModel.from_setup(setup)
    rng = np.random.default_rng(0)
    failures = []

    print(f"kappa/2pi = {setup.kappa_mhz:.4f} MHz, chi/2pi = {setup.chi_mhz:.4f} MHz")
    print(f"probe midpoint = {setup.f_probe_mhz:.6f} MHz")
    print(f"amplitude for n=1: {setup.amplitude_1ph:.5f}\n")

    # --- 1. plain trace with arbitrary scale/offset -------------------------
    n0_true, phi0_true = 1.4, 0.7
    t, y = synthetic_trace(model, n0_true, phi0_true, rng, scale=0.85, offset=0.06)
    fit = fit_ramsey(t, y, model)
    ok = abs(fit["n0"] - n0_true) < 5 * max(fit["n0_err"], 1e-3) + 0.05
    print(f"[{'ok' if ok else 'FAIL'}] free scale/offset: "
          f"n0 = {fit['n0']:.3f} +/- {fit['n0_err']:.3f} (true {n0_true})  "
          f"scale = {fit['scale']:.3f} offset = {fit['offset']:.3f}")
    if not ok:
        failures.append("scale/offset fit")

    # --- 2. excited trace with T1 loss, corrected vs uncorrected -----------
    p = setup.excited_fraction(t_relax=0.0, rectangular=True)
    n0_g, n0_e = 0.9, 1.8
    t, y_g = synthetic_trace(model, n0_g, phi0_true, rng)
    t, y_e = synthetic_trace(model, n0_e, phi0_true, rng, p=p, n0_decayed=n0_g)

    naive = fit_ramsey(t, y_e, model)                      # p = 1, no correction
    fits = fit_pair(t, y_g, y_e, model, p_excited=p)       # corrected
    corrected = fits["excited"]

    print(f"\nM1 exposure {setup.exposure_ns(0.0, rectangular=True):.0f} ns, "
          f"T1 = {setup.t1_ns:.0f} ns -> surviving p_e = {p:.3f}")
    print(f"     ground fit: n0 = {fits['ground']['n0']:.3f} (true {n0_g})")
    print(f"  uncorrected  : n0 = {naive['n0']:.3f}  (true {n0_e})  "
          f"error {100 * (naive['n0'] - n0_e) / n0_e:+.0f}%")
    print(f"  T1-corrected : n0 = {corrected['n0']:.3f}  (true {n0_e})  "
          f"error {100 * (corrected['n0'] - n0_e) / n0_e:+.0f}%")

    if abs(corrected["n0"] - n0_e) >= abs(naive["n0"] - n0_e):
        failures.append("T1 correction did not improve n0")
    if abs(corrected["n0"] - n0_e) > 0.15 * n0_e:
        failures.append("T1-corrected n0 off by more than 15%")

    # --- 3. IQ projection round-trip ---------------------------------------
    ref = StateReference(iq_ground=3.0 - 1.0j, iq_excited=1.0 + 2.0j)
    pe_true = np.array([0.0, 0.25, 0.5, 1.0])
    fake_iq = (ref.iq_ground + pe_true * (ref.iq_excited - ref.iq_ground))[None, :]
    recovered = ref.project(fake_iq)
    ok = np.allclose(recovered, pe_true, atol=1e-9)
    print(f"\n[{'ok' if ok else 'FAIL'}] IQ projection round-trip")
    if not ok:
        failures.append("IQ projection")

    print("\n" + ("ALL TESTS PASSED" if not failures else "FAILURES: " + ", ".join(failures)))
    return 0 if not failures else 1



def test_midpoint() -> list[str]:
    """End-to-end check of the midpoint calibration: synthesize n0(f) from the
    Lorentzian model with a deliberately displaced true midpoint, feed it
    through the zero-crossing finder, and check it is recovered."""
    from CLEAR_midpoint import _zero_crossing, default_frequencies_mhz

    setup = load_setup()
    rng = np.random.default_rng(3)
    failures = []

    f_true = setup.f_probe_mhz + 0.134   # the 134 kHz offset found in the TOML
    freqs = default_frequencies_mhz(setup, n=13)
    half_k2 = (setup.kappa_mhz / 2) ** 2
    lor = lambda d: 1.0 / (half_k2 + d**2)

    n_scale = 2.0 / lor(setup.chi_mhz)
    d = freqs - f_true
    n0_g = n_scale * lor(d + setup.chi_mhz)
    n0_e = n_scale * lor(d - setup.chi_mhz)

    noise = 0.02
    delta = (n0_e - n0_g) + noise * rng.standard_normal(freqs.size)
    delta_err = np.full(freqs.size, noise)

    f_fit, f_err = _zero_crossing(freqs, delta, delta_err)
    residual_khz = (f_fit - f_true) * 1e3
    ok = abs(residual_khz) < 15.0
    print(f"\n[{'ok' if ok else 'FAIL'}] midpoint recovery: "
          f"{f_fit:.6f} +/- {f_err*1e3:.0f} kHz  (true {f_true:.6f}, "
          f"off by {residual_khz:+.1f} kHz)")
    if not ok:
        failures.append("midpoint zero-crossing")

    ratio = lor(setup.chi_mhz - 0.134) / lor(setup.chi_mhz + 0.134)
    print(f"      n_g/n_e at the nominal (wrong) midpoint would be {ratio:.2f}")
    return failures


if __name__ == "__main__":
    import sys
    rc = main()
    extra = test_midpoint()
    if extra:
        print("FAILURES:", ", ".join(extra))
        rc = 1
    sys.exit(rc)