"""Single source of truth for every parameter of the CLEAR experiment.

Everything that used to be re-typed in the `__main__` of CLEAR_qibolab2.py and
CLEAR_fig3.py lives here, and is derived from three calibration files:

    parametri/risonatore.toml       resonator + qubit parameters (edit this)
    parametri/clear_amplitudes.json CLEAR segment ratios, from amplitudes_CLEAR.ipynb
    data/stark_fit_results.json     n_photons = k*amp^2 + q, from stark_analysis.py

Nothing else should hardcode kappa, chi, the probe frequency, t_kick, t_steady,
or the Stark coefficient k. Import `SETUP` instead.
"""
 
from __future__ import annotations
 
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
 
import numpy as np
 
PROJECT_DIR = Path(__file__).resolve().parent
RESONATOR_TOML = PROJECT_DIR / "parametri" / "risonatore.toml"
CLEAR_AMPLITUDES_JSON = PROJECT_DIR / "parametri" / "clear_amplitudes.json"
STARK_FIT_JSON = PROJECT_DIR / "data" / "stark_fit_results.json"
DATA_DIR = PROJECT_DIR / "data" / "2Q_chip" / "IQ_trajectories"
 
CLEAR_SEGMENTS = ("ringup_1", "ringup_2", "steady", "ringdown_1", "ringdown_2")
 
 
def mhz_to_rad_per_ns(f_mhz: float) -> float:
    """Frequency in MHz -> angular frequency in rad/ns."""
    return 2.0 * np.pi * f_mhz * 1e-3
 
 
# ---------------------------------------------------------------------------
# Calibration-file loaders
# ---------------------------------------------------------------------------
 
 
def _parse_flat_toml(path: Path) -> dict:
    """Fallback parser for flat `key = value` TOML, used only when neither
    tomllib (Python >= 3.11) nor tomli is importable."""
    out: dict = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not value:
            raise ValueError(f"Empty value for {key!r} in {path}")
        if value[:1] in "\"'" and value[-1:] == value[:1]:
            out[key] = value[1:-1]
        elif value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        else:
            out[key] = int(value) if value.lstrip("-+").isdigit() else float(value)
    return out
 
 
def load_toml(path: Path = RESONATOR_TOML) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return _parse_flat_toml(path)
    with open(path, "rb") as f:
        return tomllib.load(f)
 
 
def load_clear_ratios(path: Path = CLEAR_AMPLITUDES_JSON) -> dict:
    """CLEAR segment amplitudes, each as a complex ratio to the steady-state
    readout amplitude, plus the segment durations they were computed for.
 
    The ratios are linear in the drive amplitude, so they carry over to any
    readout power -- but they *do* depend on t_kick, kappa and chi. The
    durations are returned alongside so that nothing can silently use ratios
    computed for one t_kick with a pulse built for another.
    """
    data = json.loads(path.read_text())
    return {
        "ratios": {k: complex(*v) for k, v in data["ratios"].items()},
        "t_kick_ns": float(data["t_kick_us"]) * 1e3,
        "t_steady_ns": float(data["t_readout_us"]) * 1e3,
    }
 
 
def load_stark_k(path: Path = STARK_FIT_JSON) -> float:
    """Slope k of n_photons = k*amp^2 + q, saved by stark_analysis.py."""
    return float(json.loads(path.read_text())["k"])
 
 
# ---------------------------------------------------------------------------
# CLEAR pulse parameters
# ---------------------------------------------------------------------------
 
 
@dataclass(frozen=True)
class ClearPulseParameters:
    """Complex amplitudes (|a| <= 1, hardware units) and durations of the five
    CLEAR segments."""
 
    ringup_1: complex
    ringup_2: complex
    steady: complex
    ringdown_1: complex
    ringdown_2: complex
    t_kick: float
    t_steady: float
 
    @property
    def amplitudes(self) -> tuple[complex, ...]:
        return tuple(getattr(self, name) for name in CLEAR_SEGMENTS)
 
    @property
    def durations(self) -> tuple[float, ...]:
        return (self.t_kick, self.t_kick, self.t_steady, self.t_kick, self.t_kick)
 
    @property
    def ringdown_duration(self) -> float:
        """Length of the two ring-down segments; also the delay appended to the
        square pulse so that both M1 shapes have the same total length."""
        return 2.0 * self.t_kick
 
    def duration(self, rectangular: bool = False, match_clear_length: bool = True) -> float:
        """Total M1 length, i.e. how long the qubit is exposed to the drive."""
        if not rectangular:
            return sum(self.durations)
        return 2 * self.t_kick + self.t_steady if match_clear_length else self.t_steady
 
 
# ---------------------------------------------------------------------------
# The experiment setup
# ---------------------------------------------------------------------------
 
 
@dataclass(frozen=True)
class Setup:
    """Every number the CLEAR scripts need, derived once from the calibration
    files. Frequencies in MHz, times in ns, unless the name says otherwise."""
 
    platform: str
    qubit: int
 
    kappa_mhz: float
    f_res_g_mhz: float
    f_res_e_mhz: float
    f_drive_toml_mhz: float
 
    t1_ns: float
    t2_echo_ns: float
 
    n_target: float
    k_stark: float
 
    # Durations the CLEAR ratios were actually solved for (from the JSON).
    t_kick_ns: float
    t_steady_ns: float
    clear_ratios: dict
    # The same durations as requested in the TOML, kept only so that a stale
    # JSON can be detected.
    t_kick_toml_ns: float
    t_steady_toml_ns: float
 
    ramsey_detuning_hz: float
    nshots: int
    relaxation_time_ns: float
    t_buffer_ns: float
    data_dir: Path
 
    # -- derived resonator quantities ---------------------------------------
 
    @property
    def chi_mhz(self) -> float:
        """chi/2pi = (f_res_g - f_res_e)/2, positive by construction."""
        return abs(self.f_res_g_mhz - self.f_res_e_mhz) / 2.0
 
    @property
    def f_probe_mhz(self) -> float:
        """Midpoint of the two state-dependent resonator frequencies. Driving
        here makes the steady-state photon number independent of the qubit
        state, which is what lets a single Stark calibration serve both
        preparations."""
        return (self.f_res_g_mhz + self.f_res_e_mhz) / 2.0
 
    @property
    def readout_frequency_hz(self) -> float:
        return self.f_probe_mhz * 1e6
 
    # -- angular frequencies used by the Eq. (1) model ----------------------
 
    @property
    def kappa_rad_ns(self) -> float:
        return mhz_to_rad_per_ns(self.kappa_mhz)
 
    @property
    def chi_rad_ns(self) -> float:
        return mhz_to_rad_per_ns(self.chi_mhz)
 
    @property
    def detuning_rad_ns(self) -> float:
        return mhz_to_rad_per_ns(self.ramsey_detuning_hz / 1e6)
 
    @property
    def gamma2_rad_ns(self) -> float:
        return 1.0 / self.t2_echo_ns
 
    # -- drive amplitudes ---------------------------------------------------
 
    def amplitude_for_n(self, n: float) -> float:
        """Hardware amplitude giving n steady-state photons, from the Stark
        calibration n = k*amp^2."""
        if n < 0:
            raise ValueError("photon number must be >= 0")
        return float(np.sqrt(n / self.k_stark))
 
    @property
    def amplitude_1ph(self) -> float:
        """Amplitude for n = 1, i.e. P_norm = 1 in Fig. 3(c)."""
        return self.amplitude_for_n(1.0)
 
    def clear_pulse(self, n: float | None = None) -> ClearPulseParameters:
        """CLEAR pulse parameters at n steady-state photons (default n_target)."""
        amp = self.amplitude_for_n(self.n_target if n is None else n)
        return ClearPulseParameters(
            **{name: self.clear_ratios[name] * amp for name in CLEAR_SEGMENTS},
            t_kick=self.t_kick_ns,
            t_steady=self.t_steady_ns,
        )
 
    # -- T1 exposure --------------------------------------------------------
 
    def exposure_ns(
        self, t_relax: float, rectangular: bool = False, match_clear_length: bool = True
    ) -> float:
        """Time the qubit spends in |e> between the RX preparation and the first
        X90, i.e. M1 + t_relax. Decay during the Ramsey delay itself is not
        included: it is already described by Gamma2 in Eq. (1)."""
        pulse = self.clear_pulse()
        return pulse.duration(rectangular, match_clear_length) + float(t_relax)
 
    def excited_fraction(
        self, t_relax: float, rectangular: bool = False, match_clear_length: bool = True
    ) -> float:
        """Fraction of the ensemble still in |e> when the Ramsey starts, from a
        bare exp(-t/T1).
 
        This is a *fallback*. Measurement-induced transitions make the decay
        under the readout drive faster than the bare T1, so prefer the measured
        value from `run_excited_fraction` in CLEAR_qibolab2.py.
        """
        return float(np.exp(-self.exposure_ns(t_relax, rectangular, match_clear_length) / self.t1_ns))
 
    # -- consistency checks -------------------------------------------------
 
    def check(self) -> None:
        """Warns about the inconsistencies that silently bias n0."""
        offset_khz = (self.f_drive_toml_mhz - self.f_probe_mhz) * 1e3
        if abs(offset_khz) > 0.02 * self.kappa_mhz * 1e3:
            half_k2 = (self.kappa_mhz / 2) ** 2
            d = self.f_drive_toml_mhz - self.f_probe_mhz
            ratio = (half_k2 + (self.chi_mhz - d) ** 2) / (half_k2 + (self.chi_mhz + d) ** 2)
            warnings.warn(
                f"f_drive_MHz in the TOML is {offset_khz:+.1f} kHz off the midpoint "
                f"{self.f_probe_mhz:.6f} MHz. Driving there would make the "
                f"steady-state photon number differ by a factor {ratio:.2f} between "
                "the ground and excited preparations. The scripts use the midpoint; "
                "fix f_drive_MHz or the two f_res values.",
                stacklevel=2,
            )
        stale = [
            f"{name}: TOML asks {toml_val:.0f} ns, ratios were solved for {json_val:.0f} ns"
            for name, toml_val, json_val in (
                ("t_kick", self.t_kick_toml_ns, self.t_kick_ns),
                ("t_steady", self.t_steady_toml_ns, self.t_steady_ns),
            )
            if not np.isclose(toml_val, json_val)
        ]
        if stale:
            warnings.warn(
                "clear_amplitudes.json is out of date with risonatore.toml (\n  "
                + "\n  ".join(stale)
                + "\n). The scripts use the JSON durations, because the ring-up/ring-down "
                "ratios are only a solution of the linear-resonator model for the t_kick "
                "they were computed with. Re-run amplitudes_CLEAR.ipynb to adopt the new "
                "durations.",
                stacklevel=2,
            )
        exposure = self.exposure_ns(0.0)
        if exposure > 0.3 * self.t1_ns:
            warnings.warn(
                f"M1 lasts {exposure:.0f} ns = {exposure / self.t1_ns:.2f} T1: about "
                f"{100 * (1 - np.exp(-exposure / self.t1_ns)):.0f}% of the excited "
                "preparation decays before the Ramsey starts. The T1 correction in "
                "clear_analysis is doing real work here; consider shortening t_steady.",
                stacklevel=2,
            )
 
 
def load_setup(
    platform: str = "2q_chip_thesis",
    qubit: int = 1,
    toml_path: Path = RESONATOR_TOML,
    clear_json: Path = CLEAR_AMPLITUDES_JSON,
    stark_json: Path = STARK_FIT_JSON,
    data_dir: Path = DATA_DIR,
    **overrides,
) -> Setup:
    """Builds the Setup from the calibration files. Keyword overrides are
    accepted for one-off scans, but the files remain the default."""
 
    toml = load_toml(toml_path)
    clear = load_clear_ratios(clear_json)
 
    try:
        k_stark = load_stark_k(stark_json)
    except FileNotFoundError:
        k_stark = float(toml["k_stark_fallback"])
        warnings.warn(
            f"{stark_json} not found; using k_stark_fallback = {k_stark:g} from the "
            "TOML. Re-run stark_analysis.py at the midpoint probe frequency.",
            stacklevel=2,
        )
 
    params = dict(
        platform=platform,
        qubit=qubit,
        kappa_mhz=float(toml["kappa"]),
        f_res_g_mhz=float(toml["f_res_g_MHz"]),
        f_res_e_mhz=float(toml["f_res_e_MHz"]),
        f_drive_toml_mhz=float(toml["f_drive_MHz"]),
        t1_ns=float(toml["t1_us"]) * 1e3,
        t2_echo_ns=float(toml["t2_echo_us"]) * 1e3,
        n_target=float(toml["n_target"]),
        k_stark=k_stark,
        t_kick_ns=clear["t_kick_ns"],
        t_steady_ns=clear["t_steady_ns"],
        clear_ratios=clear["ratios"],
        t_kick_toml_ns=float(toml["t_kick"]) * 1e3,
        t_steady_toml_ns=float(toml["t_readout"]) * 1e3,
        ramsey_detuning_hz=float(toml["ramsey_detuning_MHz"]) * 1e6,
        nshots=int(toml["nshots"]),
        relaxation_time_ns=float(toml["relaxation_time_us"]) * 1e3,
        t_buffer_ns=float(toml["t_buffer_ns"]),
        data_dir=Path(data_dir),
    )
    params.update(overrides)
 
    setup = Setup(**params)
    setup.check()
    return setup
 
 
SETUP = None  # populated lazily by get_setup()
 
 
def get_setup(**kwargs) -> Setup:
    """Cached default setup, so importing modules do not each re-read and
    re-warn about the calibration files."""
    global SETUP
    if SETUP is None or kwargs:
        SETUP = load_setup(**kwargs)
    return SETUP
