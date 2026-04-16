"""
fault_augmentation.py
---------------------
Physics-based synthetic fault injection for the HP WIP-A predictive
maintenance project.

The real dataset contains ONE confirmed fault (Class 1 — TI0731 thermal
runaway). To train a 9-class XGBoost classifier, this module generates
labelled synthetic windows for fault classes 2–8 by injecting physically
realistic sensor signatures derived from the Sulzer BB5 HPcp300-405-4s
data-sheet parameters into normal-phase windows.

Fault Classes
─────────────
  0  Normal                  (real data — Jan 1 → Feb 27 2026)
  1  Thrust Bearing NDE-1    (real data — Feb 28 → Apr 8 2026)
  2  Bearing Wear            (synthetic)
  3  Shaft Imbalance         (synthetic)
  4  Misalignment            (synthetic)
  5  Cavitation              (synthetic)
  6  Seal Degradation        (synthetic)
  7  Gearbox Gear Wear       (synthetic)
  8  Motor Bearing Fault     (synthetic)

Usage
─────
  from fault_augmentation import build_augmented_dataset

  # X_train : (n, 96, 35) normalised windows
  # y_train : (n,) binary labels  (0 or 1)
  X_aug, y_aug = build_augmented_dataset(X_train, y_train)
  # Returns shuffled dataset with all 9 classes
"""

import numpy as np
import logging
from typing import Tuple, List, Optional

from config import (
    VIBRATION_SENSORS, TEMPERATURE_SENSORS, ALL_SENSORS,
    PUMP_RATED_SPEED_RPM, RANDOM_SEED,
)

logger = logging.getLogger(__name__)

# ─── Sensor index lookups (computed once at import) ──────────────────────────
_IDX = {s: i for i, s in enumerate(ALL_SENSORS)}

def _idx(tag: str) -> Optional[int]:
    return _IDX.get(tag, None)

# Convenience groups
_PUMP_NDE_X  = _idx("2026VI0731X.PV")
_PUMP_NDE_Y  = _idx("2026VI0731Y.PV")
_PUMP_DE_X   = _idx("2026VI0732X.PV")
_PUMP_DE_Y   = _idx("2026VI0732Y.PV")
_GB_LSS_NDE_X= _idx("2026VI0733X.PV")
_GB_LSS_NDE_Y= _idx("2026VI0733Y.PV")
_GB_HSS_DE_X = _idx("2026VI0734X.PV")
_GB_HSS_DE_Y = _idx("2026VI0734Y.PV")
_GB_LSS_DE_X = _idx("2026VI0735X.PV")
_GB_LSS_DE_Y = _idx("2026VI0735Y.PV")
_GB_HSS_NDE_X= _idx("2026VI0736X.PV")
_GB_HSS_NDE_Y= _idx("2026VI0736Y.PV")
_MOTOR_DE_X  = _idx("2026VI0737X.PV")
_MOTOR_DE_Y  = _idx("2026VI0737Y.PV")
_MOTOR_NDE_X = _idx("2026VI0738X.PV")
_MOTOR_NDE_Y = _idx("2026VI0738Y.PV")

_TI0724 = _idx("2026TI0724.PV")   # Motor DE Bearing Temp
_TI0725 = _idx("2026TI0725.PV")   # Motor NDE Bearing Temp
_TI0730 = _idx("2026TI0730.PV")   # Casing Temp
_TI0731 = _idx("2026TI0731.PV")   # Thrust Bearing NDE-1 ← confirmed fault sensor
_TI0735 = _idx("2026TI0735.PV")   # Pump NDE Bearing Temp
_TI0736 = _idx("2026TI0736.PV")   # Pump DE Bearing Temp
_TI0737 = _idx("2026TI0737.PV")   # GB Thrust Bearing Temp 1
_TI0738 = _idx("2026TI0738.PV")   # GB Thrust Bearing Temp 2

WINDOW_LEN = 96   # samples per window (= 24 h at 15 min)
N_SENSORS  = len(ALL_SENSORS)

# Normalised sampling frequency (15-min samples → Fs = 1/900 Hz)
FS = 1.0 / (15 * 60)
SHAFT_HZ = PUMP_RATED_SPEED_RPM / 60.0     # 24.68 Hz

# Map fault class → human-readable name
FAULT_CLASS_NAMES = {
    0: "Normal",
    1: "Thrust Bearing NDE-1 Thermal Runaway",
    2: "Bearing Wear",
    3: "Shaft Imbalance",
    4: "Misalignment",
    5: "Cavitation",
    6: "Seal Degradation",
    7: "Gearbox Gear Wear",
    8: "Motor Bearing Fault",
}

# Severity levels and multipliers
SEVERITY_LEVELS = {
    "mild":     0.08,
    "moderate": 0.20,
    "severe":   0.40,
}


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _make_time_axis(n: int) -> np.ndarray:
    """Return time vector in seconds for a window of n samples."""
    return np.arange(n) * (15 * 60)   # 15 minutes × 60 s

def _sinusoid(n: int, freq_hz: float, amplitude: float, phase: float = 0.0) -> np.ndarray:
    """Generate a sinusoidal signal at freq_hz for a window of n samples."""
    t = _make_time_axis(n)
    return amplitude * np.sin(2 * np.pi * freq_hz * t + phase)

def _impulse_train(n: int, n_impulses: int, amplitude: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Random impulse train (models bearing defect impacts)."""
    sig = np.zeros(n)
    positions = rng.integers(0, n, size=n_impulses)
    widths    = rng.integers(1, 3, size=n_impulses)
    for pos, w in zip(positions, widths):
        end = min(pos + w, n)
        sig[pos:end] += amplitude
    return sig

def _exponential_ramp(n: int, start: float, end: float) -> np.ndarray:
    """Smooth exponential ramp from start to end over n samples."""
    x = np.linspace(0, 1, n)
    return start + (end - start) * (np.exp(x) - 1) / (np.e - 1)

def _linear_ramp(n: int, start: float, end: float) -> np.ndarray:
    return np.linspace(start, end, n)

def _safe_add(window: np.ndarray, col: Optional[int],
              signal: np.ndarray, clip_min: float = 0.0,
              clip_max: float = 1.0) -> None:
    """Add signal to window[:, col] in-place with clipping. No-op if col is None."""
    if col is None:
        return
    window[:, col] = np.clip(window[:, col] + signal, clip_min, clip_max)


# ═════════════════════════════════════════════════════════════════════════════
# Fault injection functions — one per fault class
# ═════════════════════════════════════════════════════════════════════════════

def _inject_bearing_wear(window: np.ndarray, severity: float,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Class 2 — Bearing Wear (Pump NDE / DE bearings)

    Physics:
      • Degrading rolling element surfaces → random impulse bursts
      • Overall RMS rises as wear progresses
      • BPFO-like harmonic (~3.5× shaft) appears in vibration
      • Pump bearing temperatures rise gradually
    """
    w = window.copy()
    n = w.shape[0]

    # Vibration: impulse train + RMS increase
    impulse_amp = severity * 0.35
    for col in [_PUMP_NDE_X, _PUMP_NDE_Y, _PUMP_DE_X, _PUMP_DE_Y]:
        if col is None:
            continue
        impulses = _impulse_train(n, n_impulses=6, amplitude=impulse_amp, rng=rng)
        rms_gain  = _linear_ramp(n, 0, severity * 0.20)
        bpfo_harm = _sinusoid(n, 3.5 * SHAFT_HZ, severity * 0.10)
        noise     = rng.normal(0, severity * 0.02, n)
        _safe_add(w, col, impulses + rms_gain + bpfo_harm + noise)

    # Temperature: gradual rise in pump bearings
    temp_rise_nde = _linear_ramp(n, 0, severity * 0.15)
    temp_rise_de  = _linear_ramp(n, 0, severity * 0.12)
    _safe_add(w, _TI0735, temp_rise_nde)
    _safe_add(w, _TI0736, temp_rise_de)

    return w


def _inject_shaft_imbalance(window: np.ndarray, severity: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Class 3 — Shaft Imbalance

    Physics:
      • Mass imbalance → centrifugal force → strong 1× RPM harmonic
      • All vibration sensors affected (force transmitted through shaft)
      • Weak 2× harmonic also present
    """
    w = window.copy()
    n = w.shape[0]

    phase_1x = rng.uniform(0, 2 * np.pi)
    sig_1x   = _sinusoid(n, SHAFT_HZ,     severity * 0.45, phase_1x)
    sig_2x   = _sinusoid(n, 2 * SHAFT_HZ, severity * 0.15, phase_1x * 1.3)

    all_vib_cols = [_PUMP_NDE_X, _PUMP_NDE_Y, _PUMP_DE_X, _PUMP_DE_Y,
                    _MOTOR_DE_X, _MOTOR_DE_Y, _MOTOR_NDE_X, _MOTOR_NDE_Y,
                    _GB_LSS_NDE_X, _GB_LSS_DE_X]
    for col in all_vib_cols:
        noise = rng.normal(0, severity * 0.02, n)
        _safe_add(w, col, sig_1x + sig_2x + noise)

    return w


def _inject_misalignment(window: np.ndarray, severity: float,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Class 4 — Shaft/Coupling Misalignment

    Physics:
      • Angular or parallel misalignment → 2× RPM dominant
      • 1× component still present but weaker
      • 3× sideband appears in advanced cases
      • Gearbox coupling sees highest loading
    """
    w = window.copy()
    n = w.shape[0]

    phase = rng.uniform(0, 2 * np.pi)
    sig_1x = _sinusoid(n, SHAFT_HZ,     severity * 0.20, phase)
    sig_2x = _sinusoid(n, 2 * SHAFT_HZ, severity * 0.45, phase * 1.2)
    sig_3x = _sinusoid(n, 3 * SHAFT_HZ, severity * 0.10, phase * 1.5)

    # Mainly pump + gearbox coupling side
    affected_cols = [_PUMP_NDE_X, _PUMP_NDE_Y, _PUMP_DE_X, _PUMP_DE_Y,
                     _GB_LSS_NDE_X, _GB_LSS_NDE_Y, _GB_LSS_DE_X, _GB_LSS_DE_Y]
    for col in affected_cols:
        noise = rng.normal(0, severity * 0.02, n)
        _safe_add(w, col, sig_1x + sig_2x + sig_3x + noise)

    return w


def _inject_cavitation(window: np.ndarray, severity: float,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Class 5 — Cavitation

    Physics:
      • Bubble collapse → broadband random high-frequency noise
      • Vane-pass frequency (N_vanes × shaft) excitation
      • Sub-synchronous component at ~0.45× shaft (flow instability)
      • Casing temperature step up due to fluid recirculation energy
    """
    w    = window.copy()
    n    = w.shape[0]
    N_VANES = 7     # Closed impeller typical vane count

    broadband = rng.normal(0, severity * 0.35, n)
    vane_pass = _sinusoid(n, N_VANES * SHAFT_HZ, severity * 0.20)
    sub_sync  = _sinusoid(n, 0.45 * SHAFT_HZ,   severity * 0.15)

    # Pump NDE/DE vibration (closest to impeller)
    for col in [_PUMP_NDE_X, _PUMP_NDE_Y, _PUMP_DE_X, _PUMP_DE_Y]:
        _safe_add(w, col, broadband + vane_pass + sub_sync)

    # Casing temperature step
    casing_step = _linear_ramp(n, 0, severity * 0.20)
    _safe_add(w, _TI0730, casing_step)

    return w


def _inject_seal_degradation(window: np.ndarray, severity: float,
                               rng: np.random.Generator) -> np.ndarray:
    """
    Class 6 — Mechanical Seal Degradation (Plan 53B)

    Physics:
      • Worn seal faces → frictional heat generation → exponential temp rise
      • Process fluid leakage past seal → casing heating
      • Sub-synchronous vibration from pressure fluctuations
      • Pump bearing temperatures rise from thermal conduction
    """
    w = window.copy()
    n = w.shape[0]

    # Exponential casing temp rise
    casing_heat = _exponential_ramp(n, 0, severity * 0.30)
    _safe_add(w, _TI0730, casing_heat)

    # Linear pump bearing temp rise
    pump_heat = _linear_ramp(n, 0, severity * 0.18)
    _safe_add(w, _TI0735, pump_heat)
    _safe_add(w, _TI0736, pump_heat * 0.8)

    # Sub-synchronous vibration
    sub_sync = _sinusoid(n, 0.35 * SHAFT_HZ, severity * 0.12)
    noise    = rng.normal(0, severity * 0.02, n)
    for col in [_PUMP_NDE_X, _PUMP_NDE_Y]:
        _safe_add(w, col, sub_sync + noise)

    return w


def _inject_gearbox_gear_wear(window: np.ndarray, severity: float,
                               rng: np.random.Generator) -> np.ndarray:
    """
    Class 7 — Gearbox Gear Tooth Wear

    Physics:
      • Gear mesh frequency (GMF) = n_teeth × shaft speed
      • Amplitude modulation creates sidebands at ±1× shaft around GMF
      • Gearbox vibration sensors most affected
      • Gearbox bearing temperatures rise from friction
    """
    w = window.copy()
    n = w.shape[0]

    N_TEETH = 28    # Typical gearbox pinion tooth count
    GMF_HZ  = N_TEETH * SHAFT_HZ

    phase = rng.uniform(0, 2 * np.pi)
    gmf        = _sinusoid(n, GMF_HZ,              severity * 0.40, phase)
    sideband_u = _sinusoid(n, GMF_HZ + SHAFT_HZ,  severity * 0.15, phase)
    sideband_l = _sinusoid(n, GMF_HZ - SHAFT_HZ,  severity * 0.15, phase)

    gb_vib_cols = [_GB_LSS_NDE_X, _GB_LSS_NDE_Y, _GB_HSS_DE_X, _GB_HSS_DE_Y,
                   _GB_LSS_DE_X,  _GB_LSS_DE_Y,  _GB_HSS_NDE_X, _GB_HSS_NDE_Y]
    for col in gb_vib_cols:
        noise = rng.normal(0, severity * 0.02, n)
        _safe_add(w, col, gmf + sideband_u + sideband_l + noise)

    # Gearbox bearing temperatures
    gb_heat = _linear_ramp(n, 0, severity * 0.18)
    _safe_add(w, _TI0737, gb_heat)
    _safe_add(w, _TI0738, gb_heat * 0.85)

    return w


def _inject_motor_bearing_fault(window: np.ndarray, severity: float,
                                  rng: np.random.Generator) -> np.ndarray:
    """
    Class 8 — Motor Bearing Fault (DE or NDE)

    Physics:
      • Motor bearing BPFI (~5.1× shaft) and BPFO (~3.4× shaft)
      • Periodic impulse bursts from rolling element impacts
      • Motor bearing temperatures rise exponentially
    """
    w = window.copy()
    n = w.shape[0]

    BPFI_HZ = 5.1 * SHAFT_HZ
    BPFO_HZ = 3.4 * SHAFT_HZ

    bpfi = _sinusoid(n, BPFI_HZ, severity * 0.25)
    bpfo = _sinusoid(n, BPFO_HZ, severity * 0.20)
    impulses = _impulse_train(n, n_impulses=8, amplitude=severity * 0.30, rng=rng)

    motor_vib_cols = [_MOTOR_DE_X, _MOTOR_DE_Y, _MOTOR_NDE_X, _MOTOR_NDE_Y]
    for col in motor_vib_cols:
        noise = rng.normal(0, severity * 0.02, n)
        _safe_add(w, col, bpfi + bpfo + impulses + noise)

    # Motor bearing temps: exponential rise
    temp_rise = _exponential_ramp(n, 0, severity * 0.28)
    _safe_add(w, _TI0724, temp_rise)
    _safe_add(w, _TI0725, temp_rise * 0.85)

    return w


# Dispatch table: fault_class → injection function
_FAULT_INJECTORS = {
    2: _inject_bearing_wear,
    3: _inject_shaft_imbalance,
    4: _inject_misalignment,
    5: _inject_cavitation,
    6: _inject_seal_degradation,
    7: _inject_gearbox_gear_wear,
    8: _inject_motor_bearing_fault,
}


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def augment_with_faults(
    X_normal: np.ndarray,
    n_augmented_per_fault: Optional[int] = None,
    severities: Optional[List[float]] = None,
    random_seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic fault windows for Classes 2–8.

    Parameters
    ----------
    X_normal : np.ndarray
        Normal-phase windows, shape (n_normal, window_size, n_sensors).
        These are the base windows that are cloned and perturbed.
    n_augmented_per_fault : int or None
        Number of synthetic windows to generate PER FAULT CLASS.
        If None, defaults to max(len(X_normal) // 3, 20).
    severities : list of float or None
        Severity multipliers to use (cycling through the list).
        Defaults to [0.08, 0.20, 0.40] (mild / moderate / severe).
    random_seed : int
        Reproducibility seed.

    Returns
    -------
    X_synth : np.ndarray — shape (n_faults * 7, window_size, n_sensors)
    y_synth : np.ndarray — shape (n_faults * 7,) with labels 2–8
    """
    rng = np.random.default_rng(random_seed)

    if severities is None:
        severities = list(SEVERITY_LEVELS.values())   # [0.08, 0.20, 0.40]

    n_normal = len(X_normal)
    if n_augmented_per_fault is None:
        n_augmented_per_fault = max(n_normal // 3, 20)

    logger.info(
        f"Augmenting — {n_augmented_per_fault} windows × 7 fault classes "
        f"× {len(severities)} severity levels → "
        f"{n_augmented_per_fault * 7} total synthetic windows"
    )

    X_list, y_list = [], []

    for fault_class, inject_fn in _FAULT_INJECTORS.items():
        for i in range(n_augmented_per_fault):
            # Pick a random normal window as the base
            base_idx = rng.integers(0, n_normal)
            base_win = X_normal[base_idx].copy()

            # Cycle through severity levels
            severity = severities[i % len(severities)]

            synthetic = inject_fn(base_win, severity, rng)
            X_list.append(synthetic)
            y_list.append(fault_class)

    X_synth = np.array(X_list, dtype=np.float32)
    y_synth = np.array(y_list, dtype=np.int32)

    logger.info(f"Synthetic windows generated: {X_synth.shape}")
    return X_synth, y_synth


def build_augmented_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_augmented_per_fault: Optional[int] = None,
    severities: Optional[List[float]] = None,
    random_seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the complete multi-class training dataset.

    Combines:
      • Real normal windows        (Class 0)
      • Real TI0731 fault windows  (Class 1)
      • Synthetic fault windows    (Classes 2–8)

    Parameters
    ----------
    X_train : (n, window_size, n_sensors)
    y_train : (n,) with values 0 or 1 (binary labels from preprocessor)

    Returns
    -------
    X_combined : (N_total, window_size, n_sensors)
    y_combined : (N_total,) with values 0–8
    """
    normal_mask = y_train == 0
    X_normal    = X_train[normal_mask]

    if len(X_normal) == 0:
        raise ValueError("No normal windows found in X_train. "
                         "Check that y_train contains 0-labelled windows.")

    # Generate synthetic faults from normal windows
    X_synth, y_synth = augment_with_faults(
        X_normal,
        n_augmented_per_fault=n_augmented_per_fault,
        severities=severities,
        random_seed=random_seed,
    )

    # Stack real + synthetic
    X_combined = np.concatenate([X_train, X_synth], axis=0)
    y_combined = np.concatenate([y_train, y_synth], axis=0)

    # Shuffle
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(X_combined))
    X_combined = X_combined[perm]
    y_combined = y_combined[perm]

    # Log class distribution
    unique, counts = np.unique(y_combined, return_counts=True)
    dist_str = " | ".join(
        f"Class {c} ({FAULT_CLASS_NAMES.get(c, '?')}): {n}"
        for c, n in zip(unique, counts)
    )
    logger.info(f"Augmented dataset class distribution:\n  {dist_str}")
    logger.info(f"Total: {len(y_combined):,} windows  |  Shape: {X_combined.shape}")

    return X_combined.astype(np.float32), y_combined.astype(np.int32)


def get_fault_class_names() -> dict:
    """Return mapping of class index → fault name."""
    return FAULT_CLASS_NAMES.copy()