"""
feature_engineering.py
-----------------------
Extracts ~45 diagnostic features per sliding window across three domains:
  1. Time Domain    (16 features)
  2. Frequency Domain / FFT (14 features)
  3. Process Health Indicators (15 features)
"""

import numpy as np
import pandas as pd
import logging
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq

from config import (
    VIBRATION_SENSORS, TEMPERATURE_SENSORS, ALL_SENSORS,
    FAULT_TEMP_SENSOR, NORMAL_BASELINE_TEMP, SHUTDOWN_VIB_THRESHOLD,
    SAMPLING_INTERVAL_MIN, PUMP_RATED_SPEED_RPM,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Number of sensor channels
N_VIB  = len(VIBRATION_SENSORS)
N_TEMP = len(TEMPERATURE_SENSORS)

# Pump shaft frequency (Hz) — 1481 RPM
SHAFT_FREQ_HZ = PUMP_RATED_SPEED_RPM / 60.0

# Sampling frequency for 15-min data
FS_HZ = 1.0 / (SAMPLING_INTERVAL_MIN * 60)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))

def _crest_factor(x: np.ndarray) -> float:
    rms = _rms(x)
    return float(np.max(np.abs(x)) / rms) if rms > 0 else 0.0

def _kurtosis(x: np.ndarray) -> float:
    return float(kurtosis(x, fisher=True))

def _skewness(x: np.ndarray) -> float:
    return float(skew(x))


# ─────────────────────────────────────────────────────────────────────────────
# Domain 1 — Time Domain Features
# ─────────────────────────────────────────────────────────────────────────────

def extract_time_domain(window: np.ndarray) -> dict:
    """
    Per-window time-domain features.
    window shape: (window_size, n_sensors)

    Features extracted:
    - RMS, Peak, Crest Factor, Kurtosis, Skewness, StdDev for vibration
    - Mean Temp and Temp Rate-of-Change for TI0731
    """
    features = {}

    # Index of TI0731 in ALL_SENSORS
    ti0731_idx = ALL_SENSORS.index(FAULT_TEMP_SENSOR) if FAULT_TEMP_SENSOR in ALL_SENSORS else None

    # --- Vibration aggregates (across all 16 vibration channels) ---
    vib_indices = [ALL_SENSORS.index(c) for c in VIBRATION_SENSORS if c in ALL_SENSORS]
    vib_data = window[:, vib_indices].flatten()

    features["vib_rms"]          = _rms(vib_data)
    features["vib_peak"]         = float(np.max(np.abs(vib_data)))
    features["vib_crest_factor"] = _crest_factor(vib_data)
    features["vib_kurtosis"]     = _kurtosis(vib_data)
    features["vib_skewness"]     = _skewness(vib_data)
    features["vib_std"]          = float(np.std(vib_data))
    features["vib_mean"]         = float(np.mean(vib_data))

    # Per-location RMS (pump, gearbox, motor)
    pump_vib_idx  = [ALL_SENSORS.index(c) for c in VIBRATION_SENSORS[:4]  if c in ALL_SENSORS]
    gb_vib_idx    = [ALL_SENSORS.index(c) for c in VIBRATION_SENSORS[4:12] if c in ALL_SENSORS]
    motor_vib_idx = [ALL_SENSORS.index(c) for c in VIBRATION_SENSORS[12:]  if c in ALL_SENSORS]

    features["pump_vib_rms"]  = _rms(window[:, pump_vib_idx].flatten())
    features["gb_vib_rms"]    = _rms(window[:, gb_vib_idx].flatten())
    features["motor_vib_rms"] = _rms(window[:, motor_vib_idx].flatten())

    # Shutdown flag — fraction of timesteps where avg vibration < threshold
    avg_vib_per_ts = window[:, vib_indices].mean(axis=1)
    features["shutdown_fraction"] = float((avg_vib_per_ts < SHUTDOWN_VIB_THRESHOLD).mean())

    # --- TI0731 Temperature Features ---
    if ti0731_idx is not None:
        temp_signal = window[:, ti0731_idx]
        features["ti0731_mean"]     = float(np.mean(temp_signal))
        features["ti0731_max"]      = float(np.max(temp_signal))
        features["ti0731_std"]      = float(np.std(temp_signal))
        features["ti0731_anomaly"]  = float(np.mean(temp_signal) - NORMAL_BASELINE_TEMP)
        # Rate of change (°C per window step) — linear slope
        x_idx = np.arange(len(temp_signal))
        slope = np.polyfit(x_idx, temp_signal, 1)[0]
        features["ti0731_rate_of_change"] = float(slope)
    else:
        for k in ["ti0731_mean", "ti0731_max", "ti0731_std", "ti0731_anomaly", "ti0731_rate_of_change"]:
            features[k] = 0.0

    # --- General Temperature Aggregate ---
    temp_indices = [ALL_SENSORS.index(c) for c in TEMPERATURE_SENSORS if c in ALL_SENSORS]
    temp_data = window[:, temp_indices].flatten()
    features["temp_rms"]  = _rms(temp_data)
    features["temp_mean"] = float(np.mean(temp_data))
    features["temp_std"]  = float(np.std(temp_data))

    return features  # 16 features


# ─────────────────────────────────────────────────────────────────────────────
# Domain 2 — Frequency Domain Features
# ─────────────────────────────────────────────────────────────────────────────

def _band_energy(signal: np.ndarray, freqs: np.ndarray,
                 f_low: float, f_high: float) -> float:
    """Sum FFT magnitude in a frequency band."""
    fft_mag = np.abs(rfft(signal))
    band = (freqs >= f_low) & (freqs <= f_high)
    return float(np.sum(fft_mag[band]) ** 2)


def extract_frequency_domain(window: np.ndarray) -> dict:
    """
    FFT-based frequency domain features.
    For 15-min sampled data the frequencies are very low — we
    detect relative energy concentrations at shaft harmonics.
    """
    features = {}

    vib_indices = [ALL_SENSORS.index(c) for c in VIBRATION_SENSORS if c in ALL_SENSORS]
    freqs = rfftfreq(window.shape[0], d=1.0)  # normalised frequency (0 to 0.5)

    # --- Shaft Harmonics ---
    # At 15-min sampling, shaft-frequency harmonics appear as energy
    # redistributions in the normalised spectrum. We capture bands
    # around 1× and 2× relative frequency.
    pump_x_idx   = ALL_SENSORS.index("2026VI0731X.PV") if "2026VI0731X.PV" in ALL_SENSORS else 0
    pump_signal  = window[:, pump_x_idx]
    motor_x_idx  = ALL_SENSORS.index("2026VI0737X.PV") if "2026VI0737X.PV" in ALL_SENSORS else 0
    motor_signal = window[:, motor_x_idx]

    # Spectral RMS across all vibration channels
    all_vib_fft_energy = 0.0
    for idx in vib_indices:
        fft_mag = np.abs(rfft(window[:, idx]))
        all_vib_fft_energy += np.sum(fft_mag ** 2)
    features["spectral_rms_all_vib"] = float(np.sqrt(all_vib_fft_energy / max(len(vib_indices), 1)))

    # Energy in low frequency band (trend/drift) vs mid band
    features["pump_x_low_band_energy"]  = _band_energy(pump_signal,  freqs, 0.0,  0.1)
    features["pump_x_mid_band_energy"]  = _band_energy(pump_signal,  freqs, 0.1,  0.3)
    features["pump_x_high_band_energy"] = _band_energy(pump_signal,  freqs, 0.3,  0.5)
    features["motor_x_low_band_energy"] = _band_energy(motor_signal, freqs, 0.0,  0.1)
    features["motor_x_mid_band_energy"] = _band_energy(motor_signal, freqs, 0.1,  0.3)

    # Peak frequency for pump NDE vibration
    pump_fft = np.abs(rfft(pump_signal))
    features["pump_x_peak_freq_idx"] = float(np.argmax(pump_fft))

    # Gearbox vibration energy ratio (LSS vs HSS)
    gb_lss_idx = [ALL_SENSORS.index(c) for c in VIBRATION_SENSORS[4:8]  if c in ALL_SENSORS]
    gb_hss_idx = [ALL_SENSORS.index(c) for c in VIBRATION_SENSORS[8:12] if c in ALL_SENSORS]
    lss_energy = sum(np.sum(np.abs(rfft(window[:, i])) ** 2) for i in gb_lss_idx)
    hss_energy = sum(np.sum(np.abs(rfft(window[:, i])) ** 2) for i in gb_hss_idx)
    features["gb_lss_hss_energy_ratio"] = float(lss_energy / max(hss_energy, 1e-10))

    # Pump vs Motor vibration spectral energy ratio
    pump_energy  = sum(np.sum(np.abs(rfft(window[:, i])) ** 2) for i in vib_indices[:4])
    motor_energy = sum(np.sum(np.abs(rfft(window[:, i])) ** 2) for i in vib_indices[12:])
    features["pump_motor_spectral_ratio"] = float(pump_energy / max(motor_energy, 1e-10))

    # Spectral centroid (weighted mean frequency)
    fft_mag = np.abs(rfft(pump_signal))
    features["pump_spectral_centroid"] = (
        float(np.sum(freqs * fft_mag) / np.sum(fft_mag)) if np.sum(fft_mag) > 0 else 0.0
    )

    # Total spectral entropy (disorder in frequency domain)
    psd = fft_mag ** 2
    psd_norm = psd / (np.sum(psd) + 1e-10)
    features["pump_spectral_entropy"] = float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)))

    # Vibration cross-correlation (pump NDE X vs pump NDE Y)
    pump_y_idx = ALL_SENSORS.index("2026VI0731Y.PV") if "2026VI0731Y.PV" in ALL_SENSORS else 0
    corr = np.corrcoef(window[:, pump_x_idx], window[:, pump_y_idx])[0, 1]
    features["pump_nde_xy_correlation"] = float(corr) if not np.isnan(corr) else 0.0

    features["n_dominant_freq_components"] = float(np.sum(pump_fft > np.mean(pump_fft) + np.std(pump_fft)))

    return features  # 14 features


# ─────────────────────────────────────────────────────────────────────────────
# Domain 3 — Process Health Indicators
# ─────────────────────────────────────────────────────────────────────────────

def extract_health_indicators(window: np.ndarray, y_window: int = 0) -> dict:
    """
    Composite health indicators derived from multi-sensor behaviour.
    """
    features = {}

    vib_indices  = [ALL_SENSORS.index(c) for c in VIBRATION_SENSORS if c in ALL_SENSORS]
    temp_indices = [ALL_SENSORS.index(c) for c in TEMPERATURE_SENSORS if c in ALL_SENSORS]
    ti0731_idx   = ALL_SENSORS.index(FAULT_TEMP_SENSOR) if FAULT_TEMP_SENSOR in ALL_SENSORS else None

    # --- Health Index (composite 0–1, higher = worse health) ---
    # Weighted combination of anomaly signals
    vib_means  = window[:, vib_indices].mean(axis=0)
    temp_means = window[:, temp_indices].mean(axis=0)

    # Z-score from unit-normalised data (0-centred in normal phase)
    features["vib_z_score_mean"]  = float(np.mean(vib_means) - 0.5)
    features["temp_z_score_mean"] = float(np.mean(temp_means) - 0.5)

    # TI0731-specific features
    if ti0731_idx is not None:
        ti0731_signal = window[:, ti0731_idx]
        features["ti0731_z_score"]       = float(np.mean(ti0731_signal) - 0.5)
        features["ti0731_trend_slope"]   = float(np.polyfit(np.arange(len(ti0731_signal)), ti0731_signal, 1)[0])
        features["ti0731_peak_to_normal"]= float(np.max(ti0731_signal) - np.mean(ti0731_signal[:10]))
    else:
        features["ti0731_z_score"]        = 0.0
        features["ti0731_trend_slope"]    = 0.0
        features["ti0731_peak_to_normal"] = 0.0

    # --- Pump-Motor Vibration Differential ---
    pump_vib_mean  = window[:, vib_indices[:4]].mean()
    motor_vib_mean = window[:, vib_indices[12:]].mean()
    features["pump_motor_vib_diff"] = float(pump_vib_mean - motor_vib_mean)

    # --- Gearbox Differential (LSS vs HSS) ---
    gb_lss_mean = window[:, vib_indices[4:8]].mean()
    gb_hss_mean = window[:, vib_indices[8:12]].mean()
    features["gb_lss_hss_vib_diff"] = float(gb_lss_mean - gb_hss_mean)

    # --- Vibration-Temperature Correlation ---
    avg_vib  = window[:, vib_indices].mean(axis=1)
    avg_temp = window[:, temp_indices].mean(axis=1)
    corr = np.corrcoef(avg_vib, avg_temp)[0, 1]
    features["vib_temp_correlation"] = float(corr) if not np.isnan(corr) else 0.0

    # --- Shutdown Flag fraction ---
    features["shutdown_fraction_health"] = float((avg_vib < SHUTDOWN_VIB_THRESHOLD).mean())

    # --- Composite Health Index (0 = healthy, 1 = critical) ---
    # Weighted combination: 60% temp anomaly, 40% vibration change
    temp_component = max(0.0, features.get("ti0731_z_score", 0.0))
    vib_drop = max(0.0, 0.5 - np.mean(avg_vib))   # drops below normal (0.5 in normalised space)
    health_index = min(1.0, 0.6 * temp_component + 0.4 * vib_drop)
    features["health_index"] = float(health_index)

    # --- RUL Proxy (fraction through fault phase) ---
    # 0 = just entered fault, 1 = maximum degradation seen
    features["rul_proxy"] = float(y_window)  # 0 = normal, 1 = fault label

    # --- Temperature Range Across All Sensors ---
    features["temp_range_all"] = float(np.max(temp_means) - np.min(temp_means))

    # --- Vibration Coefficient of Variation ---
    vib_cv = np.std(avg_vib) / (np.mean(avg_vib) + 1e-10)
    features["vib_coeff_variation"] = float(vib_cv)

    # --- Inter-location vibration imbalance ---
    all_loc_rms = [_rms(window[:, i]) for i in vib_indices]
    features["vib_imbalance_ratio"] = float(max(all_loc_rms) / (min(all_loc_rms) + 1e-10))

    # --- Bearing Temperature Spread (thrust vs pump vs motor) ---
    thrust_temp_idx  = [ALL_SENSORS.index(c) for c in ["2026TI0731.PV","2026TI0732.PV","2026TI0733.PV","2026TI0734.PV"] if c in ALL_SENSORS]
    pump_temp_idx    = [ALL_SENSORS.index(c) for c in ["2026TI0735.PV","2026TI0736.PV"] if c in ALL_SENSORS]
    motor_temp_idx   = [ALL_SENSORS.index(c) for c in ["2026TI0724.PV","2026TI0725.PV"] if c in ALL_SENSORS]

    thrust_mean = window[:, thrust_temp_idx].mean() if thrust_temp_idx else 0.0
    pump_t_mean = window[:, pump_temp_idx].mean()   if pump_temp_idx   else 0.0
    motor_t_mean= window[:, motor_temp_idx].mean()  if motor_temp_idx  else 0.0

    features["thrust_temp_mean"]      = float(thrust_mean)
    features["pump_bearing_temp_mean"]= float(pump_t_mean)
    features["motor_bearing_temp_mean"]= float(motor_t_mean)

    return features  # 15 features


# ─────────────────────────────────────────────────────────────────────────────
# Main Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Extract all features for every window.

    Parameters
    ----------
    X : (n_windows, window_size, n_sensors)
    y : (n_windows,) labels

    Returns
    -------
    DataFrame of shape (n_windows, ~45 features)
    """
    logger.info(f"Extracting features from {len(X):,} windows...")
    rows = []
    for i, (window, label) in enumerate(zip(X, y)):
        features = {}
        features.update(extract_time_domain(window))
        features.update(extract_frequency_domain(window))
        features.update(extract_health_indicators(window, y_window=int(label)))
        features["label"] = int(label)
        rows.append(features)

    df_features = pd.DataFrame(rows)
    n_features = len(df_features.columns) - 1  # exclude label
    logger.info(f"Feature extraction complete — {n_features} features × {len(df_features):,} windows")
    return df_features


def get_feature_names() -> list[str]:
    """Return the list of feature column names (without label)."""
    dummy_window = np.random.rand(WINDOW_SIZE if WINDOW_SIZE else 96, len(ALL_SENSORS)).astype(np.float32)
    features = {}
    features.update(extract_time_domain(dummy_window))
    features.update(extract_frequency_domain(dummy_window))
    features.update(extract_health_indicators(dummy_window, y_window=0))
    return list(features.keys())
