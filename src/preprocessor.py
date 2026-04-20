

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATE_COLUMN, FAULT_LABEL_START, NORMAL_PHASE_END,
    VIBRATION_SENSORS, TEMPERATURE_SENSORS, ALL_SENSORS,
    WINDOW_SIZE, WINDOW_STEP, SHUTDOWN_VIB_THRESHOLD,
    MAX_INTERP_GAP, BUTTERWORTH_ORDER, BUTTERWORTH_CUTOFF_HZ,
    FAULT_TEMP_THRESHOLD, FAULT_TEMP_SENSOR, MODELS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Missing Value Treatment
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linear interpolation for gaps ≤ MAX_INTERP_GAP consecutive NaNs.
    Longer gaps are left as NaN and will be excluded during windowing.
    """
    df = df.copy()
    total_before = df[ALL_SENSORS].isnull().sum().sum()

    for col in ALL_SENSORS:
        # Count consecutive NaN runs
        is_null   = df[col].isnull()
        null_runs = is_null.groupby((is_null != is_null.shift()).cumsum()).transform("sum")

        # Only interpolate short runs
        mask = is_null & (null_runs <= MAX_INTERP_GAP)
        df.loc[mask, col] = np.nan  # keep as NaN temporarily
        df[col] = df[col].interpolate(method="linear", limit=MAX_INTERP_GAP)

    total_after = df[ALL_SENSORS].isnull().sum().sum()
    filled = total_before - total_after
    logger.info(f"Missing values: {total_before} → {total_after} "
                f"({filled} filled by interpolation, {total_after} remain as NaN)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Noise Filtering (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

def _butterworth_lowpass(signal: np.ndarray, cutoff: float, order: int) -> np.ndarray:
    """
    Apply zero-phase Butterworth low-pass filter. Handles NaNs by skipping.
    
    FIXED: Now preserves NaN locations after filtering.
    """
    if np.all(np.isnan(signal)):
        return signal
    
    # ✅ PRESERVE NaN LOCATIONS
    nan_mask = np.isnan(signal)
    
    b, a = butter(order, cutoff, btype="low", analog=False)
    
    # Fill NaN for filtering
    s = pd.Series(signal).ffill().bfill().values
    filtered = filtfilt(b, a, s)
    
    # ✅ RESTORE NaN LOCATIONS
    filtered[nan_mask] = np.nan
    
    return filtered


def _kalman_smooth(signal: np.ndarray) -> np.ndarray:
    """
    Simple 1D Kalman smoother for temperature drift removal.
    Models signal as a random walk with observation noise.
    
    FIXED: Now preserves NaN locations after smoothing.
    """
    if np.all(np.isnan(signal)):
        return signal

    # ✅ PRESERVE NaN LOCATIONS
    nan_mask = np.isnan(signal)
    
    n = len(signal)
    smoothed = np.zeros(n)

    # Fill NaN before Kalman
    s = pd.Series(signal).ffill().bfill().values

    # Kalman parameters — tuned for slow temperature drift
    Q = 1e-5   # process noise
    R = 0.5    # observation noise
    P = 1.0    # initial estimate error
    x = s[0]   # initial state

    for i in range(n):
        # Predict
        P += Q
        # Update
        K = P / (P + R)
        x = x + K * (s[i] - x)
        P = (1 - K) * P
        smoothed[i] = x

    # ✅ RESTORE NaN LOCATIONS
    smoothed[nan_mask] = np.nan
    
    return smoothed


def apply_noise_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Butterworth low-pass filter to vibration sensors.
    Apply Kalman smoother to temperature sensors.
    """
    df = df.copy()
    logger.info("Applying noise filtering...")

    for col in VIBRATION_SENSORS:
        if col in df.columns:
            df[col] = _butterworth_lowpass(
                df[col].values, BUTTERWORTH_CUTOFF_HZ, BUTTERWORTH_ORDER
            )

    for col in TEMPERATURE_SENSORS:
        if col in df.columns:
            df[col] = _kalman_smooth(df[col].values)

    logger.info("Noise filtering complete — Butterworth (vibration) + Kalman (temperature)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Normalisation (ENHANCED LOGGING)
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame) -> MinMaxScaler:
    """
    Fit Min-Max scaler on normal phase data ONLY.
    This prevents fault-phase statistics from contaminating normalisation.
    """
    normal_df = df[df[DATE_COLUMN] <= NORMAL_PHASE_END][ALL_SENSORS].dropna()
    
    if len(normal_df) == 0:
        logger.error("❌ ERROR: No normal-phase data found for scaler fitting!")
        logger.error(f"   Check NORMAL_PHASE_END date: {NORMAL_PHASE_END}")
        raise ValueError("Cannot fit scaler: no normal data")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(normal_df)

    # ✅ LOG SCALER STATISTICS
    logger.info(f"Scaler fitted on {len(normal_df):,} normal-phase rows")
    logger.info(f"Scaler data_min (first 5 sensors): {scaler.data_min_[:5]}")
    logger.info(f"Scaler data_max (first 5 sensors): {scaler.data_max_[:5]}")

    # Save scaler for dashboard use
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    logger.info(f"Scaler saved → {MODELS_DIR / 'scaler.pkl'}")
    
    return scaler


def apply_normalisation(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """
    Apply saved scaler to all data. NaN rows are kept as NaN.
    
    ENHANCED: Now logs normalization statistics and validates output.
    """
    df = df.copy()
    non_null_mask = df[ALL_SENSORS].notna().all(axis=1)
    
    # ✅ LOG HOW MANY ROWS WILL BE NORMALIZED
    n_to_normalize = non_null_mask.sum()
    n_total = len(df)
    logger.info(f"Normalizing {n_to_normalize:,} / {n_total:,} rows "
                f"({n_to_normalize/n_total*100:.1f}% of data)")
    
    if n_to_normalize == 0:
        logger.error("❌ ERROR: No complete rows to normalize!")
        logger.error("   All rows have at least one NaN sensor")
        raise ValueError("Cannot normalize: all rows have NaN")
    
    # Apply normalization
    df.loc[non_null_mask, ALL_SENSORS] = scaler.transform(
        df.loc[non_null_mask, ALL_SENSORS]
    )
    
    # ✅ VERIFY NORMALIZATION WORKED
    if non_null_mask.any():
        normalized_values = df.loc[non_null_mask, ALL_SENSORS].values
        min_val = normalized_values.min()
        max_val = normalized_values.max()
        mean_val = normalized_values.mean()
        
        logger.info(f"Normalized data range: [{min_val:.6f}, {max_val:.6f}]")
        logger.info(f"Normalized data mean:  {mean_val:.6f}")
        
        # Validate range
        if min_val < -0.1 or max_val > 1.1:
            logger.error(f"❌ ERROR: Normalization failed!")
            logger.error(f"   Values outside [0,1] range: [{min_val:.4f}, {max_val:.4f}]")
            logger.error(f"   This will cause LSTM reconstruction errors to explode!")
            raise ValueError("Normalization produced invalid range")
        else:
            logger.info("✅ Normalization validated: data in [0, 1] range")
    
    logger.info("Min-Max normalisation applied (scaler fitted on normal phase only)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Phase Segmentation & Labeling
# ─────────────────────────────────────────────────────────────────────────────

def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign binary health labels:
      0 = Normal  (before 2026-02-28)
      1 = Fault   (from 2026-02-28 onward)

    Also adds:
      shutdown_flag : 1 if avg vibration < SHUTDOWN_VIB_THRESHOLD (pump trip)
      temp_anomaly  : TI0731 deviation from normal-phase baseline
      days_from_fault_onset : RUL proxy (0 at fault onset, negative in normal phase)
    """
    df = df.copy()

    # Primary label
    df["label"] = 0
    df.loc[df[DATE_COLUMN] >= FAULT_LABEL_START, "label"] = 1

    # Shutdown flag — pump tripping indicator
    vib_cols_present = [c for c in VIBRATION_SENSORS if c in df.columns]
    df["avg_vibration"] = df[vib_cols_present].mean(axis=1)
    df["shutdown_flag"] = (df["avg_vibration"] < SHUTDOWN_VIB_THRESHOLD).astype(int)

    # Temperature anomaly — deviation of TI0731 from normal baseline
    from config import NORMAL_BASELINE_TEMP
    if FAULT_TEMP_SENSOR in df.columns:
        df["temp_anomaly"] = df[FAULT_TEMP_SENSOR] - NORMAL_BASELINE_TEMP

    # Days from fault onset (negative = before fault, positive = after)
    fault_onset = pd.Timestamp(FAULT_LABEL_START)
    df["days_from_fault_onset"] = (df[DATE_COLUMN] - fault_onset).dt.total_seconds() / 86400

    normal_count = (df["label"] == 0).sum()
    fault_count  = (df["label"] == 1).sum()
    shutdown_count = df["shutdown_flag"].sum()
    logger.info(f"Labels assigned — Normal: {normal_count:,} | Fault: {fault_count:,} | "
                f"Shutdown events: {shutdown_count:,}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Sliding Window Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    step: int = WINDOW_STEP,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]: 
    """
    Segment the normalised time-series into sliding windows.

    Returns
    -------
    X          : (n_windows, window_size, n_sensors) float array
    y          : (n_windows,) int array — majority label per window
    timestamps : (n_windows,) list of window-end timestamps
    """
    sensor_cols = ALL_SENSORS
    feature_matrix = df[sensor_cols].values  # (n_samples, 35)
    labels_arr     = df["label"].values
    timestamps_arr = df[DATE_COLUMN].values

    X, y, ts = [], [], []
    n = len(df)

    for start in range(0, n - window_size + 1, step):
        end    = start + window_size
        window = feature_matrix[start:end, :]

        # Skip windows with any NaN
        if np.isnan(window).any():
            continue

        X.append(window)
        # Majority label: fault if >50% of window readings are in fault phase
        y.append(int(labels_arr[start:end].mean() > 0.5))
        ts.append(pd.Timestamp(timestamps_arr[end - 1]))

    X = np.array(X, dtype=np.float32)   # (n_windows, window_size, 35)  
    y = np.array(y, dtype=np.int32)

    n_normal = (y == 0).sum()
    n_fault  = (y == 1).sum()
    logger.info(f"Windows extracted — Total: {len(X):,} | Normal: {n_normal} | Fault: {n_fault}")
    logger.info(f"Window shape: {X.shape}  (windows × timesteps × sensors)")

    return X, y, ts


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE (ENHANCED WITH VALIDATION)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(raw_df: pd.DataFrame) -> dict:
    """
    Execute all 6 preprocessing steps and return a dictionary
    with the processed data ready for model training.
    
    ENHANCED: Now includes data validation at each critical step.
    """
    logger.info("=" * 60)
    logger.info("Starting full preprocessing pipeline")
    logger.info("=" * 60)

    # Step 2
    df = handle_missing_values(raw_df)

    # Step 3
    df = apply_noise_filtering(df)

    # Step 4 — fit scaler on normal, apply to all
    scaler = fit_scaler(df)
    df_norm = apply_normalisation(df, scaler)

    # ✅ VALIDATE NORMALIZATION BEFORE PROCEEDING
    logger.info("\n" + "="*60)
    logger.info("VALIDATING NORMALIZED DATA")
    logger.info("="*60)
    
    # Check a sample of normalized values
    sample_data = df_norm[ALL_SENSORS].dropna().head(100)
    if len(sample_data) > 0:
        logger.info(f"Sample normalized data (first 100 rows):")
        logger.info(f"  Min:  {sample_data.values.min():.6f}")
        logger.info(f"  Max:  {sample_data.values.max():.6f}")
        logger.info(f"  Mean: {sample_data.values.mean():.6f}")
        
        if sample_data.values.min() < -0.1 or sample_data.values.max() > 1.1:
            logger.error("❌ CRITICAL ERROR: Normalized data outside [0,1] range!")
            raise ValueError("Normalization validation failed")
    logger.info("="*60 + "\n")

    # Step 5
    df_labeled = assign_labels(df_norm)

    # Step 6
    X, y, timestamps = extract_windows(df_labeled)

    # ✅ FINAL VALIDATION OF WINDOW DATA
    logger.info("\n" + "="*60)
    logger.info("VALIDATING EXTRACTED WINDOWS")
    logger.info("="*60)
    logger.info(f"Window array shape: {X.shape}")
    logger.info(f"Window data type:   {X.dtype}")
    logger.info(f"Window min value:   {X.min():.6f}")
    logger.info(f"Window max value:   {X.max():.6f}")
    logger.info(f"Window mean value:  {X.mean():.6f}")
    logger.info(f"Window std value:   {X.std():.6f}")
    logger.info(f"NaN count in windows: {np.isnan(X).sum()}")
    
    if X.min() < -0.1 or X.max() > 1.1:
        logger.error("❌ CRITICAL ERROR: Window data outside [0,1] range!")
        logger.error("   This will cause LSTM errors to be ~10,000x too large!")
        raise ValueError("Window validation failed")
    
    if np.isnan(X).any():
        logger.error("❌ CRITICAL ERROR: NaN values in extracted windows!")
        raise ValueError("Windows contain NaN values")
    
    logger.info("✅ All validations passed!")
    logger.info("="*60 + "\n")

    # Train / Val / Test split by date (NO shuffle to prevent leakage)
    n = len(X)
    test_cut = int(n * 0.90)
    val_cut  = int(n * 0.70)

    X_train, y_train = X[:val_cut],         y[:val_cut]
    X_val,   y_val   = X[val_cut:test_cut], y[val_cut:test_cut]
    X_test,  y_test  = X[test_cut:],        y[test_cut:]
    ts_test          = timestamps[test_cut:]

    logger.info(f"Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    logger.info("Preprocessing pipeline complete.")

    return {
        "df_processed":  df_labeled,
        "scaler":        scaler,
        "X_train":       X_train,
        "y_train":       y_train,
        "X_val":         X_val,
        "y_val":         y_val,
        "X_test":        X_test,
        "y_test":        y_test,
        "ts_test":       ts_test,
        "X_all":         X,
        "y_all":         y,
        "ts_all":        timestamps,
    }