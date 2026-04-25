

import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from config import (
    OUTPUT_DIR, MODELS_DIR, DATA_DIR, DATE_COLUMN,
    FAULT_ONSET_DATE, FAULT_TEMP_SENSOR,
    VIBRATION_SENSORS, TEMPERATURE_SENSORS, ALL_SENSORS,
    WINDOW_SIZE, WINDOW_STEP,
    TARGET_RECALL, TARGET_PRECISION, TARGET_ROC_AUC, TARGET_MAX_FPR,
)

# ── Ensure output directories exist ──────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — Must be defined BEFORE they're used
# ═════════════════════════════════════════════════════════════════════════════

def _safe_kurtosis(x: np.ndarray) -> float:
    if len(x) < 4 or np.std(x) == 0:
        return 0.0
    from scipy.stats import kurtosis
    return float(kurtosis(x, fisher=True))


def _safe_skew(x: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0:
        return 0.0
    from scipy.stats import skew
    return float(skew(x))


def _fft_features(vals: np.ndarray, prefix: str) -> dict:
    """Compute FFT-based frequency domain features."""
    feats = {}
    try:
        fft_vals  = np.abs(np.fft.rfft(vals))
        freqs     = np.fft.rfftfreq(len(vals))
        total_pow = np.sum(fft_vals**2) + 1e-10
        
        feats[f"{prefix}_spectral_centroid"] = float(
            np.sum(freqs * fft_vals**2) / total_pow
        )
        feats[f"{prefix}_spectral_entropy"] = float(
            -np.sum((fft_vals**2 / total_pow) *
                    np.log(fft_vals**2 / total_pow + 1e-10))
        )
        feats[f"{prefix}_dominant_freq"] = float(freqs[np.argmax(fft_vals)])
        feats[f"{prefix}_peak_freq_mag"] = float(np.max(fft_vals))
        
        # Band energies (low / mid / high)
        n = len(freqs)
        feats[f"{prefix}_band_low"]  = float(np.sum(fft_vals[:n//3]**2)  / total_pow)
        feats[f"{prefix}_band_mid"]  = float(np.sum(fft_vals[n//3:2*n//3]**2) / total_pow)
        feats[f"{prefix}_band_high"] = float(np.sum(fft_vals[2*n//3:]**2) / total_pow)
        
    except Exception:
        pass
    return feats


def _estimate_days_to_onset(
    slope: float,
    current_prob: float,
    threshold: float = 0.7,
    window_interval_hours: float = 12.0,
) -> float:
    """
    Estimate days until fault probability reaches threshold.
    Based on linear extrapolation of probability trend.
    """
    if slope <= 0:
        # Not increasing — use fixed estimate based on current probability
        if current_prob > 0.4:
            return 7.0
        elif current_prob > 0.2:
            return 30.0
        else:
            return None
    
    # Windows needed to reach threshold
    prob_gap       = max(threshold - current_prob, 0)
    windows_needed = prob_gap / slope if slope > 0 else float("inf")
    days_needed    = windows_needed * window_interval_hours / 24.0
    
    # Cap at reasonable range
    return min(float(days_needed), 365.0)


def _get_recommendation(fc: int, risk_level: str, fault_class_names: dict) -> str:
    """Return maintenance recommendation based on fault class and risk level."""
    recommendations = {
        2: {
            "Critical":       "Emergency bearing inspection. Check BPFO vibration signature.",
            "High":           "Schedule bearing inspection within 7 days. Monitor vibration impulses.",
            "Medium":         "Increase vibration monitoring frequency. Check lubrication.",
            "Low-Developing": "Monitor bearing temperature and vibration trends.",
            "Low":            "Continue scheduled maintenance intervals.",
        },
        3: {
            "Critical":       "Immediate shutdown for balance correction. Check rotor.",
            "High":           "Schedule dynamic balancing within 14 days.",
            "Medium":         "Monitor 1× RPM component. Check coupling alignment.",
            "Low-Developing": "Track vibration amplitude at running speed.",
            "Low":            "Include balance check in next planned outage.",
        },
        4: {
            "Critical":       "Shutdown for alignment correction. Check coupling wear.",
            "High":           "Schedule precision alignment within 7 days.",
            "Medium":         "Monitor 2× RPM component. Check foundation bolts.",
            "Low-Developing": "Laser alignment check at next opportunity.",
            "Low":            "Include alignment verification in next PM.",
        },
        5: {
            "Critical":       "Check suction conditions immediately. Risk of impeller damage.",
            "High":           "Inspect suction strainer. Check NPSH margin.",
            "Medium":         "Monitor suction pressure. Check for air ingestion.",
            "Low-Developing": "Verify suction head and flow conditions.",
            "Low":            "Review operating point vs pump curve.",
        },
        6: {
            "Critical":       "Seal replacement required. Risk of leakage/contamination.",
            "High":           "Inspect mechanical seal within 7 days.",
            "Medium":         "Monitor seal chamber temperature and leakage.",
            "Low-Developing": "Check seal flush plan. Monitor seal face condition.",
            "Low":            "Include seal inspection in next planned outage.",
        },
        7: {
            "Critical":       "Gearbox inspection required. Check gear mesh and lubrication.",
            "High":           "Vibration analysis at GMF. Oil sample analysis.",
            "Medium":         "Monitor gearbox temperature and vibration at GMF.",
            "Low-Developing": "Check gear mesh condition at next inspection.",
            "Low":            "Continue oil analysis program.",
        },
        8: {
            "Critical":       "Motor bearing replacement required. Check BPFI/BPFO.",
            "High":           "Motor bearing inspection within 7 days.",
            "Medium":         "Monitor motor bearing temperature and vibration.",
            "Low-Developing": "Check motor bearing lubrication.",
            "Low":            "Include motor bearing check in next PM.",
        },
    }
    fault_recs = recommendations.get(fc, {})
    return fault_recs.get(risk_level, "Monitor and follow standard maintenance schedule.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Data
# ═════════════════════════════════════════════════════════════════════════════

def load_training_data():
    """
    Load training data.
    Tries augmented data first (real + synthetic, 9 classes).
    Falls back to real data only (2 classes).
    
    Returns:
        df          : Full dataframe
        use_augmented: Whether augmented data was loaded
    """
    augmented_path = DATA_DIR / "Augmented_DCS_Data.xlsx"
    
    if augmented_path.exists():
        log.info("Loading AUGMENTED training data (real + synthetic, 9 classes)...")
        df = pd.read_excel(
            augmented_path,
            sheet_name="Augmented_Data",
            header=0,
        )
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        
        # Ensure fault_class column exists
        if "fault_class" not in df.columns:
            log.warning("fault_class column not found in augmented data!")
            df["fault_class"] = 0
        
        # Create binary label (0=normal, 1=any fault)
        df["label"] = (df["fault_class"] > 0).astype(int)
        
        log.info(f"Augmented data loaded: {len(df):,} rows")
        log.info(f"Fault class distribution:\n{df['fault_class'].value_counts().sort_index()}")
        
        return df, True
    
    else:
        log.warning("Augmented data not found. Falling back to real data only...")
        from config import RAW_DATA_FILE, DATA_SHEET
        
        if not RAW_DATA_FILE.exists():
            raise FileNotFoundError(f"No data found at {RAW_DATA_FILE}")
        
        df = pd.read_excel(RAW_DATA_FILE, sheet_name=DATA_SHEET, header=0)
        df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        
        fault_onset = pd.Timestamp(FAULT_ONSET_DATE)
        df["fault_class"] = 0
        df.loc[df[DATE_COLUMN] >= fault_onset, "fault_class"] = 1
        df["label"] = df["fault_class"]
        
        log.info(f"Real data loaded: {len(df):,} rows")
        return df, False


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Feature Engineering
# ═════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-domain and frequency-domain features from sensor windows.
    Works for both real and synthetic data.
    """
    log.info("Engineering features from sensor windows...")
    
    # Identify available sensors
    vib_sensors  = [s for s in VIBRATION_SENSORS    if s in df.columns]
    temp_sensors = [s for s in TEMPERATURE_SENSORS  if s in df.columns]
    all_sens     = vib_sensors + temp_sensors
    
    # Convert sensor columns to numeric
    for col in all_sens:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    
    records = []
    n = len(df)
    
    for start in range(0, n - WINDOW_SIZE + 1, WINDOW_STEP):
        end   = start + WINDOW_SIZE
        chunk = df.iloc[start:end]
        
        rec = {
            "timestamp":   chunk[DATE_COLUMN].iloc[-1],
            "true_label":  int(chunk["label"].mode()[0]),
            "fault_class": int(chunk["fault_class"].mode()[0]),
        }
        
        # Add is_synthetic flag if available
        if "is_synthetic" in chunk.columns:
            rec["is_synthetic"] = bool(chunk["is_synthetic"].iloc[-1])
        else:
            rec["is_synthetic"] = False
        
        # ── Time-domain features ──────────────────────────────────────────
        for s in all_sens:
            vals = chunk[s].dropna().values
            if len(vals) == 0:
                continue
            prefix = s.lower().replace(" ", "_")
            rec[f"{prefix}_mean"]  = float(np.mean(vals))
            rec[f"{prefix}_std"]   = float(np.std(vals))
            rec[f"{prefix}_max"]   = float(np.max(vals))
            rec[f"{prefix}_min"]   = float(np.min(vals))
            rec[f"{prefix}_rms"]   = float(np.sqrt(np.mean(vals**2)))
            rec[f"{prefix}_kurt"]  = float(_safe_kurtosis(vals))
            rec[f"{prefix}_skew"]  = float(_safe_skew(vals))
            rec[f"{prefix}_p2p"]   = float(np.max(vals) - np.min(vals))
            rec[f"{prefix}_crest"] = float(
                np.max(np.abs(vals)) / rec[f"{prefix}_rms"]
                if rec[f"{prefix}_rms"] > 0 else 0.0
            )
        
        # ── Vibration composite features ──────────────────────────────────
        if vib_sensors:
            vib_data = chunk[vib_sensors].dropna(axis=0)
            if len(vib_data) > 0:
                vib_arr = vib_data.values
                rec["vib_rms_mean"]    = float(np.sqrt(np.mean(vib_arr**2)))
                rec["vib_peak_max"]    = float(np.max(np.abs(vib_arr)))
                rec["vib_std_mean"]    = float(np.mean(np.std(vib_arr, axis=0)))
                rec["vib_kurt_mean"]   = float(np.mean([_safe_kurtosis(vib_arr[:, i])
                                                         for i in range(vib_arr.shape[1])]))
                
                # Cross-sensor correlation
                if vib_arr.shape[1] >= 2:
                    try:
                        corr_mat = np.corrcoef(vib_arr.T)
                        upper    = corr_mat[np.triu_indices_from(corr_mat, k=1)]
                        rec["vib_corr_mean"] = float(np.nanmean(upper))
                        rec["vib_corr_max"]  = float(np.nanmax(np.abs(upper)))
                    except Exception:
                        rec["vib_corr_mean"] = 0.0
                        rec["vib_corr_max"]  = 0.0
                
                # Frequency-domain features (FFT)
                for i, s in enumerate(vib_sensors[:4]):
                    col_vals = chunk[s].dropna().values
                    if len(col_vals) >= 8:
                        fft_feats = _fft_features(col_vals, prefix=s.lower())
                        rec.update(fft_feats)
        
        # ── Temperature composite features ────────────────────────────────
        if temp_sensors:
            temp_data = chunk[temp_sensors].dropna(axis=0)
            if len(temp_data) > 0:
                temp_arr = temp_data.values
                rec["temp_mean_all"]  = float(np.mean(temp_arr))
                rec["temp_max_all"]   = float(np.max(temp_arr))
                rec["temp_std_all"]   = float(np.std(temp_arr))
                rec["temp_rise_rate"] = float(
                    (temp_arr[-1].mean() - temp_arr[0].mean())
                    / max(WINDOW_SIZE, 1)
                )
        
        # ── Fault-specific sensor features ───────────────────────────────
        if FAULT_TEMP_SENSOR in df.columns:
            ti_vals = chunk[FAULT_TEMP_SENSOR].dropna().values
            if len(ti_vals) > 0:
                rec["ti0731_mean"]        = float(np.mean(ti_vals))
                rec["ti0731_max"]         = float(np.max(ti_vals))
                rec["ti0731_rise_rate"]   = float(ti_vals[-1] - ti_vals[0]) / max(len(ti_vals), 1)
                rec["ti0731_above_165"]   = float(np.mean(ti_vals > 165))
                rec["ti0731_above_200"]   = float(np.mean(ti_vals > 200))
        
        records.append(rec)
    
    feat_df = pd.DataFrame(records)
    
    # Fill NaN values
    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
    feat_df[numeric_cols] = feat_df[numeric_cols].fillna(0)
    
    log.info(f"Feature engineering complete: {len(feat_df):,} windows, "
             f"{len(feat_df.columns)} columns")
    return feat_df


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train Models
# ═════════════════════════════════════════════════════════════════════════════

def train_isolation_forest(X_train: np.ndarray):
    """Train Isolation Forest for anomaly detection."""
    from sklearn.ensemble import IsolationForest
    log.info("Training Isolation Forest...")
    
    model = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def train_lstm_autoencoder(X_train: np.ndarray, input_dim: int):
    """Train LSTM Autoencoder for reconstruction-error anomaly detection."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
        )
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        log.warning("TensorFlow not available. Skipping LSTM training.")
        return None
    
    log.info("Training LSTM Autoencoder...")
    
    # Reshape for LSTM: (samples, timesteps, features)
    X_3d = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    
    inp  = Input(shape=(1, input_dim))
    x    = LSTM(64, return_sequences=False)(inp)
    x    = Dropout(0.2)(x)
    x    = RepeatVector(1)(x)
    x    = LSTM(64, return_sequences=True)(x)
    x    = Dropout(0.2)(x)
    out  = TimeDistributed(Dense(input_dim))(x)
    
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
    ]
    
    model.fit(
        X_3d, X_3d,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=0,
    )
    
    return model


def train_xgboost_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fault_class_train: np.ndarray,
    use_smote: bool = True,
):
    """
    Train XGBoost for multi-class fault detection (9 classes).
    Applies SMOTE for class balancing when training on augmented data.
    """
    from xgboost import XGBClassifier
    log.info("Training XGBoost Multi-Class Classifier (9 classes)...")
    
    X_mc = X_train.copy()
    y_mc = fault_class_train.copy()
    
    # Apply SMOTE for class balancing
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            unique, counts = np.unique(y_mc, return_counts=True)
            min_count = counts.min()
            
            # Only apply SMOTE if we have enough samples
            if min_count >= 2:
                k_neighbors = min(5, min_count - 1)
                smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
                X_mc, y_mc = smote.fit_resample(X_mc, y_mc)
                log.info(f"SMOTE applied: {len(X_mc):,} samples after resampling")
            else:
                log.warning("Not enough samples for SMOTE. Training without resampling.")
        except ImportError:
            log.warning("imbalanced-learn not available. Training without SMOTE.")
        except Exception as e:
            log.warning(f"SMOTE failed: {e}. Training without resampling.")
    
    n_classes = len(np.unique(y_mc))
    log.info(f"Training XGBoost with {n_classes} classes...")
    
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=9,          # Always 9 classes for consistency
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    
    model.fit(
        X_mc, y_mc,
        eval_set=[(X_mc, y_mc)],
        verbose=False,
    )
    
    return model


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Generate Predictions
# ═════════════════════════════════════════════════════════════════════════════

def generate_predictions(
    feat_df: pd.DataFrame,
    if_model,
    lstm_model,
    xgb_model,
    scaler,
    feature_cols: list,
    fault_class_names: dict,
    use_augmented: bool,
) -> pd.DataFrame:
    """Generate predictions from all three models on feature windows."""
    
    log.info("Generating predictions...")
    
    X = feat_df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    # ── Isolation Forest ──────────────────────────────────────────────────────
    if_raw    = if_model.decision_function(X_scaled)
    if_scores = 1 - (if_raw - if_raw.min()) / (if_raw.max() - if_raw.min() + 1e-10)
    if_preds  = (if_model.predict(X_scaled) == -1).astype(int)
    
    # ── LSTM Autoencoder ──────────────────────────────────────────────────────
    if lstm_model is not None:
        X_3d = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        recon = lstm_model.predict(X_3d, verbose=0)
        lstm_errors = np.mean((X_3d - recon) ** 2, axis=(1, 2))
    else:
        lstm_errors = np.zeros(len(X_scaled))
    
    # Normalise LSTM errors to [0, 1]
    lstm_min, lstm_max = lstm_errors.min(), lstm_errors.max()
    lstm_scores = (lstm_errors - lstm_min) / (lstm_max - lstm_min + 1e-10)
    lstm_preds  = (lstm_scores > 0.5).astype(int)
    
    # ── XGBoost Multi-Class ───────────────────────────────────────────────────
    xgb_proba     = xgb_model.predict_proba(X_scaled)      # shape: (n, 9)
    xgb_classes   = xgb_model.predict(X_scaled)             # shape: (n,)
    xgb_fault_prob = 1 - xgb_proba[:, 0]                   # prob of any fault
    xgb_preds     = (xgb_fault_prob > 0.5).astype(int)
    
    # ── Ensemble ──────────────────────────────────────────────────────────────
    votes          = if_preds + lstm_preds + xgb_preds
    ensemble_score = (if_scores + lstm_scores + xgb_fault_prob) / 3
    ensemble_preds = (votes >= 2).astype(int)
    
    # ── Health Index (rolling degradation) ───────────────────────────────────
    health_idx = pd.Series(ensemble_score).rolling(12, min_periods=1).mean().values
    
    # ── Build output DataFrame ────────────────────────────────────────────────
    out = pd.DataFrame({
        "timestamp":        feat_df["timestamp"].values,
        "true_label":       feat_df["true_label"].values,
        "fault_class":      feat_df["fault_class"].values,
        "is_synthetic":     feat_df.get("is_synthetic", pd.Series([False]*len(feat_df))).values,
        "ensemble_pred":    ensemble_preds,
        "ensemble_score":   ensemble_score,
        "votes":            votes,
        "health_index":     health_idx,
        "if_pred":          if_preds,
        "if_score":         if_scores,
        "lstm_pred":        lstm_preds,
        "lstm_score":       lstm_scores,
        "lstm_recon_error": lstm_errors,
        "xgb_pred":         xgb_preds,
        "xgb_score":        xgb_fault_prob,
        "xgb_fault_class":  xgb_classes,
        "fault_class_name": [fault_class_names.get(int(c), f"Class {c}") for c in xgb_classes],
    })
    
    # Add per-class probabilities
    for i in range(9):
        out[f"xgb_prob_class_{i}"] = xgb_proba[:, i] if i < xgb_proba.shape[1] else 0.0
    
    return out


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Predict Next Fault Onsets
# ═════════════════════════════════════════════════════════════════════════════

def predict_next_fault_onsets(
    feat_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    xgb_model,
    scaler,
    feature_cols: list,
    fault_class_names: dict,
) -> dict:
    """
    Predict WHEN each fault class will next appear.

    Strategy (in priority order):
    1. If augmented data exists → use scheduled onset dates as BASE timeline
       then adjust forward/backward based on current probability signal
    2. If probability trend is rising → extrapolate to critical threshold
    3. If probability is low + stable → assign physics-based default intervals
       derived from typical MTBF for each fault type on BB5 pumps
    4. Class 1 (real fault) → confirmed onset from real data

    This ensures EVERY class gets a predicted onset date.
    """
    log.info("Predicting next fault onsets for all 9 classes...")

    last_timestamp   = preds_df["timestamp"].max()
    fault_onset_real = pd.Timestamp(FAULT_ONSET_DATE)

    # ── Physics-based default MTBF intervals (days) per fault class ───────────
    # Based on Sulzer BB5 / API 610 maintenance guidelines
    # These are "expected time to next occurrence" for a healthy pump
    DEFAULT_ONSET_DAYS = {
        0: None,   # Normal — always active
        1: None,   # Real fault — confirmed date
        2: 120,    # Bearing Wear — 4 months (bearing life cycle)
        3: 90,     # Shaft Imbalance — 3 months (periodic balance check)
        4: 60,     # Misalignment — 2 months (thermal growth cycles)
        5: 45,     # Cavitation — 1.5 months (seasonal flow changes)
        6: 150,    # Seal Degradation — 5 months (seal life)
        7: 180,    # Gearbox Gear Wear — 6 months (gear inspection interval)
        8: 200,    # Motor Bearing Fault — ~6.5 months (motor bearing life)
    }

    # ── Try to load augmented data scheduled onset dates ──────────────────────
    augmented_schedule = {}
    augmented_path = DATA_DIR / "Augmented_DCS_Data.xlsx"

    if augmented_path.exists():
        try:
            aug_df = pd.read_excel(
                augmented_path, sheet_name="Augmented_Data"
            )
            aug_df[DATE_COLUMN] = pd.to_datetime(aug_df[DATE_COLUMN])

            for fc in range(9):
                fc_data = aug_df[aug_df["fault_class"] == fc]
                if len(fc_data) > 0:
                    augmented_schedule[fc] = {
                        "onset": fc_data[DATE_COLUMN].min(),
                        "end":   fc_data[DATE_COLUMN].max(),
                    }
            log.info(
                f"Augmented schedule loaded for "
                f"{len(augmented_schedule)} classes"
            )
        except Exception as e:
            log.warning(f"Could not load augmented schedule: {e}")

    onset_predictions = {}

    for fc in range(9):
        prob_col = f"xgb_prob_class_{fc}"

        # Get probability series (last 50 windows for trend, full for avg)
        if prob_col in preds_df.columns:
            all_probs    = preds_df[prob_col].values
            recent_probs = preds_df[prob_col].tail(50).values
        else:
            all_probs    = np.zeros(len(preds_df))
            recent_probs = np.zeros(50)

        current_prob = float(recent_probs[-1]) if len(recent_probs) > 0 else 0.0
        avg_prob_10  = float(np.mean(recent_probs[-10:])) if len(recent_probs) >= 10 else current_prob
        avg_prob_50  = float(np.mean(recent_probs))

        # Calculate trend slope over last 50 windows
        if len(recent_probs) >= 5:
            x_idx        = np.arange(len(recent_probs))
            slope, inter = np.polyfit(x_idx, recent_probs, 1)
        else:
            slope, inter = 0.0, current_prob

        # ── CLASS 0: Normal ────────────────────────────────────────────────────
        if fc == 0:
            onset_predictions[fc] = {
                "fault_class":      fc,
                "fault_name":       fault_class_names.get(fc, "Normal"),
                "status":           "Active (Normal Operation)",
                "current_prob":     current_prob,
                "avg_prob_recent":  avg_prob_10,
                "trend_slope":      float(slope),
                "predicted_onset":  None,
                "days_until_onset": None,
                "confidence":       "High",
                "risk_level":       "None",
                "recommendation":   "Continue normal monitoring",
                "is_real_fault":    True,
            }
            continue

        # ── CLASS 1: Real confirmed fault ──────────────────────────────────────
        if fc == 1:
            days_since = (fault_onset_real - last_timestamp).days

            # Determine risk based on current probability
            if current_prob > 0.5:
                risk = "Critical"
            elif current_prob > 0.2:
                risk = "High"
            elif current_prob > 0.1:
                risk = "Medium"
            else:
                risk = "Low"

            onset_predictions[fc] = {
                "fault_class":      fc,
                "fault_name":       fault_class_names.get(fc, "Thrust Bearing NDE-1"),
                "status":           (
                    "Active" if current_prob > 0.3
                    else "Historical (Confirmed)"
                ),
                "current_prob":     current_prob,
                "avg_prob_recent":  avg_prob_10,
                "trend_slope":      float(slope),
                "predicted_onset":  fault_onset_real.isoformat(),
                "days_until_onset": float(days_since),
                "confidence":       "Confirmed (Real Data)",
                "risk_level":       risk,
                "recommendation":   "Immediate bearing inspection/replacement",
                "is_real_fault":    True,
            }
            continue

        # ── CLASSES 2-8: Synthetic fault classes ───────────────────────────────
        # Priority 1: Rising probability trend → extrapolate to threshold
        # Priority 2: Augmented data scheduled onset → use as base + signal adjust
        # Priority 3: Physics-based MTBF default interval
        # All three paths ALWAYS produce a predicted_onset date.

        # ── Determine risk level ──────────────────────────────────────────────
        if avg_prob_10 > 0.4:
            risk_level = "Critical"
        elif avg_prob_10 > 0.2:
            risk_level = "High"
        elif avg_prob_10 > 0.1:
            risk_level = "Medium"
        elif slope > 0.001:
            risk_level = "Low-Developing"
        else:
            risk_level = "Low"

        # ── Priority 1: Extrapolate rising trend ──────────────────────────────
        days_from_trend = None
        if slope > 0 and current_prob < 0.7:
            # Windows needed to reach critical threshold (0.7)
            prob_gap        = 0.7 - current_prob
            windows_needed  = prob_gap / slope
            days_from_trend = windows_needed * 12.0 / 24.0  # 12h windows → days
            days_from_trend = min(float(days_from_trend), 730.0)  # cap 2 years

        elif slope > 0 and current_prob >= 0.7:
            # Already above threshold → imminent
            days_from_trend = 0.0

        # ── Priority 2: Augmented schedule base date ──────────────────────────
        days_from_schedule = None
        if fc in augmented_schedule:
            scheduled_onset   = augmented_schedule[fc]["onset"]
            raw_days          = (scheduled_onset - last_timestamp).days

            # Adjust schedule based on current probability signal:
            # High probability → fault coming sooner than scheduled
            # Low probability  → fault likely later than scheduled
            if avg_prob_10 > 0.3:
                # Compress timeline by up to 50%
                adjustment = -abs(raw_days) * 0.5
            elif avg_prob_10 > 0.1:
                # Slight compression
                adjustment = -abs(raw_days) * 0.2
            elif avg_prob_10 < 0.02 and raw_days < 0:
                # Very low prob + scheduled date already past
                # → push to default MTBF from now
                adjustment = abs(raw_days) + DEFAULT_ONSET_DAYS.get(fc, 180)
            else:
                adjustment = 0.0

            days_from_schedule = float(raw_days + adjustment)

        # ── Priority 3: Physics-based MTBF default ────────────────────────────
        days_from_default = float(DEFAULT_ONSET_DAYS.get(fc, 180))

        # ── Merge: choose best estimate ───────────────────────────────────────
        # Rule: always produce a date; prefer trend > schedule > default
        if days_from_trend is not None and days_from_trend >= 0:
            # Rising trend gives clearest signal
            final_days = days_from_trend
            method     = "Trend Extrapolation"
            confidence = (
                "High"   if avg_prob_10 > 0.3 else
                "Medium" if slope > 0.005       else
                "Low"
            )
        elif (
            days_from_schedule is not None and
            days_from_schedule > 0
        ):
            # Use adjusted augmented schedule
            final_days = days_from_schedule
            method     = "Augmented Schedule (Adjusted)"
            confidence = (
                "Medium" if avg_prob_10 > 0.05 else
                "Low (schedule-based)"
            )
        elif (
            days_from_schedule is not None and
            days_from_schedule <= 0
        ):
            # Schedule date has passed → use default MTBF from now
            final_days = days_from_default
            method     = "MTBF Default (Schedule Elapsed)"
            confidence = "Low (MTBF estimate)"
        else:
            # No augmented data → pure MTBF default
            final_days = days_from_default
            method     = "MTBF Default (No Schedule)"
            confidence = "Low (MTBF estimate)"

        # Always ensure onset is in the future
        final_days       = max(float(final_days), 1.0)
        predicted_onset  = last_timestamp + pd.Timedelta(days=final_days)

        # ── Status string ─────────────────────────────────────────────────────
        if final_days <= 7:
            status = f"⚠ Imminent (~{final_days:.0f} days)"
        elif final_days <= 30:
            status = f"Predicted in ~{final_days:.0f} days (this month)"
        elif final_days <= 90:
            status = f"Predicted in ~{final_days:.0f} days (~{final_days/30:.1f} months)"
        else:
            status = f"Predicted in ~{final_days:.0f} days (~{final_days/30:.1f} months)"

        recommendation = _get_recommendation(fc, risk_level, fault_class_names)

        onset_predictions[fc] = {
            "fault_class":      fc,
            "fault_name":       fault_class_names.get(fc, f"Class {fc}"),
            "status":           status,
            "current_prob":     current_prob,
            "avg_prob_recent":  avg_prob_10,
            "trend_slope":      float(slope),
            "predicted_onset":  predicted_onset.isoformat(),
            "days_until_onset": final_days,
            "confidence":       confidence,
            "risk_level":       risk_level,
            "recommendation":   recommendation,
            "is_real_fault":    False,
            "prediction_method": method,
            # Extra debug info
            "days_from_trend":    days_from_trend,
            "days_from_schedule": days_from_schedule,
            "days_from_default":  days_from_default,
        }

    log.info("=" * 55)
    log.info("NEXT FAULT ONSET PREDICTIONS:")
    log.info(f"{'Class':<8} {'Fault':<28} {'Risk':<16} "
             f"{'Prob':>6} {'Days':>8} {'Method'}")
    log.info("-" * 90)
    for fc, pred in onset_predictions.items():
        days_str = (
            f"{pred['days_until_onset']:>8.0f}"
            if pred["days_until_onset"] is not None
            else "      N/A"
        )
        log.info(
            f"{fc:<8} "
            f"{pred['fault_name'][:27]:<28} "
            f"{pred.get('risk_level','?'):<16} "
            f"{pred['current_prob']:>6.1%} "
            f"{days_str} "
            f"{pred.get('prediction_method','')}"
        )
    log.info("=" * 55)

    return onset_predictions


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Evaluate Models
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_models(preds_df: pd.DataFrame) -> pd.DataFrame:
    """Compute evaluation metrics for all models."""
    from sklearn.metrics import (
        accuracy_score, mean_squared_error,
        recall_score, precision_score, f1_score,
        roc_auc_score, confusion_matrix,
    )
    
    y_true = preds_df["true_label"].values
    
    rows = {}
    models = {
        "Isolation Forest": ("if_pred",       "if_score"),
        "LSTM Autoencoder": ("lstm_pred",      "lstm_score"),
        "XGBoost":          ("xgb_pred",       "xgb_score"),
        "Ensemble":         ("ensemble_pred",  "ensemble_score"),
    }
    
    for name, (pred_col, score_col) in models.items():
        if pred_col not in preds_df.columns:
            continue
        
        y_pred = preds_df[pred_col].values
        y_score = preds_df[score_col].values if score_col in preds_df.columns else y_pred
        
        try:
            cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            rows[name] = {
                "accuracy":            accuracy_score(y_true, y_pred),
                "mse":                 mean_squared_error(y_true, y_pred),
                "recall":              recall_score(y_true, y_pred, zero_division=0),
                "precision":           precision_score(y_true, y_pred, zero_division=0),
                "f1":                  f1_score(y_true, y_pred, zero_division=0),
                "roc_auc":             roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5,
                "false_positive_rate": fp / (fp + tn + 1e-10),
                "TP": int(tp), "FP": int(fp),
                "TN": int(tn), "FN": int(fn),
            }
        except Exception as e:
            log.warning(f"Could not evaluate {name}: {e}")
    
    return pd.DataFrame(rows).T


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Compute RUL
# ═════════════════════════════════════════════════════════════════════════════

def compute_rul(preds_df: pd.DataFrame) -> dict:
    """Estimate Remaining Useful Life from LSTM reconstruction error trend."""
    
    fault_mask = preds_df["true_label"] == 1
    
    if not fault_mask.any():
        return {"message": "No fault period detected", "rul_hours": None}
    
    fault_errors = preds_df[fault_mask]["lstm_recon_error"].values
    
    if len(fault_errors) < 3:
        return {"message": "Insufficient fault data for RUL", "rul_hours": None}
    
    normal_errors = preds_df[~fault_mask]["lstm_recon_error"].values
    baseline      = np.percentile(normal_errors, 95) if len(normal_errors) > 0 else fault_errors[0]
    threshold     = baseline * 2.0
    
    x_idx      = np.arange(len(fault_errors))
    slope, intercept = np.polyfit(x_idx, fault_errors, 1)
    
    current_error = fault_errors[-1]
    
    if slope > 0 and current_error < threshold:
        windows_to_threshold = (threshold - current_error) / slope
        rul_hours            = windows_to_threshold * 12  # 12h per window
    elif slope > 0:
        rul_hours = 0  # Already past threshold
    else:
        rul_hours = -1  # Stable/improving
    
    return {
        "rul_hours":           float(rul_hours) if rul_hours != -1 else None,
        "rul_windows":         int(windows_to_threshold) if slope > 0 and current_error < threshold else 0,
        "trend_slope":         float(slope),
        "current_error":       float(current_error),
        "critical_threshold":  float(threshold),
        "baseline_error":      float(baseline),
        "message":             (
            f"Estimated {rul_hours:.0f}h before critical threshold" if rul_hours and rul_hours > 0
            else "Error above critical threshold — immediate action required"
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — Compute Sensor Summary
# ═════════════════════════════════════════════════════════════════════════════

def compute_sensor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sensor statistics split by normal vs fault phase."""
    fault_onset = pd.Timestamp(FAULT_ONSET_DATE)
    rows = []
    
    for s in ALL_SENSORS:
        if s not in df.columns:
            continue
        norm_data  = df[df[DATE_COLUMN] < fault_onset][s].dropna()
        fault_data = df[df[DATE_COLUMN] >= fault_onset][s].dropna()
        
        rows.append({
            "Sensor":       s,
            "Normal Mean":  norm_data.mean()  if len(norm_data) > 0  else np.nan,
            "Normal Std":   norm_data.std()   if len(norm_data) > 0  else np.nan,
            "Fault Mean":   fault_data.mean() if len(fault_data) > 0 else np.nan,
            "Fault Std":    fault_data.std()  if len(fault_data) > 0 else np.nan,
            "Max Value":    df[s].max(),
            "Missing (%)":  df[s].isnull().mean() * 100,
        })
    
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 65)
    log.info("HP WIP-A Predictive Maintenance — Training Pipeline")
    log.info("AUGMENTED DATA: Real + Synthetic (9 Fault Classes)")
    log.info("=" * 65)
    
    # ── Load data ─────────────────────────────────────────────────────────────
    df, use_augmented = load_training_data()
    
    # ── Feature engineering ───────────────────────────────────────────────────
    feat_df = engineer_features(df)
    
    # ── Define feature/target columns ─────────────────────────────────────────
    meta_cols = [
        "timestamp", "true_label", "fault_class", "is_synthetic"
    ]
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]
    
    X = feat_df[feature_cols].values
    y = feat_df["true_label"].values
    y_class = feat_df["fault_class"].values
    
    log.info(f"Feature matrix: {X.shape[0]:,} windows × {X.shape[1]} features")
    log.info(f"Fault class distribution in windows:\n"
             f"{pd.Series(y_class).value_counts().sort_index().to_dict()}")
    
    # ── Train/test split (chronological) ──────────────────────────────────────
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    y_class_train   = y_class[:split_idx]
    
    # ── Scale features ─────────────────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_all_scaled   = scaler.transform(X)
    
    # ── Train models ──────────────────────────────────────────────────────────
    # Train only on normal windows for IF and LSTM (anomaly detection)
    normal_mask     = y_class_train == 0
    X_normal_scaled = X_train_scaled[normal_mask]
    
    if_model   = train_isolation_forest(X_normal_scaled)
    lstm_model = train_lstm_autoencoder(X_normal_scaled, input_dim=X_normal_scaled.shape[1])
    xgb_model  = train_xgboost_multiclass(
        X_train_scaled, y_train, y_class_train,
        use_smote=use_augmented,
    )
    
    # ── Fault class names ──────────────────────────────────────────────────────
    fault_class_names = {
        0: "Normal",
        1: "Thrust Bearing NDE-1",
        2: "Bearing Wear",
        3: "Shaft Imbalance",
        4: "Misalignment",
        5: "Cavitation",
        6: "Seal Degradation",
        7: "Gearbox Gear Wear",
        8: "Motor Bearing Fault",
    }
    
    # ── Generate predictions ───────────────────────────────────────────────────
    preds_df = generate_predictions(
        feat_df=feat_df,
        if_model=if_model,
        lstm_model=lstm_model,
        xgb_model=xgb_model,
        scaler=scaler,
        feature_cols=feature_cols,
        fault_class_names=fault_class_names,
        use_augmented=use_augmented,
    )
    
    # ── Predict next fault onsets ──────────────────────────────────────────────
    fault_onset_predictions = predict_next_fault_onsets(
        feat_df=feat_df,
        preds_df=preds_df,
        xgb_model=xgb_model,
        scaler=scaler,
        feature_cols=feature_cols,
        fault_class_names=fault_class_names,
    )
    
    # ── Evaluate ───────────────────────────────────────────────────────────────
    metrics_df  = evaluate_models(preds_df)
    rul_info    = compute_rul(preds_df)
    sensor_summ = compute_sensor_summary(df)
    fi_df       = pd.DataFrame({
        "feature":    feature_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    # ── Log target achievement ─────────────────────────────────────────────────
    if "Ensemble" in metrics_df.index:
        ens = metrics_df.loc["Ensemble"]
        log.info("─" * 50)
        log.info("ENSEMBLE PERFORMANCE:")
        for metric in ["recall", "precision", "f1", "roc_auc", "false_positive_rate"]:
            if metric in metrics_df.columns:
                log.info(f"  {metric:25s}: {float(ens[metric]):.4f}")
        log.info("─" * 50)
    
    # ── Log fault onset predictions ───────────────────────────────────────────
    log.info("NEXT FAULT ONSET PREDICTIONS:")
    for fc, pred in fault_onset_predictions.items():
        log.info(
            f"  Class {fc} ({pred['fault_name'][:25]:25s}): "
            f"Risk={pred['risk_level']:15s} | "
            f"Prob={pred['current_prob']:.3f} | "
            f"Onset={pred.get('predicted_onset', 'N/A')}"
        )
    
    # ── Save outputs ───────────────────────────────────────────────────────────
    import joblib
    
    preds_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "evaluation_metrics.csv")
    fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    sensor_summ.to_csv(OUTPUT_DIR / "sensor_summary.csv", index=False)
    
    with open(OUTPUT_DIR / "rul_estimate.json", "w") as f:
        json.dump(rul_info, f, indent=2)
    
    with open(OUTPUT_DIR / "fault_onset_predictions.json", "w") as f:
        json.dump(fault_onset_predictions, f, indent=2, default=str)
    
    with open(MODELS_DIR / "fault_class_names.json", "w") as f:
        json.dump({str(k): v for k, v in fault_class_names.items()}, f, indent=2)
    
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    
    joblib.dump(if_model,  MODELS_DIR / "isolation_forest.pkl")
    joblib.dump(scaler,    MODELS_DIR / "scaler.pkl")
    joblib.dump(xgb_model, MODELS_DIR / "xgboost_multiclass.pkl")
    
    if lstm_model is not None:
        lstm_model.save(MODELS_DIR / "lstm_autoencoder.h5")
    
    log.info("=" * 65)
    log.info(f"✓ All outputs saved to {OUTPUT_DIR}")
    log.info(f"✓ Models saved to {MODELS_DIR}")
    log.info("✓ Training complete!")
    log.info("=" * 65)


if __name__ == "__main__":
    main()


