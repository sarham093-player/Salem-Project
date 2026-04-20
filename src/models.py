"""
models.py
---------
Three ML models for the predictive maintenance ensemble.

ENHANCED VERSION with improved logging for debugging.
"""

import numpy as np
import pandas as pd
import joblib
import logging
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    recall_score, precision_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from config import (
    RANDOM_SEED, IF_CONTAMINATION, IF_N_ESTIMATORS, IF_MAX_SAMPLES,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LEARNING_RATE, LSTM_LATENT_DIM,
    LSTM_THRESHOLD_PERCENTILE,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE_BYTREE,
    SMOTE_RANDOM_STATE, MODELS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Model 1 — Isolation Forest
# ═════════════════════════════════════════════════════════════════════════════

class IsolationForestModel:
    """Unsupervised anomaly detection."""

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION,
            max_samples=IF_MAX_SAMPLES,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X_normal_flat: np.ndarray) -> "IsolationForestModel":
        logger.info(f"Fitting Isolation Forest on {len(X_normal_flat):,} normal windows...")
        self.model.fit(X_normal_flat)
        self._fitted = True
        logger.info("Isolation Forest fitted.")
        return self

    def score(self, X_flat: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        raw = self.model.decision_function(X_flat)
        normalised = 1.0 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-10)
        return normalised.astype(np.float32)

    def predict(self, X_flat: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        raw = self.model.predict(X_flat)
        return ((raw == -1).astype(np.int32))

    def save(self, path: Path = MODELS_DIR / "isolation_forest.pkl"):
        joblib.dump(self.model, path)
        logger.info(f"Isolation Forest saved → {path}")

    def load(self, path: Path = MODELS_DIR / "isolation_forest.pkl"):
        self.model = joblib.load(path)
        self._fitted = True
        logger.info(f"Isolation Forest loaded ← {path}")
        return self


# ═════════════════════════════════════════════════════════════════════════════
# Model 2 — LSTM Autoencoder (ENHANCED LOGGING)
# ═════════════════════════════════════════════════════════════════════════════

class LSTMAutoencoder:
    """LSTM Autoencoder for temporal pattern learning."""

    def __init__(self, window_size: int, n_features: int):
        self.window_size = window_size
        self.n_features  = n_features
        self.model       = None
        self.threshold   = None
        self._build()

    def _build(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import (
                Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
            )
            from tensorflow.keras.optimizers import Adam

            tf.random.set_seed(RANDOM_SEED)

            inp = Input(shape=(self.window_size, self.n_features))

            # Encoder
            x = LSTM(64, return_sequences=True, name="enc_lstm1")(inp)
            x = Dropout(0.1)(x)
            x = LSTM(32, return_sequences=False, name="enc_lstm2")(x)
            x = Dense(LSTM_LATENT_DIM, activation="relu", name="latent")(x)

            # Decoder
            x = RepeatVector(self.window_size)(x)
            x = LSTM(32, return_sequences=True, name="dec_lstm1")(x)
            x = Dropout(0.1)(x)
            x = LSTM(64, return_sequences=True, name="dec_lstm2")(x)
            out = TimeDistributed(Dense(self.n_features), name="reconstruction")(x)

            self.model = Model(inputs=inp, outputs=out, name="LSTM_Autoencoder")
            self.model.compile(
                optimizer=Adam(learning_rate=LSTM_LEARNING_RATE),
                loss="mse"
            )
            logger.info(f"LSTM Autoencoder built — "
                        f"Input: ({self.window_size}, {self.n_features}), "
                        f"Latent dim: {LSTM_LATENT_DIM}")
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise

    def fit(self, X_normal: np.ndarray, X_val: np.ndarray = None) -> "LSTMAutoencoder":
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        # ✅ VALIDATE INPUT DATA
        logger.info(f"\n{'='*60}")
        logger.info("LSTM TRAINING DATA VALIDATION")
        logger.info(f"{'='*60}")
        logger.info(f"X_normal shape: {X_normal.shape}")
        logger.info(f"X_normal min:   {X_normal.min():.6f}")
        logger.info(f"X_normal max:   {X_normal.max():.6f}")
        logger.info(f"X_normal mean:  {X_normal.mean():.6f}")
        
        if X_normal.min() < -0.1 or X_normal.max() > 1.1:
            logger.error("❌ CRITICAL ERROR: Training data NOT normalized!")
            logger.error(f"   Expected: [0, 1]")
            logger.error(f"   Actual:   [{X_normal.min():.4f}, {X_normal.max():.4f}]")
            raise ValueError("Cannot train LSTM on non-normalized data")
        
        logger.info("✅ Training data validated")
        logger.info(f"{'='*60}\n")

        logger.info(f"Training LSTM Autoencoder on {len(X_normal):,} normal windows...")

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]

        validation_data = (X_val, X_val) if X_val is not None else None

        history = self.model.fit(
            X_normal, X_normal,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,  # ✅ Show training progress
        )

        # ✅ LOG TRAINING RESULTS
        final_loss = history.history['loss'][-1]
        logger.info(f"Final training loss: {final_loss:.6f}")
        
        if validation_data is not None:
            final_val_loss = history.history['val_loss'][-1]
            logger.info(f"Final validation loss: {final_val_loss:.6f}")

        # Compute reconstruction errors on normal windows to set threshold
        recon = self.model.predict(X_normal, verbose=0)
        mse_normal = np.mean((X_normal - recon) ** 2, axis=(1, 2))
        
        # ✅ LOG ERROR DISTRIBUTION
        logger.info(f"\nReconstruction error statistics (normal windows):")
        logger.info(f"  Min:    {mse_normal.min():.6f}")
        logger.info(f"  Max:    {mse_normal.max():.6f}")
        logger.info(f"  Mean:   {mse_normal.mean():.6f}")
        logger.info(f"  Median: {np.median(mse_normal):.6f}")
        logger.info(f"  Std:    {mse_normal.std():.6f}")
        
        self.threshold = float(np.percentile(mse_normal, LSTM_THRESHOLD_PERCENTILE))
        
        logger.info(f"\nLSTM Autoencoder trained. Threshold set at "
                    f"{LSTM_THRESHOLD_PERCENTILE}th percentile = {self.threshold:.6f}")
        
        # ✅ VALIDATE THRESHOLD
        if self.threshold > 1.0:
            logger.error(f"❌ WARNING: Threshold very high ({self.threshold:.4f})!")
            logger.error("   This suggests training data was NOT normalized")
            logger.error("   Expected threshold: 0.001 - 0.01")
            logger.error("   Dashboard will show errors 30-50 instead of 0.001-0.01")
        elif self.threshold < 0.0001:
            logger.warning(f"⚠️  Threshold very low ({self.threshold:.6f})")
            logger.warning("   Model may be too sensitive (high false positive rate)")
        else:
            logger.info(f"✅ Threshold in expected range (0.001 - 0.01)")
        
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Return per-window MSE reconstruction error."""
        if self.model is None:
            raise RuntimeError("Model not built.")
        recon = self.model.predict(X, verbose=0)
        errors = np.mean((X - recon) ** 2, axis=(1, 2)).astype(np.float32)
        
        # ✅ LOG WARNING IF ERRORS ARE HUGE
        if errors.mean() > 1.0:
            logger.warning(f"⚠️  Reconstruction errors very high (mean={errors.mean():.4f})!")
            logger.warning("   This suggests input data is NOT normalized")
            logger.warning("   Dashboard will show incorrect LSTM Recon Error values")
        
        return errors

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return normalised anomaly scores [0, 1]."""
        errors = self.reconstruction_error(X)
        return np.clip(errors / (self.threshold + 1e-10), 0, 1).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary prediction: 1 = anomaly, 0 = normal."""
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Train model first.")
        errors = self.reconstruction_error(X)
        return (errors > self.threshold).astype(np.int32)

    def estimate_rul(self, error_series: np.ndarray, n_future: int = 20) -> dict:
        """Estimate Remaining Useful Life from reconstruction error trend."""
        if self.threshold is None:
            return {"rul_windows": None, "trend_slope": None}

        x = np.arange(len(error_series))
        slope, intercept = np.polyfit(x, error_series, 1)

        if slope <= 0:
            return {"rul_windows": None, "trend_slope": float(slope), "message": "No degradation trend"}

        critical = 2.0 * self.threshold
        current_error = error_series[-1]

        if current_error >= critical:
            rul_windows = 0
        else:
            rul_windows = int((critical - current_error) / slope)

        return {
            "rul_windows":        rul_windows,
            "trend_slope":        float(slope),
            "current_error":      float(current_error),
            "critical_threshold": float(critical),
            "rul_hours":          rul_windows * (48 * 15 / 60),
            "message":            f"Estimated RUL: {rul_windows} windows ({rul_windows * 12:.0f} hours)"
        }

    def save(self, dir_path: Path = MODELS_DIR):
        dir_path.mkdir(parents=True, exist_ok=True)
        self.model.save(dir_path / "lstm_autoencoder.keras")
        joblib.dump(self.threshold, dir_path / "lstm_threshold.pkl")
        logger.info(f"LSTM Autoencoder saved → {dir_path}")

    def load(self, dir_path: Path = MODELS_DIR):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(dir_path / "lstm_autoencoder.keras")
        self.threshold = joblib.load(dir_path / "lstm_threshold.pkl")
        logger.info(f"LSTM Autoencoder loaded ← {dir_path}")
        logger.info(f"Loaded threshold: {self.threshold:.6f}")
        return self


# ═════════════════════════════════════════════════════════════════════════════
# Model 3 — XGBoost (UNCHANGED)
# ═════════════════════════════════════════════════════════════════════════════

class XGBoostClassifier:
    """Supervised MULTI-CLASS classifier."""

    def __init__(self):
        self.model        = None
        self._fitted      = False
        self.feature_names: list = []
        self._n_classes   = 2
        self._is_multiclass = False

    def _build_model(self, n_classes: int):
        if n_classes == 2:
            self.model = xgb.XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE_BYTREE,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            self._is_multiclass = False
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE_BYTREE,
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=n_classes,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            self._is_multiclass = True
        self._n_classes = n_classes

    def fit(self, X_flat: np.ndarray, y: np.ndarray,
            feature_names: list = None) -> "XGBoostClassifier":
        self.feature_names = feature_names or [f"f{i}" for i in range(X_flat.shape[1])]

        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        self._build_model(n_classes)

        logger.info(f"XGBoost mode: {'multi-class (' + str(n_classes) + ' classes)' if n_classes > 2 else 'binary'}")
        logger.info(f"Class distribution before SMOTE: {dict(zip(*np.unique(y, return_counts=True)))}")

        min_class_count = min(np.bincount(y[y >= 0]))
        k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1

        try:
            smote = SMOTE(random_state=SMOTE_RANDOM_STATE, k_neighbors=k_neighbors)
            X_res, y_res = smote.fit_resample(X_flat, y)
            logger.info(f"After SMOTE: {dict(zip(*np.unique(y_res, return_counts=True)))}")
        except Exception as e:
            logger.warning(f"SMOTE failed ({e}). Training without oversampling.")
            X_res, y_res = X_flat, y

        logger.info(f"Training XGBoost on {len(X_res):,} windows ({n_classes} classes)...")
        self.model.fit(X_res, y_res, verbose=False)
        self._fitted = True
        logger.info("XGBoost training complete.")
        return self

    def predict(self, X_flat: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict(X_flat).astype(np.int32)

    def predict_binary(self, X_flat: np.ndarray) -> np.ndarray:
        preds = self.predict(X_flat)
        return (preds > 0).astype(np.int32)

    def score(self, X_flat: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        proba = self.model.predict_proba(X_flat)
        return (1.0 - proba[:, 0]).astype(np.float32)

    def predict_proba_all(self, X_flat: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return self.model.predict_proba(X_flat).astype(np.float32)

    def feature_importance_df(self) -> pd.DataFrame:
        importances = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path = MODELS_DIR / "xgboost.pkl"):
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "n_classes": self._n_classes,
            "is_multiclass": self._is_multiclass,
        }, path)
        logger.info(f"XGBoost saved → {path}")

    def load(self, path: Path = MODELS_DIR / "xgboost.pkl"):
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names   = data.get("feature_names", [])
        self._n_classes      = data.get("n_classes", 2)
        self._is_multiclass  = data.get("is_multiclass", False)
        self._fitted = True
        logger.info(f"XGBoost loaded ← {path}")
        return self


# ═════════════════════════════════════════════════════════════════════════════
# Ensemble & Evaluation (UNCHANGED)
# ═════════════════════════════════════════════════════════════════════════════

class PdMEnsemble:
    """2-of-3 voting ensemble."""

    def __init__(self, if_model, lstm_model, xgb_model):
        self.if_model   = if_model
        self.lstm_model = lstm_model
        self.xgb_model  = xgb_model

    def predict(self, X_windows: np.ndarray, X_flat: np.ndarray) -> dict:
        from config import ENSEMBLE_MIN_VOTES

        X_windows_flat = X_windows.reshape(len(X_windows), -1)

        if_preds  = self.if_model.predict(X_windows_flat)
        if_scores = self.if_model.score(X_windows_flat)

        lstm_preds  = self.lstm_model.predict(X_windows)
        lstm_scores = self.lstm_model.score(X_windows)

        xgb_class_preds = self.xgb_model.predict(X_flat)
        xgb_preds       = self.xgb_model.predict_binary(X_flat)
        xgb_scores      = self.xgb_model.score(X_flat)

        vote_matrix = np.stack([if_preds, lstm_preds, xgb_preds], axis=1)
        total_votes = vote_matrix.sum(axis=1)
        ensemble_preds = (total_votes >= ENSEMBLE_MIN_VOTES).astype(np.int32)

        ensemble_scores = (if_scores + lstm_scores + xgb_scores) / 3.0

        return {
            "ensemble_preds":   ensemble_preds,
            "ensemble_scores":  ensemble_scores,
            "if_preds":         if_preds,
            "if_scores":        if_scores,
            "lstm_preds":       lstm_preds,
            "lstm_scores":      lstm_scores,
            "xgb_class_preds":  xgb_class_preds,
            "xgb_preds":        xgb_preds,
            "xgb_scores":       xgb_scores,
            "vote_counts":      total_votes,
        }


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_score: np.ndarray = None, model_name: str = "") -> dict:
    from config import TARGET_RECALL, TARGET_PRECISION, TARGET_ROC_AUC, TARGET_MAX_FPR 

    y_pred_binary  = (y_pred  > 0).astype(np.int32)
    y_true_binary  = (y_true  > 0).astype(np.int32)

    metrics = {
        "model":     model_name,
        "recall":    round(recall_score(y_true_binary, y_pred_binary, zero_division=0), 4),
        "precision": round(precision_score(y_true_binary, y_pred_binary, zero_division=0), 4),
        "f1":        round(f1_score(y_true_binary, y_pred_binary, zero_division=0), 4),
    }

    if y_score is not None and len(np.unique(y_true_binary)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_true_binary, y_score), 4)
    else:
        metrics["roc_auc"] = None

    if len(np.unique(y_true_binary)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    else:
        tn = fp = fn = tp = 0

    metrics.update({
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
    })

    metrics["recall_ok"]    = metrics["recall"]    >= TARGET_RECALL
    metrics["precision_ok"] = metrics["precision"] >= TARGET_PRECISION
    metrics["roc_auc_ok"]   = (metrics["roc_auc"] or 0) >= TARGET_ROC_AUC
    metrics["fpr_ok"]       = metrics["false_positive_rate"] <= TARGET_MAX_FPR

    logger.info(f"\n{'='*50}\n{model_name} Evaluation")
    logger.info(f"Recall:    {metrics['recall']:.4f}  (target ≥{TARGET_RECALL}) {'✓' if metrics['recall_ok'] else '✗'}")
    logger.info(f"Precision: {metrics['precision']:.4f}  (target ≥{TARGET_PRECISION}) {'✓' if metrics['precision_ok'] else '✗'}")
    logger.info(f"ROC-AUC:   {metrics.get('roc_auc','N/A')}  (target ≥{TARGET_ROC_AUC})")
    logger.info(f"FPR:       {metrics['false_positive_rate']:.4f}  (target ≤{TARGET_MAX_FPR})")

    return metrics