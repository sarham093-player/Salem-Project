"""
train.py
--------
Full training pipeline for HP WIP-A Predictive Maintenance.

ENHANCED VERSION with data validation and emergency fixes.

Steps:
  1. Load and validate raw DCS data
  2. Run preprocessing pipeline (6 steps)
  3. Extract ~45 features per window
  4a. Train Isolation Forest on normal windows only
  4b. Train LSTM Autoencoder on normal windows only
  4c. Build augmented multi-class dataset (9 fault classes)
       → Train XGBoost multi-class classifier on augmented features
  5. Evaluate ensemble on held-out test set
  6. Compute RUL estimate from LSTM error trajectory
  7. Save all predictions and metrics to outputs/

Usage:
  cd pdm_project
  python src/train.py
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add src/ to path when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, OUTPUT_DIR, RANDOM_SEED
from data_loader import load_raw_data, validate_structure, summarise_data
from preprocessor import run_full_pipeline
from feature_engineering import extract_all_features, get_feature_names
from fault_augmentation import build_augmented_dataset, get_fault_class_names
from models import (
    IsolationForestModel, LSTMAutoencoder, XGBoostClassifier,
    PdMEnsemble, evaluate_model
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)


def validate_and_fix_data(X_train, y_train, X_val, y_val, X_test, y_test, X_all, y_all):
    """
    Validate data normalization and apply emergency fix if needed.
    
    Returns: Fixed X_train, X_val, X_test, X_all
    """
    logger.info("\n" + "="*70)
    logger.info("🔍 DATA VALIDATION & EMERGENCY FIX CHECK")
    logger.info("="*70)

    # Check 1: Data range (should be [0, 1])
    logger.info(f"\n1️⃣  DATA RANGE VERIFICATION:")
    logger.info(f"   X_train min:  {X_train.min():.6f}")
    logger.info(f"   X_train max:  {X_train.max():.6f}")
    logger.info(f"   X_train mean: {X_train.mean():.6f}")
    logger.info(f"   X_train std:  {X_train.std():.6f}")

    needs_fix = X_train.min() < -0.1 or X_train.max() > 1.1

    if needs_fix:
        logger.error(f"   ❌ ERROR: Data NOT normalized!")
        logger.error(f"      Expected: [0, 1]")
        logger.error(f"      Actual:   [{X_train.min():.2f}, {X_train.max():.2f}]")
        logger.error(f"\n   🔧 FIX: Applying emergency normalization...")
        
        from sklearn.preprocessing import MinMaxScaler
        
        # Fit on normal windows only
        normal_mask = y_train == 0
        X_train_normal = X_train[normal_mask]
        
        if len(X_train_normal) == 0:
            logger.error("      ❌ CRITICAL: No normal windows to fit scaler!")
            raise ValueError("Cannot normalize: no normal training windows")
        
        # Reshape for scaler
        n_train, n_time, n_feat = X_train.shape
        n_val = len(X_val)
        n_test = len(X_test)
        n_all = len(X_all)
        
        scaler_emergency = MinMaxScaler(feature_range=(0, 1))
        X_train_normal_2d = X_train_normal.reshape(-1, n_feat)
        scaler_emergency.fit(X_train_normal_2d)
        
        logger.info(f"      Emergency scaler fitted on {len(X_train_normal):,} normal windows")
        logger.info(f"      Scaler min (first 5): {scaler_emergency.data_min_[:5]}")
        logger.info(f"      Scaler max (first 5): {scaler_emergency.data_max_[:5]}")
        
        # Transform all
        X_train = scaler_emergency.transform(X_train.reshape(-1, n_feat)).reshape(n_train, n_time, n_feat)
        X_val = scaler_emergency.transform(X_val.reshape(-1, n_feat)).reshape(n_val, n_time, n_feat)
        X_test = scaler_emergency.transform(X_test.reshape(-1, n_feat)).reshape(n_test, n_time, n_feat)
        X_all = scaler_emergency.transform(X_all.reshape(-1, n_feat)).reshape(n_all, n_time, n_feat)
        
        logger.info(f"      ✅ Emergency normalization complete!")
        logger.info(f"         New range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        
        # Save emergency scaler
        import joblib
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler_emergency, MODELS_DIR / "scaler_emergency.pkl")
        logger.warning(f"      ⚠️  Emergency scaler saved → {MODELS_DIR / 'scaler_emergency.pkl'}")
        logger.warning(f"      ⚠️  Fix preprocessor.py to avoid this in future!")
    else:
        logger.info(f"   ✅ Data properly normalized!")

    # Check 2: NaN presence
    logger.info(f"\n2️⃣  NaN CHECK:")
    nan_train = np.isnan(X_train).sum()
    nan_val = np.isnan(X_val).sum()
    nan_test = np.isnan(X_test).sum()
    
    logger.info(f"   X_train NaNs: {nan_train}")
    logger.info(f"   X_val NaNs:   {nan_val}")
    logger.info(f"   X_test NaNs:  {nan_test}")

    if nan_train + nan_val + nan_test > 0:
        logger.error(f"   ❌ ERROR: NaNs present after preprocessing!")
        raise ValueError("NaN values detected in training data")
    else:
        logger.info(f"   ✅ No NaN values detected!")

    # Check 3: Label distribution
    logger.info(f"\n3️⃣  LABEL DISTRIBUTION:")
    logger.info(f"   Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logger.info(f"   Val:   {dict(zip(*np.unique(y_val, return_counts=True)))}")
    logger.info(f"   Test:  {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Validate we have both classes
    if len(np.unique(y_train)) < 2:
        logger.error(f"   ❌ ERROR: Training data has only one class!")
        raise ValueError("Insufficient class diversity in training data")

    # Check 4: Sample sensor values
    logger.info(f"\n4️⃣  SAMPLE SENSOR VALUES (first window, first timestep):")
    logger.info(f"   Expected: All sensors in [0, 1] after normalization")
    logger.info(f"   First 10 sensor values: {X_train[0, 0, :10]}")
    logger.info(f"   All in range [0,1]: {np.all((X_train[0, 0, :] >= 0) & (X_train[0, 0, :] <= 1))}")

    logger.info("="*70 + "\n")
    
    return X_train, X_val, X_test, X_all


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    logger.info("STEP 1 — Loading raw DCS data")
    raw_df, tags_df = load_raw_data()
    validate_structure(raw_df)

    summary = summarise_data(raw_df)
    summary.to_csv(OUTPUT_DIR / "sensor_summary.csv", index=False)
    logger.info(f"Sensor summary saved → {OUTPUT_DIR / 'sensor_summary.csv'}")

    # ── 2. Run preprocessing pipeline ────────────────────────────────────────
    logger.info("STEP 2 — Preprocessing pipeline")
    pipeline_output = run_full_pipeline(raw_df)

    X_train      = pipeline_output["X_train"]       # (n, 96, 35) normalised windows
    y_train      = pipeline_output["y_train"]        # binary labels (0/1)
    X_val        = pipeline_output["X_val"]
    y_val        = pipeline_output["y_val"]
    X_test       = pipeline_output["X_test"]
    y_test       = pipeline_output["y_test"]
    ts_test      = pipeline_output["ts_test"]
    X_all        = pipeline_output["X_all"]
    y_all        = pipeline_output["y_all"]
    ts_all       = pipeline_output["ts_all"]
    scaler       = pipeline_output["scaler"]
    df_processed = pipeline_output["df_processed"]

    logger.info(
        f"Windows — train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}"
    )
    logger.info(
        f"Label distribution train: {dict(zip(*np.unique(y_train, return_counts=True)))}"
    )

    # ✅ VALIDATE AND FIX DATA IF NEEDED
    X_train, X_val, X_test, X_all = validate_and_fix_data(
        X_train, y_train, X_val, y_val, X_test, y_test, X_all, y_all
    )

    # ── 3. Feature extraction (for original binary labels) ───────────────────
    logger.info("STEP 3 — Feature extraction (original dataset)")
    df_feat_all   = extract_all_features(X_all, y_all)
    feature_names = [c for c in df_feat_all.columns if c != "label"]

    n_train = len(X_train)
    n_val   = len(X_val)

    df_feat_train = df_feat_all.iloc[:n_train]
    df_feat_val   = df_feat_all.iloc[n_train:n_train + n_val]
    df_feat_test  = df_feat_all.iloc[n_train + n_val:]

    X_feat_train = df_feat_train[feature_names].values.astype(np.float32)
    X_feat_val   = df_feat_val[feature_names].values.astype(np.float32)
    X_feat_test  = df_feat_test[feature_names].values.astype(np.float32)

    logger.info(f"Features: {len(feature_names)}")
    logger.info(
        f"Feature shapes — train: {X_feat_train.shape} | "
        f"val: {X_feat_val.shape} | test: {X_feat_test.shape}"
    )

    # Save feature names
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # ── 4a. Train Isolation Forest ────────────────────────────────────────────
    logger.info("STEP 4a — Training Isolation Forest (normal windows only)")
    normal_mask_train    = y_train == 0
    X_train_normal_flat  = X_train[normal_mask_train].reshape(
        normal_mask_train.sum(), -1
    )

    if_model = IsolationForestModel()
    if_model.fit(X_train_normal_flat)
    if_model.save()

    # ── 4b. Train LSTM Autoencoder ────────────────────────────────────────────
    logger.info("STEP 4b — Training LSTM Autoencoder (normal windows only)")
    
    # ✅ FINAL VALIDATION BEFORE LSTM TRAINING
    logger.info("\n" + "="*60)
    logger.info("PRE-LSTM VALIDATION")
    logger.info("="*60)
    logger.info(f"Normal windows for LSTM: {normal_mask_train.sum()}")
    logger.info(f"Data range check: [{X_train[normal_mask_train].min():.6f}, "
                f"{X_train[normal_mask_train].max():.6f}]")
    
    if X_train[normal_mask_train].min() < -0.1 or X_train[normal_mask_train].max() > 1.1:
        logger.error("❌ CRITICAL: Data still not normalized before LSTM training!")
        raise ValueError("Cannot train LSTM on non-normalized data")
    
    logger.info("✅ Data validated for LSTM training")
    logger.info("="*60 + "\n")
    
    window_size = X_train.shape[1]
    n_sensors   = X_train.shape[2]

    lstm_model   = LSTMAutoencoder(window_size=window_size, n_features=n_sensors)
    X_val_normal = X_val[y_val == 0] if (y_val == 0).any() else X_val
    lstm_model.fit(X_train[normal_mask_train], X_val_normal)
    lstm_model.save()
    
    # ✅ LOG LSTM THRESHOLD (THIS IS CRITICAL!)
    logger.info("\n" + "="*60)
    logger.info("LSTM THRESHOLD CHECK")
    logger.info("="*60)
    logger.info(f"LSTM threshold: {lstm_model.threshold:.6f}")
    
    if lstm_model.threshold > 1.0:
        logger.error(f"❌ CRITICAL: LSTM threshold too high ({lstm_model.threshold:.4f})!")
        logger.error("   This indicates training data was NOT normalized!")
        logger.error("   Expected: 0.001 - 0.01")
        logger.error("   Dashboard will show errors in range 30-50 instead of 0.001-0.01")
        raise ValueError("LSTM threshold validation failed")
    else:
        logger.info(f"✅ LSTM threshold OK (expected range: 0.001 - 0.01)")
    
    # Test on sample windows
    sample_normal = X_train[normal_mask_train][0:1]
    sample_normal_error = lstm_model.reconstruction_error(sample_normal)[0]
    logger.info(f"Sample normal window error: {sample_normal_error:.6f}")
    
    if (y_train == 1).any():
        sample_fault = X_train[y_train == 1][0:1]
        sample_fault_error = lstm_model.reconstruction_error(sample_fault)[0]
        logger.info(f"Sample fault window error:  {sample_fault_error:.6f}")
        logger.info(f"Fault/Normal ratio: {sample_fault_error/sample_normal_error:.2f}x")
    
    logger.info("="*60 + "\n")

    # ── 4c. Data augmentation + XGBoost multi-class training ─────────────────
    logger.info("STEP 4c — Building augmented multi-class dataset")

    # Build augmented windows (real normal + real fault1 + synthetic faults 2-8)
    X_train_aug, y_train_aug = build_augmented_dataset(X_train, y_train)

    logger.info(f"Augmented training set: {X_train_aug.shape}")
    logger.info(
        f"Class distribution: {dict(zip(*np.unique(y_train_aug, return_counts=True)))}"
    )

    # Extract features from augmented windows
    logger.info("Extracting features from augmented windows...")
    df_feat_aug   = extract_all_features(X_train_aug, y_train_aug)
    X_feat_aug    = df_feat_aug[feature_names].values.astype(np.float32)
    y_feat_aug    = df_feat_aug["label"].values.astype(np.int32)

    logger.info(
        f"Augmented feature matrix: {X_feat_aug.shape} | "
        f"Labels: {np.unique(y_feat_aug)}"
    )

    # Train XGBoost multi-class
    logger.info("STEP 4c — Training XGBoost multi-class classifier")
    xgb_model = XGBoostClassifier()
    xgb_model.fit(X_feat_aug, y_feat_aug, feature_names=feature_names)
    xgb_model.save()

    # Feature importance
    fi_df = xgb_model.feature_importance_df()
    fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    logger.info(f"\nTop 10 features:\n{fi_df.head(10).to_string(index=False)}")

    # Save fault class names for dashboard
    fault_class_names = get_fault_class_names()
    with open(MODELS_DIR / "fault_class_names.json", "w") as f:
        json.dump({str(k): v for k, v in fault_class_names.items()}, f, indent=2)
    logger.info("Fault class names saved → models/fault_class_names.json")

    # ── 5. Ensemble evaluation on test set ───────────────────────────────────
    logger.info("STEP 5 — Ensemble evaluation on test set")
    ensemble = PdMEnsemble(if_model, lstm_model, xgb_model)

    results = ensemble.predict(X_windows=X_test, X_flat=X_feat_test)

    # Individual model metrics
    all_metrics = {}
    for name, preds, scores in [
        ("Isolation Forest", results["if_preds"],       results["if_scores"]),
        ("LSTM Autoencoder", results["lstm_preds"],      results["lstm_scores"]),
        ("XGBoost",          results["xgb_preds"],       results["xgb_scores"]),
        ("Ensemble",         results["ensemble_preds"],  results["ensemble_scores"]),
    ]:
        m = evaluate_model(y_test, preds, scores, model_name=name)
        all_metrics[name] = m

    # Detailed XGBoost multi-class report on test features
    xgb_class_preds_test = results["xgb_class_preds"]
    unique_test = np.unique(y_test)
    if len(unique_test) > 1:
        logger.info(
            "\nXGBoost multi-class distribution on test set:\n"
            + str(dict(zip(*np.unique(xgb_class_preds_test, return_counts=True))))
        )

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(OUTPUT_DIR / "evaluation_metrics.csv")
    logger.info(f"\nEvaluation saved → {OUTPUT_DIR / 'evaluation_metrics.csv'}")

    # ── 6. Compute RUL estimate ────────────────────────────────────────────────
    logger.info("STEP 6 — RUL Estimation")
    all_lstm_errors = lstm_model.reconstruction_error(X_all)
    
    # ✅ LOG LSTM ERROR STATISTICS
    logger.info("\n" + "="*60)
    logger.info("LSTM RECONSTRUCTION ERROR STATISTICS")
    logger.info("="*60)
    logger.info(f"All windows - Min:    {all_lstm_errors.min():.6f}")
    logger.info(f"All windows - Max:    {all_lstm_errors.max():.6f}")
    logger.info(f"All windows - Mean:   {all_lstm_errors.mean():.6f}")
    logger.info(f"All windows - Median: {np.median(all_lstm_errors):.6f}")
    
    normal_errors = all_lstm_errors[y_all == 0]
    fault_errors_all = all_lstm_errors[y_all == 1]
    
    if len(normal_errors) > 0:
        logger.info(f"Normal windows - Mean: {normal_errors.mean():.6f}")
    if len(fault_errors_all) > 0:
        logger.info(f"Fault windows - Mean:  {fault_errors_all.mean():.6f}")
        if len(normal_errors) > 0:
            logger.info(f"Fault/Normal ratio:    {fault_errors_all.mean() / normal_errors.mean():.2f}x")
    
    if all_lstm_errors.mean() > 1.0:
        logger.error("❌ WARNING: LSTM errors abnormally high!")
        logger.error("   This will appear as 30-50 in dashboard instead of 0.001-0.01")
        logger.error("   Root cause: Data normalization issue in preprocessing")
    else:
        logger.info("✅ LSTM errors in expected range")
    logger.info("="*60 + "\n")
    
    fault_mask = y_all == 1
    if fault_mask.any():
        fault_errors = all_lstm_errors[fault_mask]
        rul_info = lstm_model.estimate_rul(fault_errors)
        logger.info(f"RUL Estimate: {rul_info}")
        with open(OUTPUT_DIR / "rul_estimate.json", "w") as f:
            json.dump(
                {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                 for k, v in rul_info.items()},
                f, indent=2,
            )

    # ── 7. Save prediction results ────────────────────────────────────────────
    logger.info("STEP 7 — Saving full prediction results")

    # Re-run ensemble on all windows for prediction CSV
    df_feat_full = extract_all_features(X_all, y_all)
    X_feat_full  = df_feat_full[feature_names].values.astype(np.float32)

    ens_results_all = ensemble.predict(X_windows=X_all, X_flat=X_feat_full)

    predictions_df = pd.DataFrame({
        "timestamp":          ts_all,
        "true_label":         y_all,
        "if_pred":            if_model.predict(X_all.reshape(len(X_all), -1)),
        "if_score":           if_model.score(X_all.reshape(len(X_all), -1)),
        "lstm_pred":          lstm_model.predict(X_all),
        "lstm_score":         lstm_model.score(X_all),
        "lstm_recon_error":   all_lstm_errors,
        "xgb_fault_class":    ens_results_all["xgb_class_preds"],   # 0–8
        "xgb_pred":           ens_results_all["xgb_preds"],          # binary
        "xgb_score":          ens_results_all["xgb_scores"],
        "ensemble_pred":      ens_results_all["ensemble_preds"],
        "ensemble_score":     ens_results_all["ensemble_scores"],
        "votes":              ens_results_all["vote_counts"],
        "health_index":       df_feat_full["health_index"].values,
    })

    # Add human-readable fault class names
    predictions_df["fault_class_name"] = predictions_df["xgb_fault_class"].map(
        {int(k): v for k, v in fault_class_names.items()}
    ).fillna("Unknown")

    predictions_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    logger.info(f"Predictions saved → {OUTPUT_DIR / 'predictions.csv'}")
    
    # ✅ FINAL SANITY CHECK
    logger.info("\n" + "="*60)
    logger.info("FINAL PREDICTIONS SANITY CHECK")
    logger.info("="*60)
    logger.info(f"Predictions CSV shape: {predictions_df.shape}")
    logger.info(f"LSTM errors in CSV:")
    logger.info(f"  Min:  {predictions_df['lstm_recon_error'].min():.6f}")
    logger.info(f"  Max:  {predictions_df['lstm_recon_error'].max():.6f}")
    logger.info(f"  Mean: {predictions_df['lstm_recon_error'].mean():.6f}")
    
    if predictions_df['lstm_recon_error'].mean() > 1.0:
        logger.error("❌ FINAL CHECK FAILED: LSTM errors too high in predictions!")
        logger.error("   Dashboard will display incorrect values")
    else:
        logger.info("✅ FINAL CHECK PASSED: Predictions look correct")
    logger.info("="*60 + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"  Models saved to : {MODELS_DIR}")
    logger.info(f"  Results saved to: {OUTPUT_DIR}")
    logger.info("")
    logger.info("  Fault classes trained:")
    for cls_id, cls_name in fault_class_names.items():
        logger.info(f"    [{cls_id}] {cls_name}")
    logger.info("")
    logger.info("  Launch dashboard:")
    logger.info("    streamlit run dashboard/app.py")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()