# check_saved_models.py
import joblib
import numpy as np
from pathlib import Path

print("="*60)
print("CHECKING SAVED MODELS")
print("="*60)

MODELS_DIR = Path("models")

# 1. Check LSTM threshold
print("\n1️⃣  LSTM THRESHOLD CHECK:")
try:
    threshold = joblib.load(MODELS_DIR / "lstm_threshold.pkl")
    print(f"   Saved threshold: {threshold:.6f}")
    
    if threshold > 1.0:
        print(f"   ❌ ERROR: Threshold too high ({threshold:.4f})")
        print(f"      This means training data was NOT normalized")
        print(f"      Expected: 0.001 - 0.01")
    elif threshold < 0.0001:
        print(f"   ⚠️  WARNING: Threshold very low ({threshold:.6f})")
    else:
        print(f"   ✅ Threshold OK")
except Exception as e:
    print(f"   ❌ Error loading: {e}")

# 2. Check scaler
print("\n2️⃣  SCALER CHECK:")
try:
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    print(f"   Scaler type: {type(scaler).__name__}")
    print(f"   Number of features: {len(scaler.data_min_)}")
    print(f"   Data min (first 5 sensors): {scaler.data_min_[:5]}")
    print(f"   Data max (first 5 sensors): {scaler.data_max_[:5]}")
    
    # Check if scaler looks reasonable
    if scaler.data_max_[0] < 2.0:
        print(f"   ⚠️  WARNING: Scaler fitted on already-normalized data?")
    else:
        print(f"   ✅ Scaler looks OK (fitted on raw data)")
except Exception as e:
    print(f"   ❌ Error loading: {e}")

# 3. Check predictions.csv
print("\n3️⃣  PREDICTIONS FILE CHECK:")
try:
    import pandas as pd
    preds = pd.read_csv("outputs/predictions.csv")
    
    print(f"   Rows: {len(preds)}")
    print(f"   LSTM error stats:")
    print(f"      Min:  {preds['lstm_recon_error'].min():.6f}")
    print(f"      Max:  {preds['lstm_recon_error'].max():.6f}")
    print(f"      Mean: {preds['lstm_recon_error'].mean():.6f}")
    
    if preds['lstm_recon_error'].mean() > 1.0:
        print(f"   ❌ ERROR: Predictions show non-normalized data")
    else:
        print(f"   ✅ Predictions look OK")
except Exception as e:
    print(f"   ❌ Error loading: {e}")

print("\n" + "="*60)