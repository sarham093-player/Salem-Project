"""Generate augmented DCS data with synthetic faults - WINDOWS COMPATIBLE"""

import sys
from pathlib import Path

# Force UTF-8 encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*70)
print("AUGMENTATION GENERATOR")
print("="*70)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from datetime import timedelta

from config import (
    RAW_DATA_FILE, DATA_SHEET, DATE_COLUMN, ALL_SENSORS,
    NORMAL_PHASE_END, FAULT_LABEL_START, DATA_DIR,
)
from fault_augmentation import _FAULT_INJECTORS, get_fault_class_names

print(f"\nData file: {RAW_DATA_FILE}")
print(f"Exists: {RAW_DATA_FILE.exists()}")

if not RAW_DATA_FILE.exists():
    print(f"\nERROR: File not found!")
    print(f"\nFiles in data/:")
    for f in DATA_DIR.iterdir():
        print(f"  {f.name}")
    sys.exit(1)

def main():
    print("\nLoading data...")
    df_real = pd.read_excel(RAW_DATA_FILE, sheet_name=DATA_SHEET, header=0)
    df_real = df_real.loc[:, ~df_real.columns.str.contains(r"^Unnamed")]
    df_real[DATE_COLUMN] = pd.to_datetime(df_real[DATE_COLUMN])
    
    for col in ALL_SENSORS:
        if col in df_real.columns:
            df_real[col] = pd.to_numeric(df_real[col], errors="coerce")
    
    df_real = df_real.sort_values(DATE_COLUMN).reset_index(drop=True)
    print(f"  Loaded {len(df_real):,} rows")
    
    print("\nLabeling real data...")
    df_real["fault_class"] = 0
    df_real.loc[df_real[DATE_COLUMN] >= FAULT_LABEL_START, "fault_class"] = 1
    df_real["fault_name"] = df_real["fault_class"].map({
        0: "Normal",
        1: "Thrust Bearing NDE-1 Thermal Runaway"
    })
    df_real["is_synthetic"] = False
    print(f"  Class 0: {(df_real['fault_class']==0).sum():,}")
    print(f"  Class 1: {(df_real['fault_class']==1).sum():,}")
    
    print("\nExtracting normal baseline...")
    normal_phase_end = pd.Timestamp(NORMAL_PHASE_END)
    df_normal = df_real[df_real[DATE_COLUMN] <= normal_phase_end].copy()
    print(f"  {len(df_normal):,} normal samples")
    
    print("\nGenerating synthetic faults (Classes 2-8)...")
    fault_names = get_fault_class_names()
    rng = np.random.default_rng(42)
    
    synthetic_parts = []
    last_date = df_real[DATE_COLUMN].max()
    
    for fault_class in range(2, 9):
        print(f"\n  Class {fault_class}: {fault_names[fault_class]}")
        
        start_date = last_date + timedelta(days=1)
        n_samples = 14 * 24 * 4
        
        timestamps = pd.date_range(start=start_date, periods=n_samples, freq="15T")
        synthetic_data = np.zeros((n_samples, len(ALL_SENSORS)))
        
        inject_fn = _FAULT_INJECTORS[fault_class]
        
        for i in range(0, n_samples, 96):
            end = min(i + 96, n_samples)
            actual_size = end - i
            
            random_start = rng.integers(0, len(df_normal) - 96)
            base_window = df_normal.iloc[random_start:random_start + actual_size][ALL_SENSORS].values
            
            synthetic_window = inject_fn(base_window, 0.20, rng)
            synthetic_data[i:end] = synthetic_window
        
        df_synth = pd.DataFrame(synthetic_data, columns=ALL_SENSORS)
        df_synth[DATE_COLUMN] = timestamps
        df_synth["fault_class"] = fault_class
        df_synth["fault_name"] = fault_names[fault_class]
        df_synth["is_synthetic"] = True
        
        synthetic_parts.append(df_synth)
        print(f"    Generated {len(df_synth):,} samples")
        
        last_date = timestamps[-1]
    
    print("\nCombining datasets...")
    df_augmented = pd.concat([df_real] + synthetic_parts, ignore_index=True)
    df_augmented = df_augmented.sort_values(DATE_COLUMN).reset_index(drop=True)
    print(f"  Total: {len(df_augmented):,} samples")
    
    print("\nSaving files...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    output_xlsx = DATA_DIR / "Augmented_DCS_Data.xlsx"
    output_csv = DATA_DIR / "Augmented_DCS_Data.csv"
    
    cols = [DATE_COLUMN, "fault_class", "fault_name", "is_synthetic"] + ALL_SENSORS
    
    print(f"  Writing {output_xlsx.name}...")
    df_augmented[cols].to_excel(output_xlsx, index=False, sheet_name="Augmented_Data")
    print(f"    Done ({output_xlsx.stat().st_size / 1e6:.1f} MB)")
    
    print(f"  Writing {output_csv.name}...")
    df_augmented[cols].to_csv(output_csv, index=False)
    print(f"    Done ({output_csv.stat().st_size / 1e6:.1f} MB)")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    
    for fc in sorted(df_augmented["fault_class"].unique()):
        count = (df_augmented["fault_class"] == fc).sum()
        name = df_augmented[df_augmented["fault_class"] == fc]["fault_name"].iloc[0]
        print(f"  Class {fc}: {name} - {count:,} samples")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()