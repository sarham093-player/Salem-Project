"""
config.py
---------
Central configuration for the HP WIP-A Predictive Maintenance project.
All tunable parameters, file paths, and sensor mappings live here.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Place your Excel file in the data/ folder
RAW_DATA_FILE = DATA_DIR / "XXXX HP WIP-A  PI DCS Process and Mech data.xlsx"

# ─── Data Sheet Config ────────────────────────────────────────────────────────
DATA_SHEET  = "HP WIP A"
TAGS_SHEET  = "Tags"
DATE_COLUMN = "Date"

# ─── Fault Event (confirmed from data analysis) ───────────────────────────────
FAULT_ONSET_DATE    = "2026-02-28 16:15:00"   # TI0731 first crossed 165°C
FAULT_LABEL_START   = "2026-02-28"            # Full fault phase from this date
NORMAL_PHASE_END    = "2026-02-27 23:59:59"   # Last reading in normal phase
FAULT_TEMP_SENSOR   = "2026TI0731.PV"         # Primary fault sensor
FAULT_TEMP_THRESHOLD = 165.0                   # °C — confirmed fault onset threshold

# ─── Sensor Groups ────────────────────────────────────────────────────────────
VIBRATION_SENSORS = [
    "2026VI0731X.PV", "2026VI0731Y.PV",   # Pump NDE
    "2026VI0732X.PV", "2026VI0732Y.PV",   # Pump DE
    "2026VI0733X.PV", "2026VI0733Y.PV",   # GB LSS NDE
    "2026VI0734X.PV", "2026VI0734Y.PV",   # GB HSS DE
    "2026VI0735X.PV", "2026VI0735Y.PV",   # GB LSS DE
    "2026VI0736X.PV", "2026VI0736Y.PV",   # GB HSS NDE
    "2026VI0737X.PV", "2026VI0737Y.PV",   # Motor DE
    "2026VI0738X.PV", "2026VI0738Y.PV",   # Motor NDE
]

TEMPERATURE_SENSORS = [
    "2026TI0724.PV", "2026TI0725.PV",     # Motor DE/NDE Bearing
    "2026TI0730.PV",                       # Casing
    "2026TI0731.PV", "2026TI0732.PV",     # Thrust Bearing NDE 1 & 2
    "2026TI0733.PV", "2026TI0734.PV",     # Thrust Bearing DE 2 & 1
    "2026TI0735.PV", "2026TI0736.PV",     # Pump NDE/DE Bearing
    "2026TI0737.PV", "2026TI0738.PV",     # GB Thrust Bearing
    "2026TI0739.PV", "2026TI0740.PV",     # GB Thrust Bearing
    "2026TI0741.PV", "2026TI0742.PV",     # GB LS/HS NDE Bearing
    "2026TI0743.PV", "2026TI0744.PV",     # GB Thrust / LS DE Bearing
    "2026TI0745.PV", "2026TI0746.PV",     # Motor DE/NDE Bearing
]

ALL_SENSORS = VIBRATION_SENSORS + TEMPERATURE_SENSORS

# Human-readable labels for sensors
SENSOR_LABELS = {
    "2026VI0731X.PV": "Pump NDE Vib X",
    "2026VI0731Y.PV": "Pump NDE Vib Y",
    "2026VI0732X.PV": "Pump DE Vib X",
    "2026VI0732Y.PV": "Pump DE Vib Y",
    "2026VI0733X.PV": "GB LSS NDE Vib X",
    "2026VI0733Y.PV": "GB LSS NDE Vib Y",
    "2026VI0734X.PV": "GB HSS DE Vib X",
    "2026VI0734Y.PV": "GB HSS DE Vib Y",
    "2026VI0735X.PV": "GB LSS DE Vib X",
    "2026VI0735Y.PV": "GB LSS DE Vib Y",
    "2026VI0736X.PV": "GB HSS NDE Vib X",
    "2026VI0736Y.PV": "GB HSS NDE Vib Y",
    "2026VI0737X.PV": "Motor DE Vib X",
    "2026VI0737Y.PV": "Motor DE Vib Y",
    "2026VI0738X.PV": "Motor NDE Vib X",
    "2026VI0738Y.PV": "Motor NDE Vib Y",
    "2026TI0724.PV":  "Motor DE Bearing Temp",
    "2026TI0725.PV":  "Motor NDE Bearing Temp",
    "2026TI0730.PV":  "Casing Temp",
    "2026TI0731.PV":  "Thrust Brg NDE1 Temp ⚠",
    "2026TI0732.PV":  "Thrust Brg NDE2 Temp",
    "2026TI0733.PV":  "Thrust Brg DE2 Temp",
    "2026TI0734.PV":  "Thrust Brg DE1 Temp",
    "2026TI0735.PV":  "Pump NDE Bearing Temp",
    "2026TI0736.PV":  "Pump DE Bearing Temp",
    "2026TI0737.PV":  "GB Thrust Brg Temp 1",
    "2026TI0738.PV":  "GB Thrust Brg Temp 2",
    "2026TI0739.PV":  "GB Thrust Brg Temp 3",
    "2026TI0740.PV":  "GB Thrust Brg Temp 4",
    "2026TI0741.PV":  "GB LS NDE Bearing Temp",
    "2026TI0742.PV":  "GB HS NDE Bearing Temp",
    "2026TI0743.PV":  "GB Thrust Brg Temp 5",
    "2026TI0744.PV":  "GB LS DE Bearing Temp", 
    "2026TI0745.PV":  "Motor DE Bearing Temp 2",
    "2026TI0746.PV":  "Motor NDE Bearing Temp 2",
}

# ─── Preprocessing Parameters ────────────────────────────────────────────────
SAMPLING_INTERVAL_MIN   = 15          # minutes between readings
WINDOW_SIZE             = 96          # samples per window = 24 hours
WINDOW_STEP             = 48          # 50% overlap
SHUTDOWN_VIB_THRESHOLD  = 2.0         # µm/s — below this = pump trip
MAX_INTERP_GAP          = 3           # max consecutive NaN to interpolate
BUTTERWORTH_ORDER       = 4           # filter order for vibration
BUTTERWORTH_CUTOFF_HZ   = 0.1         # normalised cutoff (0–1 relative to Nyquist)
NORMAL_BASELINE_TEMP    = 108.6       # TI0731 average in normal phase (°C)

# ─── Model Parameters ────────────────────────────────────────────────────────
RANDOM_SEED             = 42
TEST_SIZE               = 0.10        # 10% test
VAL_SIZE                = 0.20        # 20% validation (of remaining)
CV_FOLDS                = 5
SMOTE_RANDOM_STATE      = 42

# Isolation Forest
IF_CONTAMINATION        = 0.05
IF_N_ESTIMATORS         = 200
IF_MAX_SAMPLES          = 96

# LSTM Autoencoder
LSTM_EPOCHS             = 100
LSTM_BATCH_SIZE         = 16
LSTM_LEARNING_RATE      = 0.001
LSTM_LATENT_DIM         = 16          # bottleneck size
LSTM_THRESHOLD_PERCENTILE = 95        # anomaly threshold on normal reconstruction errors

# XGBoost
XGB_N_ESTIMATORS        = 300
XGB_MAX_DEPTH           = 6
XGB_LEARNING_RATE       = 0.05
XGB_SUBSAMPLE           = 0.8
XGB_COLSAMPLE_BYTREE    = 0.8

# Ensemble
ENSEMBLE_MIN_VOTES      = 2           # alerts when >= 2 of 3 models agree

# ─── Evaluation Targets ──────────────────────────────────────────────────────
TARGET_RECALL           = 0.9
TARGET_PRECISION        = 0.88
TARGET_ROC_AUC          = 0.95
TARGET_LEAD_TIME_HOURS  = 24
TARGET_MAX_FPR          = 0.08

# ─── Pump Specs (from Sulzer data sheet) ─────────────────────────────────────
PUMP_MODEL              = "HPcp300-405-4s"
PUMP_RATED_FLOW_USGPM   = 6416
PUMP_RATED_SPEED_RPM    = 1481
PUMP_STAGES             = 4
PUMP_MOTOR_KW           = 12000
PUMP_DISCHARGE_PSIG     = 3089
PUMP_SUCTION_PSIG       = 157
