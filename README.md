# HP WIP-A Predictive Maintenance System
### AI-Based Bearing Fault Detection & RUL Estimation
**MEME685 | MS Engineering Management | UAEU | Spring 2026**

---

## Project Overview
AI-based predictive maintenance system for the **Sulzer BB5 API 610 High-Pressure Water Injection Pump (HP WIP-A)** using live DCS historian data. Detects Thrust Bearing NDE 1 thermal runaway (confirmed fault onset: Feb 28, 2026) and estimates Remaining Useful Life.

---

## Project Structure

```
pdm_project/
│
├── data/                          ← Place your Excel file here
│   └── XXXX_HP_WIP-A__PI_DCS_Process_and_Mech_data.xlsx
│
├── src/
│   ├── config.py                  ← All parameters, paths, sensor mappings
│   ├── data_loader.py             ← Step 1: Raw ingestion & validation
│   ├── preprocessor.py            ← Steps 2–6: Full preprocessing pipeline
│   ├── feature_engineering.py     ← ~45 features (time, freq, health)
│   ├── models.py                  ← IsolationForest + LSTM + XGBoost + Ensemble
│   └── train.py                   ← Main training script (run this first)
│
├── dashboard/
│   └── app.py                     ← Streamlit monitoring dashboard
│
├── models/                        ← Saved model files (auto-created)
│   ├── isolation_forest.pkl
│   ├── lstm_autoencoder.keras
│   ├── lstm_threshold.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── feature_names.json
│
├── outputs/                       ← Results (auto-created)
│   ├── predictions.csv
│   ├── evaluation_metrics.csv
│   ├── feature_importance.csv
│   ├── sensor_summary.csv
│   └── rul_estimate.json
│
└── requirements.txt
```

---

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Place Data File
Copy your DCS Excel file into the `data/` folder:
```
data/XXXX_HP_WIP-A__PI_DCS_Process_and_Mech_data.xlsx
```

---

## Running the System

### Step 1 — Train Models
```bash
cd src
python train.py
```
This runs the full pipeline:
- Loads and validates 9,313 DCS samples
- Applies 6-step preprocessing (missing values → filtering → normalisation → labeling → windowing)
- Extracts ~45 features per 24-hour window
- Trains Isolation Forest, LSTM Autoencoder, XGBoost
- Evaluates 2-of-3 ensemble on test set
- Saves all models to `models/` and results to `outputs/`

Expected runtime: ~5–15 minutes (LSTM training dominates)

### Step 2 — Launch Dashboard
```bash
cd ..   # back to project root
streamlit run dashboard/app.py
```
Opens at: http://localhost:8501

---

## Methodology Summary

| Step | Action |
|---|---|
| **1. Load** | Read PI Historian Excel export, map 35 sensor tags |
| **2. Missing Values** | Linear interpolation for gaps ≤3 readings (0.06% missing) |
| **3. Filter** | Butterworth low-pass (vibration) + Kalman smoother (temperature) |
| **4. Normalise** | Min-Max scaling fitted on Normal phase only (no leakage) |
| **5. Label** | Normal=0 (Jan–Feb 27), Fault=1 (Feb 28–Apr 8), confirmed by TI0731 + vibration cross-evidence |
| **6. Window** | 96-sample windows (24 hrs), 50% overlap → ~259 windows |
| **Features** | 16 time-domain + 14 frequency-domain + 15 health indicators = 45 total |
| **Model 1** | Isolation Forest — unsupervised anomaly detection |
| **Model 2** | LSTM Autoencoder — temporal degradation + RUL estimation |
| **Model 3** | XGBoost — supervised fault classification + feature importance |
| **Ensemble** | Alert when ≥ 2 of 3 models flag anomaly (2-of-3 voting) |

---

## Evaluation Targets

| Metric | Target |
|---|---|
| Recall | ≥ 92% |
| Precision | ≥ 88% |
| ROC-AUC | ≥ 0.95 |
| Alert Lead Time | ≥ 24 hours |
| False Positive Rate | ≤ 8% |

---

## Key Finding
**TI0731 (Thrust Bearing NDE 1) confirmed thermal runaway:**
- Normal phase average: 108.6°C (Jan 1 – Feb 27)
- Fault onset: Feb 28, 2026 at 16:15 — first reading at 165.0°C
- Rate of rise: 7.69°C per 15-minute reading
- Peak (Apr 8): 300°C (sensor ceiling — real temperature may be higher)
- Simultaneous vibration drop on all 16 channels: −56% to −62% (pump tripping)

---

## Pump Specification
| Property | Value |
|---|---|
| Manufacturer | Sulzer Pumps Middle East Ltd |
| Model | HPcp300-405-4s |
| API Type | BB5 (Barrel, Between Bearings) |
| Rated Flow | 6,416 USGPM |
| Discharge Pressure | 3,089 psig |
| Motor | 12 MW VFD, 1,481 RPM |
| Data Sheet | PDS100156098001 Rev B1 (ZADCO, 2017) |
