"""
dashboard/app.py
----------------
Streamlit real-time monitoring dashboard for HP WIP-A Predictive Maintenance.

**UPDATED FOR MULTI-CLASS FAULT DETECTION (9 CLASSES) + AUGMENTED DATA + NEXT FAULT ONSET**

Tabs:
  1. Live Health Monitor   — current pump health + 9-class fault breakdown
  2. Sensor Trends         — interactive time-series with fault period highlighting
  3. Fault Analysis        — Multi-class fault distribution & physics insights
  4. Model Performance     — evaluation metrics for all 3 models
  5. Feature Importance    — XGBoost top features (multi-class trained)
  6. RUL & Fault Onset     — RUL forecast + next fault onset for all 9 classes
  7. Data Explorer         — full dataset browse & augmented data stats

Run:
  streamlit run dashboard/app.py
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import (
    OUTPUT_DIR, MODELS_DIR, FAULT_ONSET_DATE, FAULT_TEMP_SENSOR,
    VIBRATION_SENSORS, TEMPERATURE_SENSORS, ALL_SENSORS, SENSOR_LABELS,
    TARGET_RECALL, TARGET_PRECISION, TARGET_ROC_AUC, TARGET_MAX_FPR,
    TARGET_LEAD_TIME_HOURS, DATA_DIR, DATE_COLUMN,
)

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="HP WIP-A Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = "#1B2B4B"
TEAL   = "#00877A"
BLUE   = "#2E6CB8"
RED    = "#CC0000"
AMBER  = "#E67E22"
GREEN  = "#1A7A4A"
GOLD   = "#C8922A"
LIGHT  = "#F7F9FC"
PURPLE = "#8E44AD"
ORANGE = "#D35400"
CYAN   = "#16A085"

# Fault class color mapping (9 classes)
FAULT_COLORS = {
    0: GREEN,
    1: RED,
    2: AMBER,
    3: BLUE,
    4: PURPLE,
    5: CYAN,
    6: ORANGE,
    7: GOLD,
    8: "#E74C3C",
}

# Risk level color mapping
RISK_BG = {
    "Critical":       "#FFF0F0",
    "High":           "#FFF4E6",
    "Medium":         "#FFFBE6",
    "Low-Developing": "#F0F7FF",
    "Low":            "#F0FFF4",
    "None":           "#F5F5F5",
}
RISK_BORDER = {
    "Critical":       RED,
    "High":           AMBER,
    "Medium":         GOLD,
    "Low-Developing": BLUE,
    "Low":            GREEN,
    "None":           "#AAAAAA",
}
RISK_ORDER = ["Critical", "High", "Medium", "Low-Developing", "Low", "None"]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F7F9FC; }
    .stMetric { background-color: white; border-radius: 8px;
                padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .fault-alert { background-color: #FFF0F0; border-left: 5px solid #CC0000;
                   padding: 14px; border-radius: 4px; margin: 10px 0; }
    .normal-alert { background-color: #EAF7F5; border-left: 5px solid #00877A;
                    padding: 14px; border-radius: 4px; margin: 10px 0; }
    .warning-alert { background-color: #FFF8E7; border-left: 5px solid #E67E22;
                     padding: 14px; border-radius: 4px; margin: 10px 0; }
    .section-header { font-size: 1.1rem; font-weight: 700;
                      color: #1B2B4B; margin-bottom: 6px; }
    div[data-testid="metric-container"] { border: 1px solid #D1DAE8;
        border-radius: 8px; padding: 8px; }
    .fault-badge { display: inline-block; padding: 4px 10px; border-radius: 12px;
                   font-size: 11px; font-weight: 600; margin: 2px; }
    .onset-card { padding: 12px; border-radius: 6px; margin-bottom: 10px;
                  border-left: 4px solid; }
    /* Hide all metric delta indicators (arrows + colored text) */
    [data-testid="stMetricDelta"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_predictions() -> Optional[pd.DataFrame]:
    path = OUTPUT_DIR / "predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_fault_class_names() -> dict:
    path = MODELS_DIR / "fault_class_names.json"
    if not path.exists():
        return {
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
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


@st.cache_data(show_spinner=False)
def load_fault_onset_predictions() -> Optional[dict]:
    path = OUTPUT_DIR / "fault_onset_predictions.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


@st.cache_data(show_spinner=False)
def load_raw_sensor_data() -> Optional[pd.DataFrame]:
    augmented_path = DATA_DIR / "Augmented_DCS_Data.xlsx"

    if augmented_path.exists():
        df = pd.read_excel(augmented_path, sheet_name="Augmented_Data", header=0)
        st.sidebar.success("📊 Viewing Augmented Data (9 fault classes)")
    else:
        from config import RAW_DATA_FILE, DATA_SHEET
        if not RAW_DATA_FILE.exists():
            return None
        df = pd.read_excel(RAW_DATA_FILE, sheet_name=DATA_SHEET, header=0)
        st.sidebar.warning("⚠️ Viewing Real Data Only (2 classes)")

    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    for col in df.columns:
        if col != DATE_COLUMN:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(DATE_COLUMN).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metrics() -> Optional[pd.DataFrame]:
    path = OUTPUT_DIR / "evaluation_metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


@st.cache_data(show_spinner=False)
def load_feature_importance() -> Optional[pd.DataFrame]:
    path = OUTPUT_DIR / "feature_importance.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_rul() -> Optional[dict]:
    path = OUTPUT_DIR / "rul_estimate.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_sensor_summary() -> Optional[pd.DataFrame]:
    path = OUTPUT_DIR / "sensor_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _check_training_done() -> bool:
    return (OUTPUT_DIR / "predictions.csv").exists()


# ═════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════════════════════

def add_vertical_marker(
    fig,
    x,
    text: Optional[str] = None,
    color: str = RED,
    dash: str = "dash",
    width: float = 1.5,
    row: Optional[int] = None,
    col: Optional[int] = None,
):
    if isinstance(x, pd.Timestamp):
        x = x.to_pydatetime()

    def _is_subplot(figure):
        return getattr(figure, "_grid_ref", None) is not None

    shape_kw = {
        "type": "line",
        "x0": x, "x1": x,
        "y0": 0, "y1": 1,
        "xref": "x", "yref": "paper",
        "line": dict(color=color, dash=dash, width=width),
    }
    if row is not None and col is not None and _is_subplot(fig):
        shape_kw["row"] = row
        shape_kw["col"] = col
    fig.add_shape(**shape_kw)

    if text is not None:
        ann_kw = {
            "x": x, "y": 1,
            "xref": "x", "yref": "paper",
            "text": text,
            "showarrow": False,
            "xanchor": "left", "yanchor": "bottom",
            "font": dict(color=color, size=10),
        }
        if row is not None and col is not None and _is_subplot(fig):
            ann_kw["row"] = row
            ann_kw["col"] = col
        fig.add_annotation(**ann_kw)


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='background:{NAVY};padding:18px;border-radius:8px;margin-bottom:16px;'>
            <p style='color:#AAC3E0;font-size:11px;margin:0;'>UAEU | MEME685 | Spring 2026</p>
            <p style='color:white;font-size:16px;font-weight:700;margin:4px 0;'>HP WIP-A</p>
            <p style='color:#00B4A6;font-size:13px;margin:0;'>Predictive Maintenance</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Pump Specification**")
        st.markdown("""
        | Property | Value |
        |---|---|
        | Model | HPcp300-405-4s |
        | API Type | BB5 |
        | Flow | 6,416 USGPM |
        | Motor | 12 MW VFD |
        | Speed | 1,481 RPM |
        | Fluid | Treated Seawater |
        """)

        st.divider()

        st.markdown(f"""
        <div style='background:#E8F4F8;border-left:4px solid {BLUE};
                    padding:10px;border-radius:4px;margin-bottom:10px;'>
            <p style='color:{BLUE};font-weight:700;margin:0;font-size:12px;'>
            🧠 Multi-Class AI Detection</p>
            <p style='color:#333;font-size:11px;margin:4px 0 0;'>
            9-Class XGBoost Classifier<br>
            Real + 7 Synthetic Fault Types<br>
            Physics-Based Augmentation</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#FFF0F0;border-left:4px solid {RED};
                    padding:10px;border-radius:4px;'>
            <p style='color:{RED};font-weight:700;margin:0;font-size:12px;'>
            ⚠ ACTIVE FAULT DETECTED</p>
            <p style='color:#333;font-size:11px;margin:4px 0 0;'>
            TI0731 Thrust Bearing NDE 1<br>
            Onset: 28 Feb 2026 16:15<br>
            Peak: 300°C (sensor ceiling)</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        training_done = _check_training_done()
        if training_done:
            st.success("✓ Models trained & ready")
        else:
            st.warning("⚠ Models not trained yet")
            st.code("python src/train.py", language="bash")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 1 — Live Health Monitor
# ═════════════════════════════════════════════════════════════════════════════

def tab_health_monitor(preds_df):
    st.markdown("###  Live Health Monitor  Multi-Class Fault Detection")

    if preds_df is None:
        st.info("Run `python src/train.py` first to generate predictions.")
        return

    fault_class_names = load_fault_class_names()
    latest      = preds_df.iloc[-1]
    fault_onset = pd.Timestamp(FAULT_ONSET_DATE)

    ensemble_status = int(latest.get("ensemble_pred", 0))
    xgb_fault_class = int(latest.get("xgb_fault_class", 0))
    health_idx      = float(latest.get("health_index", 0))
    votes           = int(latest.get("votes", 0))
    fault_class_name = latest.get(
        "fault_class_name",
        fault_class_names.get(xgb_fault_class, "Unknown"),
    )

    if ensemble_status == 1:
        fault_color = FAULT_COLORS.get(xgb_fault_class, RED)
        st.markdown(f"""
        <div class='fault-alert'>
            <b style='font-size:16px;'>🔴 FAULT ALERT — Ensemble Triggered
            ({votes}/3 models agree)</b><br>
            <span class='fault-badge' style='background:{fault_color};color:white;'>
                Class {xgb_fault_class}: {fault_class_name}
            </span><br>
            <span style='font-size:13px;'>XGBoost Multi-Class Prediction</span><br>
            Immediate inspection recommended.
        </div>
        """, unsafe_allow_html=True)
    elif votes >= 1:
        st.markdown(f"""
        <div class='warning-alert'>
            <b style='font-size:15px;'>🟡 WARNING — Degradation Detected
            ({votes}/3 models agree)</b><br>
            Monitor closely. Prepare for planned maintenance.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='normal-alert'>
            <b style='font-size:15px;'>🟢 NORMAL — No anomaly detected</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Health Index", f"{health_idx:.3f}")
    with c2:
        st.metric("Detected Fault", fault_class_name[:20])
    with c3:
        st.metric("Ensemble Votes", f"{votes}/3")
    with c4:
        st.metric("LSTM Recon Error",
                  f"{latest.get('lstm_recon_error', 0):.5f}")
    with c5:
        st.metric("Anomaly Score",
                  f"{latest.get('if_score', 0):.3f}")
    with c6:
        st.metric("XGB Fault Prob",
                  f"{latest.get('xgb_score', 0):.3f}")

    st.markdown("---")
    st.markdown("####  XGBoost Multi-Class Probability Distribution (Latest Window)")

    prob_cols = [f"xgb_prob_class_{i}" for i in range(9)]
    if all(c in preds_df.columns for c in prob_cols):
        latest_probs = [latest[c] for c in prob_cols]
    else:
        latest_probs = [0.0] * 9
        latest_probs[xgb_fault_class] = 1.0

    fig_prob = go.Figure()
    fig_prob.add_trace(go.Bar(
        x=[fault_class_names.get(i, f"Class {i}") for i in range(9)],
        y=latest_probs,
        marker=dict(color=[FAULT_COLORS.get(i, NAVY) for i in range(9)]),
        text=[f"{p:.1%}" for p in latest_probs],
        textposition="outside",
    ))
    fig_prob.update_layout(
        title="Fault Class Probabilities — Latest Window",
        yaxis_title="Probability", xaxis_title="Fault Class",
        template="plotly_white", height=300, showlegend=False,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            "Ensemble Anomaly Score",
            "XGBoost Multi-Class Predictions (0-8)",
            "Individual Model Scores",
        ],
        vertical_spacing=0.08,
        row_heights=[0.3, 0.25, 0.45],
    )

    ts       = preds_df["timestamp"]
    fault_ts = preds_df[preds_df["true_label"] == 1]["timestamp"]

    if len(fault_ts) > 0:
        for r in [1, 2, 3]:
            fig.add_vrect(
                x0=fault_ts.min().to_pydatetime(),
                x1=fault_ts.max().to_pydatetime(),
                fillcolor="rgba(204,0,0,0.08)", line_width=0,
                row=r, col=1,
            )
            add_vertical_marker(
                fig, x=fault_onset,
                text="Fault Onset" if r == 1 else "",
                color=RED, width=1.5, row=r, col=1,
            )

    fig.add_trace(go.Scatter(
        x=ts, y=preds_df["ensemble_score"],
        name="Ensemble", line=dict(color=NAVY, width=2),
        fill="tozeroy", fillcolor="rgba(27,43,75,0.12)",
    ), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color=RED, line_width=1,
                  annotation_text="Alert Threshold", row=1, col=1)

    fc_series     = preds_df["xgb_fault_class"].values
    fc_colors_ser = [FAULT_COLORS.get(int(c), NAVY) for c in fc_series]
    fig.add_trace(go.Scatter(
        x=ts, y=fc_series, mode="markers",
        marker=dict(color=fc_colors_ser, size=6,
                    line=dict(width=0.5, color="white")),
        name="XGBoost Class",
        text=[fault_class_names.get(int(c), f"Class {c}") for c in fc_series],
        hovertemplate="<b>%{text}</b><br>Time: %{x}<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(title_text="Fault Class", tickmode="linear",
                     tick0=0, dtick=1, range=[-0.5, 8.5], row=2, col=1)

    for name, col_key, color in [
        ("Isolation Forest", "if_score",   TEAL),
        ("LSTM Autoencoder", "lstm_score", BLUE),
        ("XGBoost",          "xgb_score",  AMBER),
    ]:
        if col_key in preds_df.columns:
            fig.add_trace(go.Scatter(
                x=ts, y=preds_df[col_key],
                name=name, line=dict(color=color, width=1.5),
            ), row=3, col=1)

    fig.update_layout(
        height=700, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig.update_yaxes(range=[0, 1.1], row=1, col=1)
    fig.update_yaxes(range=[0, 1.1], row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Last 20 Windows — Prediction Detail")
    display_cols = [
        "timestamp", "true_label", "ensemble_pred", "ensemble_score",
        "xgb_fault_class", "fault_class_name", "xgb_score",
        "if_pred", "lstm_pred", "votes", "health_index",
    ]
    avail_cols = [c for c in display_cols if c in preds_df.columns]
    recent     = preds_df[avail_cols].tail(20).copy()

    def colour_row(row):
        if row.get("ensemble_pred", 0) == 1:
            return ["background-color: #FFF0F0"] * len(row)
        return [""] * len(row)

    st.dataframe(
        recent.style.apply(colour_row, axis=1).format(
            {c: "{:.3f}" for c in ["ensemble_score", "xgb_score", "health_index"]
             if c in recent.columns}
        ),
        use_container_width=True, height=400,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tab 2 — Sensor Trends
# ═════════════════════════════════════════════════════════════════════════════

def tab_sensor_trends(raw_df):
    st.markdown("###  Sensor Trends")

    if raw_df is None:
        st.info("Place the raw DCS Excel file in the data/ folder.")
        return

    fault_onset = pd.Timestamp(FAULT_ONSET_DATE)

    col1, col2 = st.columns([2, 1])
    with col1:
        sensor_type = st.radio(
            "Sensor Type", ["Temperature", "Vibration", "Both"], horizontal=True
        )
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(raw_df[DATE_COLUMN].min().date(),
                   raw_df[DATE_COLUMN].max().date()),
        )

    sensors_to_show = (
        TEMPERATURE_SENSORS if sensor_type == "Temperature"
        else VIBRATION_SENSORS if sensor_type == "Vibration"
        else ALL_SENSORS
    )
    sensors_available = [s for s in sensors_to_show if s in raw_df.columns]

    selected_sensors = st.multiselect(
        "Select Sensors",
        options=sensors_available,
        default=sensors_available[:4] if len(sensors_available) >= 4 else sensors_available,
        format_func=lambda x: SENSOR_LABELS.get(x, x),
    )
    if not selected_sensors:
        st.warning("Please select at least one sensor.")
        return

    mask = (
        (raw_df[DATE_COLUMN] >= pd.Timestamp(date_range[0])) &
        (raw_df[DATE_COLUMN] <= pd.Timestamp(date_range[1]))
    )
    plot_df = raw_df[mask]

    fig = go.Figure()
    for sensor in selected_sensors:
        label = SENSOR_LABELS.get(sensor, sensor)
        color = RED if sensor == FAULT_TEMP_SENSOR else None
        width = 2.5 if sensor == FAULT_TEMP_SENSOR else 1.5
        fig.add_trace(go.Scatter(
            x=plot_df[DATE_COLUMN], y=plot_df[sensor],
            name=label, line=dict(width=width, color=color),
        ))

    if "fault_class" in plot_df.columns:
        fcn = load_fault_class_names()
        for fc in range(1, 9):
            fd = plot_df[plot_df["fault_class"] == fc]
            if len(fd) > 0:
                fig.add_vrect(
                    x0=fd[DATE_COLUMN].min(), x1=fd[DATE_COLUMN].max(),
                    fillcolor=FAULT_COLORS.get(fc, NAVY), opacity=0.08,
                    line_width=0,
                    annotation_text=f"Class {fc}: {fcn.get(fc, '')}",
                    annotation_position="top left", annotation_font_size=9,
                )
    elif (
        fault_onset >= pd.Timestamp(date_range[0]) and
        fault_onset <= pd.Timestamp(date_range[1])
    ):
        add_vertical_marker(fig, x=fault_onset,
                            text="Fault Onset (Feb 28)", color=RED, width=2)

    unit = ("°C" if sensor_type == "Temperature"
            else "µm/s" if sensor_type == "Vibration" else "")
    fig.update_layout(
        title="Sensor Time-Series", xaxis_title="Date", yaxis_title=unit,
        template="plotly_white", height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Statistics — Selected Sensors")
    stats_rows = []
    for s in selected_sensors:
        nv = plot_df[plot_df[DATE_COLUMN] < fault_onset][s].dropna()
        fv = plot_df[plot_df[DATE_COLUMN] >= fault_onset][s].dropna()
        stats_rows.append({
            "Sensor":      SENSOR_LABELS.get(s, s),
            "Normal Mean": round(nv.mean(), 2) if len(nv) > 0 else "—",
            "Normal Std":  round(nv.std(),  2) if len(nv) > 0 else "—",
            "Fault Mean":  round(fv.mean(), 2) if len(fv) > 0 else "—",
            "Max Value":   round(plot_df[s].max(), 2),
            "Missing":     plot_df[s].isnull().sum(),
        })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Tab 3 — Fault Analysis
# ═════════════════════════════════════════════════════════════════════════════

def tab_fault_analysis(raw_df, preds_df):
    st.markdown("###  Fault Analysis  Multi-Class Detection & Physics Insights")

    fault_class_names = load_fault_class_names()

    if preds_df is None:
        st.info("Run training to see fault analysis.")
        return

    st.markdown("####  Fault Class Distribution (All Windows)")
    fault_counts = preds_df["xgb_fault_class"].value_counts().sort_index()

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=[fault_class_names.get(int(i), f"Class {i}") for i in fault_counts.index],
        y=fault_counts.values,
        marker=dict(color=[FAULT_COLORS.get(int(i), NAVY) for i in fault_counts.index]),
        text=fault_counts.values, textposition="outside",
    ))
    fig_dist.update_layout(
        title="Detected Fault Classes Across All Windows",
        xaxis_title="Fault Type", yaxis_title="Count",
        template="plotly_white", height=350,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔬 Physics-Based Fault Signatures")

    fault_descriptions = {
        0: "**Normal Operation** — Baseline condition with all parameters within spec.",
        1: "**Thrust Bearing NDE-1 Thermal Runaway** (Real Data) — TI0731 temperature spike to 300°C. Confirmed bearing failure.",
        2: "**Bearing Wear** (Synthetic) — Random impulse bursts + BPFO harmonic (~3.5× shaft) + gradual temperature rise in pump bearings.",
        3: "**Shaft Imbalance** (Synthetic) — Strong 1× RPM harmonic across all vibration sensors due to centrifugal force.",
        4: "**Misalignment** (Synthetic) — 2× RPM dominant with 1× and 3× sidebands, strongest at coupling/gearbox.",
        5: "**Cavitation** (Synthetic) — Broadband noise + vane-pass frequency + sub-synchronous component + casing temp rise.",
        6: "**Seal Degradation** (Synthetic) — Exponential casing temperature rise + sub-synchronous vibration from pressure fluctuations.",
        7: "**Gearbox Gear Wear** (Synthetic) — GMF (28× shaft) + sidebands (±1× shaft) + gearbox bearing temperature rise.",
        8: "**Motor Bearing Fault** (Synthetic) — BPFI/BPFO harmonics (~5.1× and ~3.4× shaft) + motor bearing temp rise.",
    }

    for cid in sorted(fault_descriptions.keys()):
        color = FAULT_COLORS.get(cid, NAVY)
        count = fault_counts.get(cid, 0)
        st.markdown(f"""
        <div style='border-left:4px solid {color};padding:10px;margin:8px 0;
                    background:white;border-radius:4px;'>
            <span class='fault-badge' style='background:{color};color:white;'>
                Class {cid}
            </span>
            <b>{fault_class_names.get(cid, f"Class {cid}")}</b>
            — {count} windows detected<br>
            <span style='font-size:13px;color:#555;'>
                {fault_descriptions[cid]}
            </span>
        </div>
        """, unsafe_allow_html=True)

    if raw_df is not None:
        st.markdown("---")
        st.markdown("#### 🌡️ TI0731 Thermal Runaway — Detailed Progression")

        fault_onset = pd.Timestamp(FAULT_ONSET_DATE)
        raw_df[DATE_COLUMN] = pd.to_datetime(raw_df[DATE_COLUMN])
        ti_data = raw_df[[DATE_COLUMN, FAULT_TEMP_SENSOR]].dropna()

        fig1 = go.Figure()
        nm = ti_data[DATE_COLUMN] < fault_onset
        fm = ti_data[DATE_COLUMN] >= fault_onset

        fig1.add_trace(go.Scatter(
            x=ti_data[nm][DATE_COLUMN], y=ti_data[nm][FAULT_TEMP_SENSOR],
            name="Normal Phase", line=dict(color=TEAL, width=2),
            fill="tozeroy", fillcolor="rgba(0,135,122,0.08)",
        ))
        fig1.add_trace(go.Scatter(
            x=ti_data[fm][DATE_COLUMN], y=ti_data[fm][FAULT_TEMP_SENSOR],
            name="Fault Phase", line=dict(color=RED, width=2.5),
            fill="tozeroy", fillcolor="rgba(204,0,0,0.10)",
        ))
        add_vertical_marker(fig1, x=fault_onset,
                            text=f"Fault Onset: {FAULT_ONSET_DATE}",
                            color=RED, width=2)
        fig1.add_hline(y=165, line_dash="dot", line_color=AMBER,
                       annotation_text="165°C Alert Threshold")
        fig1.update_layout(
            title="TI0731 — Thrust Bearing NDE 1 Temperature (Full Study Period)",
            yaxis_title="Temperature (°C)", xaxis_title="Date",
            template="plotly_white", height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig1, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Tab 4 — Model Performance
# ═════════════════════════════════════════════════════════════════════════════

def tab_model_performance(metrics_df):
    st.markdown("###  Model Performance")

    if metrics_df is None:
        st.info("Run training first to see model performance metrics.")
        return

    if "XGBoost" in metrics_df.index:
        ens = metrics_df.loc["XGBoost"]
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics_display = [
            (c1, "Recall",    "recall"),
            (c2, "Precision", "precision"),
            (c3, "ROC-AUC",   "roc_auc"),
            (c4, "F1 Score",  "f1"),
            (c5, "FPR",       "false_positive_rate"),
        ]
        for col, label, key in metrics_display:
            val = ens.get(key, None)
            if val is not None and str(val) != "nan":
                col.metric(f"XGBoost {label}", f"{float(val):.4f}")

    st.markdown("---")

    models       = ["Isolation Forest", "LSTM Autoencoder", "XGBoost", "Ensemble"]
    radar_mets   = ["recall", "precision", "f1"]
    radar_data   = {}
    for m in models:
        if m in metrics_df.index:
            vals = []
            for met in radar_mets:
                v = metrics_df.loc[m, met]
                vals.append(float(v) if str(v) != "nan" else 0.0)
            radar_data[m] = vals

    if radar_data:
        fig_radar = go.Figure()
        colors = [TEAL, BLUE, AMBER, NAVY]
        for i, (model, vals) in enumerate(radar_data.items()):
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=radar_mets + [radar_mets[0]],
                fill="toself", name=model,
                line=dict(color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)],
                opacity=0.3 if i < len(radar_data) - 1 else 0.5,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Model Comparison — Radar Chart",
            template="plotly_white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_radar, use_container_width=True)
        with col2:
            st.markdown("#### Full Metrics Table")
            disp_cols = ["recall", "precision", "f1", "roc_auc",
                         "false_positive_rate", "TP", "FP", "TN", "FN"]
            avail = [c for c in disp_cols if c in metrics_df.columns]
            st.dataframe(
                metrics_df[avail].style.format(
                    {c: "{:.4f}" for c in ["recall", "precision", "f1",
                                            "roc_auc", "false_positive_rate"]}
                ).highlight_max(
                    subset=["recall", "precision", "f1"],
                    color="rgba(0,135,122,0.2)",
                ),
                use_container_width=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# Tab 5 — Feature Importance
# ═════════════════════════════════════════════════════════════════════════════

def tab_feature_importance(fi_df):
    st.markdown("###  Feature Importance — XGBoost Multi-Class Classifier")
    st.caption("Trained on **augmented dataset** with 9 fault classes")

    if fi_df is None:
        st.info("Train the models first to see feature importance.")
        return

    top_n  = st.slider("Show top N features", 5, len(fi_df), 20)
    top_df = fi_df.head(top_n)

    def domain_color(name: str) -> str:
        if any(k in name for k in ["vib_rms", "vib_peak", "vib_kurt", "vib_std",
                                    "vib_mean", "vib_crest", "pump_vib", "gb_vib",
                                    "motor_vib", "temp_rms", "temp_mean",
                                    "ti0731", "shutdown"]):
            return TEAL
        if any(k in name for k in ["spectral", "band_energy", "peak_freq",
                                    "entropy", "centroid", "corr", "dominant"]):
            return BLUE
        return AMBER

    colors = [domain_color(f) for f in top_df["feature"]]

    fig = go.Figure(go.Bar(
        x=top_df["importance"], y=top_df["feature"],
        orientation="h", marker=dict(color=colors),
    ))
    fig.update_layout(
        title=f"Top {top_n} XGBoost Feature Importances (Multi-Class Model)",
        xaxis_title="Importance Score",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=max(350, top_n * 22),
        margin=dict(l=220),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"<span style='color:{TEAL}'>■ Time Domain</span> &nbsp; "
        f"<span style='color:{BLUE}'>■ Frequency Domain</span> &nbsp; "
        f"<span style='color:{AMBER}'>■ Process Health Indicators</span>",
        unsafe_allow_html=True,
    )

    with st.expander("Full Feature Importance Table"):
        st.dataframe(fi_df.style.format({"importance": "{:.6f}"}),
                     use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Tab 6 — RUL & Next Fault Onset
# ═════════════════════════════════════════════════════════════════════════════

def tab_rul(preds_df, rul_info):
    st.markdown("### ⏱️ RUL Estimation & Next Fault Onset Prediction")
    st.markdown(
        "*LSTM degradation trend (Class 1) + "
        "XGBoost probability extrapolation (Classes 2-8)*"
    )

    if preds_df is None:
        st.info("Run `python src/train.py` first to generate predictions.")
        return

    fault_class_names  = load_fault_class_names()
    fault_onset_preds  = load_fault_onset_predictions()

    t1, t2 = st.tabs([
        "🔮 Next Fault Onset (All Classes)",
        "📋 Maintenance Schedule",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # SUB-TAB 1 — NEXT FAULT ONSET (ALL 9 CLASSES)
    # ═════════════════════════════════════════════════════════════════════════

    with t1:
        st.markdown("####  Next Fault Onset — All 9 Classes")
        st.caption(
            "Predicted onset dates extrapolated from XGBoost class probability "
            "trends over the last 50 windows. Classes 2-8 trained on "
            "physics based augmented data."
        )

        if fault_onset_preds is None:
            st.warning("""
            **Fault onset predictions not found.**

            Possible reasons:
            - Training was run on real data only (no augmented data)
            - `outputs/fault_onset_predictions.json` does not exist

            **Fix:** Ensure `data/Augmented_DCS_Data.xlsx` exists, then run:
            ```
            python src/train.py
            ```
            """)
        else:
            st.markdown("#####  Risk Summary")

            non_normal = {
                k: v for k, v in fault_onset_preds.items() if int(k) > 0
            }
            sorted_by_risk = sorted(
                non_normal.items(),
                key=lambda x: RISK_ORDER.index(
                    x[1].get("risk_level", "Low")
                    if x[1].get("risk_level", "Low") in RISK_ORDER else "Low"
                ),
            )

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            if sorted_by_risk:
                top_fc, top_pred = sorted_by_risk[0]
                kpi1.metric(
                    "Highest Risk Fault",
                    top_pred["fault_name"][:22],
                )

            future_preds = [
                (k, v) for k, v in sorted_by_risk
                if v.get("days_until_onset") is not None
                and v["days_until_onset"] > 0
            ]
            if future_preds:
                next_k, next_v = min(
                    future_preds, key=lambda x: x[1]["days_until_onset"]
                )
                kpi2.metric(
                    "Next Onset In",
                    f"{next_v['days_until_onset']:.0f} days",
                )

            rising = sum(
                1 for v in fault_onset_preds.values()
                if v.get("trend_slope", 0) > 0.001
            )
            kpi3.metric(
                "Rising Probability Trends",
                f"{rising} / 9 classes",
            )

            critical_count = sum(
                1 for v in fault_onset_preds.values()
                if v.get("risk_level") == "Critical"
            )
            high_count = sum(
                1 for v in fault_onset_preds.values()
                if v.get("risk_level") == "High"
            )
            kpi4.metric(
                "Critical / High Risk",
                f"{critical_count} / {high_count}",
            )

            st.markdown("---")

            st.markdown("#####  Per-Class Fault Onset Cards")

            cols3 = st.columns(3)
            for idx, (fc, pred) in enumerate(sorted_by_risk):
                fc     = int(fc)
                col    = cols3[idx % 3]
                color  = FAULT_COLORS.get(fc, NAVY)
                rl     = pred.get("risk_level", "Low")
                bg     = RISK_BG.get(rl, "#F5F5F5")
                border = RISK_BORDER.get(rl, "#AAAAAA")

                days = pred.get("days_until_onset")
                days_str = f"⏰ ~{days:.0f} days" if days is not None else "No immediate risk"

                onset_raw = pred.get("predicted_onset")
                if onset_raw and str(onset_raw) not in ("None", "null", ""):
                    try:
                        onset_disp = pd.Timestamp(onset_raw).strftime("%b %d, %Y")
                    except Exception:
                        onset_disp = "—"
                else:
                    onset_disp = "—"

                trend_arrow = (
                    "📈 Rising" if pred.get("trend_slope", 0) > 0.001
                    else "📉 Falling" if pred.get("trend_slope", 0) < -0.001
                    else "➡ Stable"
                )

                col.markdown(f"""
                <div style='background:{bg};border-left:4px solid {border};
                            padding:12px;border-radius:6px;margin-bottom:10px;'>
                    <div style='margin-bottom:6px;'>
                        <span style='background:{color};color:white;padding:2px 8px;
                                     border-radius:10px;font-size:11px;font-weight:700;'>
                            Class {fc}
                        </span>
                        <b style='font-size:13px;'> {pred['fault_name']}</b>
                    </div>
                    <div style='font-size:12px;color:#444;line-height:1.8;'>
                        🔴 <b>Risk:</b> {rl}<br>
                        📊 <b>Probability:</b> {pred['current_prob']:.1%}<br>
                        {trend_arrow} ({pred.get('trend_slope', 0):+.4f}/win)<br>
                        {days_str}<br>
                        📅 <b>Est. Onset:</b> {onset_disp}<br>
                        🔬 <b>Confidence:</b> {pred.get('confidence','Low')}<br>
                        🧮 <b>Method:</b> <span style='font-size:11px;color:#777;'>
                            {pred.get('prediction_method','—')}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if 0 in fault_onset_preds:
                pred0 = fault_onset_preds[0]
                st.markdown(f"""
                <div style='background:#F0FFF4;border-left:4px solid {GREEN};
                            padding:12px;border-radius:6px;margin-top:4px;'>
                    <span style='background:{GREEN};color:white;padding:2px 8px;
                                 border-radius:10px;font-size:11px;font-weight:700;'>
                        Class 0
                    </span>
                    <b> Normal Operation</b> — Probability: 
                    {pred0.get('current_prob', 0):.1%} |
                    Status: {pred0.get('status','Active')}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            st.markdown("#####  Fault Class Probability Trends (Last 50 Windows)")

            prob_cols_avail = [
                f"xgb_prob_class_{i}" for i in range(9)
                if f"xgb_prob_class_{i}" in preds_df.columns
            ]

            if prob_cols_avail:
                recent_50  = preds_df.tail(50)
                fig_trends = go.Figure()

                for pc in prob_cols_avail:
                    fc_idx = int(pc.split("_")[-1])
                    if fc_idx == 0:
                        continue
                    fig_trends.add_trace(go.Scatter(
                        x=recent_50["timestamp"],
                        y=recent_50[pc],
                        name=fault_class_names.get(fc_idx, f"Class {fc_idx}"),
                        line=dict(color=FAULT_COLORS.get(fc_idx, NAVY), width=2),
                        mode="lines",
                    ))

                fig_trends.add_hline(
                    y=0.4, line_dash="dot", line_color=AMBER,
                    annotation_text="High Risk (0.4)",
                )
                fig_trends.add_hline(
                    y=0.7, line_dash="dot", line_color=RED,
                    annotation_text="Critical (0.7)",
                )
                fig_trends.update_layout(
                    title="Fault Class Probabilities — Last 50 Windows",
                    xaxis_title="Date", yaxis_title="Probability",
                    yaxis=dict(range=[0, 1]),
                    template="plotly_white", height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_trends, use_container_width=True)

            st.markdown("#####  Predicted Fault Onset Timeline")

            timeline_rows = []
            for fc, pred in fault_onset_preds.items():
                fc = int(fc)
                onset_raw = pred.get("predicted_onset")
                if onset_raw and str(onset_raw) not in ("None", "null", ""):
                    try:
                        onset_ts = pd.Timestamp(onset_raw)
                        timeline_rows.append({
                            "Class":      fc,
                            "Fault":      pred["fault_name"],
                            "Onset":      onset_ts,
                            "Risk":       pred.get("risk_level", "Low"),
                            "Confidence": pred.get("confidence", "Low"),
                            "Color":      FAULT_COLORS.get(fc, NAVY),
                            "Days":       pred.get("days_until_onset"),
                        })
                    except Exception:
                        pass

            if timeline_rows:
                tl_df  = pd.DataFrame(timeline_rows).sort_values("Onset")
                fig_tl = go.Figure()

                for _, row in tl_df.iterrows():
                    days_label = (
                        f"+{row['Days']:.0f}d" if row["Days"] is not None else ""
                    )
                    fig_tl.add_trace(go.Scatter(
                        x=[row["Onset"]],
                        y=[row["Class"]],
                        mode="markers+text",
                        marker=dict(
                            color=row["Color"], size=18,
                            symbol="diamond",
                            line=dict(width=2, color="white"),
                        ),
                        text=[f"  {row['Fault']} ({row['Risk']}) {days_label}"],
                        textposition="middle right",
                        name=f"Class {row['Class']}: {row['Fault']}",
                        hovertemplate=(
                            f"<b>Class {row['Class']}: {row['Fault']}</b><br>"
                            f"Est. Onset: {row['Onset'].strftime('%Y-%m-%d')}<br>"
                            f"Risk: {row['Risk']}<br>"
                            f"Confidence: {row['Confidence']}"
                            f"<extra></extra>"
                        ),
                    ))

                add_vertical_marker(
                    fig_tl,
                    x=preds_df["timestamp"].max(),
                    text="Now",
                    color=NAVY, dash="dash", width=2,
                )
                add_vertical_marker(
                    fig_tl,
                    x=pd.Timestamp(FAULT_ONSET_DATE),
                    text="Real Fault Onset (Class 1)",
                    color=RED, dash="dot", width=1.5,
                )

                fig_tl.update_layout(
                    title="Predicted Fault Onset Timeline — All Classes",
                    xaxis_title="Date", yaxis_title="Fault Class",
                    yaxis=dict(tickmode="linear", tick0=0,
                               dtick=1, range=[-0.5, 8.5]),
                    height=420,
                    template="plotly_white",
                    showlegend=False,
                    hovermode="closest",
                )
                st.plotly_chart(fig_tl, use_container_width=True)
            else:
                st.info("""
                No predicted onset dates available.
                This may indicate all fault class probabilities are low
                (healthy operation), or the model needs more fault data.
                """)

            st.markdown("#####  Current Risk Level — All Fault Classes")

            bar_classes = [int(k) for k in fault_onset_preds.keys() if int(k) > 0]
            bar_probs   = [fault_onset_preds[k]["current_prob"] for k in bar_classes]
            bar_labels  = [fault_onset_preds[k]["fault_name"] for k in bar_classes]
            bar_colors  = [FAULT_COLORS.get(k, NAVY) for k in bar_classes]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=bar_labels,
                y=bar_probs,
                marker=dict(color=bar_colors),
                text=[f"{p:.1%}" for p in bar_probs],
                textposition="outside",
                name="Current Probability",
            ))
            fig_bar.add_hline(
                y=0.4, line_dash="dot", line_color=AMBER,
                annotation_text="High Risk Threshold",
            )
            fig_bar.add_hline(
                y=0.7, line_dash="dot", line_color=RED,
                annotation_text="Critical Threshold",
            )
            fig_bar.update_layout(
                title="Current Fault Class Probabilities vs Risk Thresholds",
                yaxis_title="Probability", xaxis_title="Fault Class",
                yaxis=dict(range=[0, 1.1]),
                template="plotly_white", height=380,
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # SUB-TAB 2 — MAINTENANCE SCHEDULE
    # ═════════════════════════════════════════════════════════════════════════

    with t2:
        st.markdown("#### AI Generated Maintenance Schedule")
        st.caption("Prioritised by risk level and predicted onset date")

        if fault_onset_preds is None:
            st.info("""
            Run `python src/train.py` with augmented data to generate
            the maintenance schedule.
            """)
            return

        sched_rows = []
        for fc, pred in fault_onset_preds.items():
            fc = int(fc)
            if fc == 0:
                continue

            days      = pred.get("days_until_onset")
            onset_raw = pred.get("predicted_onset")
            if onset_raw and str(onset_raw) not in ("None", "null", ""):
                try:
                    onset_disp = pd.Timestamp(onset_raw).strftime("%Y-%m-%d")
                except Exception:
                    onset_disp = "—"
            else:
                onset_disp = "—"

            priority = {
                "Critical":       1, "High":           2,
                "Medium":         3, "Low-Developing": 4,
                "Low":            5, "None":           6,
            }.get(pred.get("risk_level", "Low"), 6)

            sched_rows.append({
                "Priority":        priority,
                "Class":           fc,
                "Fault Type":      pred["fault_name"],
                "Risk Level":      pred.get("risk_level", "Low"),
                "Current Prob":    f"{pred['current_prob']:.1%}",
                "Trend":           (
                    "📈 Rising" if pred.get("trend_slope", 0) > 0.001
                    else "📉 Falling" if pred.get("trend_slope", 0) < -0.001
                    else "➡ Stable"
                ),
                "Est. Onset":      onset_disp,
                "Days Until":      f"{days:.0f}" if days is not None else "—",
                "Confidence":      pred.get("confidence", "Low"),
                "Action Required": pred.get("recommendation",
                                            "Monitor and follow maintenance schedule"),
                "Source":          "✅ Real" if pred.get("is_real_fault") else "Augmented Data",
            })

        sched_df = (
            pd.DataFrame(sched_rows)
            .sort_values("Priority")
            .reset_index(drop=True)
        )

        def highlight_risk_row(row):
            colors_map = {
                "Critical": "background-color: #FFF0F0",
                "High":     "background-color: #FFF4E6",
                "Medium":   "background-color: #FFFBE6",
            }
            c = colors_map.get(row["Risk Level"], "")
            return [c] * len(row)

        st.dataframe(
            sched_df.drop(columns=["Priority"])
                    .style.apply(highlight_risk_row, axis=1),
            use_container_width=True,
            height=520,
        )

        csv_sched = sched_df.drop(columns=["Priority"]).to_csv(index=False)
        st.download_button(
            "⬇️ Download Maintenance Schedule (CSV)",
            data=csv_sched,
            file_name="maintenance_schedule.csv",
            mime="text/csv",
        )

        st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 7 — Data Explorer
# ═════════════════════════════════════════════════════════════════════════════

def tab_data_explorer(raw_df, sensor_summary, preds_df):
    st.markdown("###  Data Explorer & Augmentation Statistics")

    fault_class_names = load_fault_class_names()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("####  Original Dataset")
        if raw_df is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Samples",   f"{len(raw_df):,}")
            c2.metric("Sensor Channels", "35")
            c3.metric("Date Range",
                      f"{(raw_df[DATE_COLUMN].max() - raw_df[DATE_COLUMN].min()).days} days")
            c4.metric("Sample Interval", "15 min")

    with col2:
        st.markdown("####  Augmented Training Dataset")
        if preds_df is not None:
            fault_counts  = preds_df["xgb_fault_class"].value_counts()
            total_windows = len(preds_df)
            real_windows  = fault_counts.get(0, 0) + fault_counts.get(1, 0)
            synth_windows = total_windows - real_windows
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Windows", f"{total_windows:,}")
            c2.metric("Real Data",     f"{real_windows:,}")
            c3.metric("Synthetic",     f"{synth_windows:,}")
            c4.metric("Fault Classes", "9")

    st.markdown("---")

    ta, tb, tc, td = st.tabs([
        "Raw Data Preview",
        "Sensor Summary",
        "Correlation Heatmap",
        "Augmentation Breakdown",
    ])

    with ta:
        st.markdown("#### Raw DCS Data (first 200 rows)")
        if raw_df is not None:
            display_cols = [DATE_COLUMN] + [c for c in ALL_SENSORS if c in raw_df.columns]
            st.dataframe(
                raw_df[display_cols].head(200).rename(columns=SENSOR_LABELS),
                use_container_width=True, height=400,
            )
            csv = raw_df[display_cols].to_csv(index=False)
            st.download_button(
                "⬇️ Download Full Dataset (CSV)",
                data=csv, file_name="hp_wipa_dcs_data.csv", mime="text/csv",
            )

    with tb:
        if sensor_summary is not None:
            st.markdown("#### Sensor Statistics — Normal vs Fault Phase")
            st.dataframe(
                sensor_summary.style.format(
                    {c: "{:.2f}" for c in
                     sensor_summary.select_dtypes("number").columns}
                ),
                use_container_width=True,
            )

    with tc:
        st.markdown("#### Sensor Correlation Matrix")
        if raw_df is not None:
            sensors_for_corr = [
                c for c in VIBRATION_SENSORS[:8] + TEMPERATURE_SENSORS[:8]
                if c in raw_df.columns
            ]
            corr_matrix = raw_df[sensors_for_corr].corr()
            labels = [SENSOR_LABELS.get(c, c) for c in sensors_for_corr]
            fig_corr = px.imshow(
                corr_matrix.values, x=labels, y=labels,
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Sensor Correlation Heatmap", aspect="auto",
            )
            fig_corr.update_layout(height=520, template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)

    with td:
        st.markdown("####  Physics-Based Data Augmentation Breakdown")
        st.markdown("""
        The training dataset was augmented using **physics-based synthetic fault
        injection** to create labeled examples for fault classes not present in
        the real data.
        """)

        if preds_df is not None:
            fault_counts = preds_df["xgb_fault_class"].value_counts().sort_index()

            aug_df = pd.DataFrame({
                "Class": [int(i) for i in fault_counts.index],
                "Fault Type": [
                    fault_class_names.get(int(i), f"Class {i}")
                    for i in fault_counts.index
                ],
                "Count":  fault_counts.values,
                "Source": [
                    "Real Data" if i in [0, 1] else "Synthetic"
                    for i in fault_counts.index
                ],
                "Physics Basis": [
                    "Baseline operation"                       if i == 0 else
                    "TI0731 thermal runaway (confirmed)"       if i == 1 else
                    "Bearing BPFO harmonics + impulses"        if i == 2 else
                    "1× RPM centrifugal force"                 if i == 3 else
                    "2× RPM misalignment signature"            if i == 4 else
                    "Vane-pass frequency + broadband"          if i == 5 else
                    "Seal friction heat + sub-sync vibration"  if i == 6 else
                    "GMF + ±1× sidebands"                      if i == 7 else
                    "BPFI/BPFO bearing defect frequencies"     if i == 8 else
                    "Unknown"
                    for i in fault_counts.index
                ],
            })

            st.dataframe(
                aug_df.style.apply(
                    lambda row: [
                        f"background-color: {FAULT_COLORS.get(row['Class'], NAVY)}30"
                    ] * len(row),
                    axis=1,
                ),
                use_container_width=True,
            )

            fig_pie = go.Figure(data=[go.Pie(
                labels=[
                    fault_class_names.get(int(i), f"Class {i}")
                    for i in fault_counts.index
                ],
                values=fault_counts.values,
                marker=dict(colors=[
                    FAULT_COLORS.get(int(i), NAVY) for i in fault_counts.index
                ]),
                textinfo="label+percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Count: %{value}<br>"
                    "Percentage: %{percent}<extra></extra>"
                ),
            )])
            fig_pie.update_layout(
                title="Training Dataset Composition by Fault Class",
                template="plotly_white", height=400,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.info("""
            **Augmentation Methodology:**
            - Classes 2-8 generated by injecting physics-based signatures
              into normal windows
            - Each fault type uses domain-specific parameters from Sulzer BB5
              datasheets
            - Severity levels: Mild (8%), Moderate (20%), Severe (40%)
            - SMOTE resampling applied during XGBoost training to balance
              class distribution
            """)


# ═════════════════════════════════════════════════════════════════════════════
# Main App
# ═════════════════════════════════════════════════════════════════════════════

def main():
    render_sidebar()

    st.markdown(f"""
    <div style='background:{NAVY};padding:16px 22px;border-radius:8px;
                margin-bottom:20px;border-left:5px solid {GOLD};'>
        <p style='color:#AAC3E0;font-size:11px;margin:0;'>
            MEME685 | MS Engineering Management | UAEU | Spring 2026</p>
        <p style='color:white;font-size:20px;font-weight:700;margin:4px 0;'>
            HP WIP-A Predictive Maintenance Dashboard</p>
        <p style='color:#00B4A6;font-size:13px;margin:0;'>
            AI-Based Multi-Class Fault Detection & RUL Estimation —
            Sulzer BB5 API 610</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        preds_df       = load_predictions()
        raw_df         = load_raw_sensor_data()
        metrics_df     = load_metrics()
        fi_df          = load_feature_importance()
        rul_info       = load_rul()
        sensor_summary = load_sensor_summary()

    tabs = st.tabs([
        "⚙️ Health Monitor",
        "📈 Sensor Trends",
        "🔴 Fault Analysis",
        "📊 Model Performance",
        "🔬 Feature Importance",
        "⏱️ RUL & Fault Onset",
        "🗄️ Data Explorer",
    ])

    with tabs[0]: tab_health_monitor(preds_df)
    with tabs[1]: tab_sensor_trends(raw_df)
    with tabs[2]: tab_fault_analysis(raw_df, preds_df)
    with tabs[3]: tab_model_performance(metrics_df)
    with tabs[4]: tab_feature_importance(fi_df)
    with tabs[5]: tab_rul(preds_df, rul_info)
    with tabs[6]: tab_data_explorer(raw_df, sensor_summary, preds_df)


if __name__ == "__main__":
    main()
