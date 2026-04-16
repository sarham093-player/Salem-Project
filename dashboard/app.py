"""
dashboard/app.py
----------------
Streamlit real-time monitoring dashboard for HP WIP-A Predictive Maintenance.

**UPDATED FOR MULTI-CLASS FAULT DETECTION (9 CLASSES)**

Tabs:
  1. Live Health Monitor   — current pump health + 9-class fault breakdown
  2. Sensor Trends         — interactive time-series for all 35 sensors
  3. Fault Analysis        — Multi-class fault distribution & physics insights
  4. Model Performance     — evaluation metrics for all 3 models
  5. Feature Importance    — XGBoost top features (multi-class trained)
  6. RUL Estimation        — Remaining Useful Life forecast
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
    TARGET_LEAD_TIME_HOURS,
)

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="HP WIP-A Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY    = "#1B2B4B"
TEAL    = "#00877A"
BLUE    = "#2E6CB8"
RED     = "#CC0000"
AMBER   = "#E67E22"
GREEN   = "#1A7A4A"
GOLD    = "#C8922A"
LIGHT   = "#F7F9FC"
PURPLE  = "#8E44AD"
ORANGE  = "#D35400"
CYAN    = "#16A085"

# Fault class color mapping (9 classes)
FAULT_COLORS = {
    0: GREEN,      # Normal
    1: RED,        # Thrust Bearing NDE-1 (real fault)
    2: AMBER,      # Bearing Wear
    3: BLUE,       # Shaft Imbalance
    4: PURPLE,     # Misalignment
    5: CYAN,       # Cavitation
    6: ORANGE,     # Seal Degradation
    7: GOLD,       # Gearbox Gear Wear
    8: "#E74C3C",  # Motor Bearing Fault
}

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
    """Load fault class names from training artifacts."""
    path = MODELS_DIR / "fault_class_names.json"
    if not path.exists():
        # Fallback defaults
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
def load_raw_sensor_data() -> Optional[pd.DataFrame]:
    """Load the original sensor time-series for trend plots."""
    from config import RAW_DATA_FILE, DATA_SHEET, DATE_COLUMN
    if not RAW_DATA_FILE.exists():
        return None
    df = pd.read_excel(RAW_DATA_FILE, sheet_name=DATA_SHEET, header=0)
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

    def _is_subplot_figure(figure):
        return getattr(figure, "_grid_ref", None) is not None

    shape_kwargs = {
        "type": "line",
        "x0": x,
        "x1": x,
        "y0": 0,
        "y1": 1,
        "xref": "x",
        "yref": "paper",
        "line": dict(color=color, dash=dash, width=width),
    }
    if row is not None and col is not None and _is_subplot_figure(fig):
        shape_kwargs["row"] = row
        shape_kwargs["col"] = col
    fig.add_shape(**shape_kwargs)

    if text is not None:
        annotation_kwargs = {
            "x": x,
            "y": 1,
            "xref": "x",
            "yref": "paper",
            "text": text,
            "showarrow": False,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": dict(color=color, size=10),
        }
        if row is not None and col is not None and _is_subplot_figure(fig):
            annotation_kwargs["row"] = row
            annotation_kwargs["col"] = col
        fig.add_annotation(**annotation_kwargs)


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
        
        # Multi-class fault detection badge
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
# Tab 1 — Live Health Monitor (UPDATED FOR MULTI-CLASS)
# ═════════════════════════════════════════════════════════════════════════════

def tab_health_monitor(preds_df):
    st.markdown("### ⚙️ Live Health Monitor — Multi-Class Fault Detection")

    if preds_df is None:
        st.info("Run `python src/train.py` first to generate predictions.")
        return

    fault_class_names = load_fault_class_names()

    # Latest window stats
    latest = preds_df.iloc[-1]
    fault_onset = pd.Timestamp(FAULT_ONSET_DATE)

    ensemble_status = int(latest.get("ensemble_pred", 0))
    xgb_fault_class = int(latest.get("xgb_fault_class", 0))
    health_idx      = float(latest.get("health_index", 0))
    votes           = int(latest.get("votes", 0))
    
    fault_class_name = latest.get("fault_class_name", 
                                   fault_class_names.get(xgb_fault_class, "Unknown"))

    # ── Status Banner with Fault Type ──
    if ensemble_status == 1:
        fault_color = FAULT_COLORS.get(xgb_fault_class, RED)
        st.markdown(f"""
        <div class='fault-alert'>
            <b style='font-size:16px;'>🔴 FAULT ALERT — Ensemble Triggered ({votes}/3 models agree)</b><br>
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
            <b style='font-size:15px;'>🟡 WARNING — Degradation Detected ({votes}/3 models agree)</b><br>
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

    # ── KPI Metrics Row ──
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Health Index", f"{health_idx:.3f}",
                  delta="↑ Degrading" if health_idx > 0.5 else "↓ Normal",
                  delta_color="inverse")
    with col2:
        st.metric("Detected Fault", fault_class_name[:20], 
                  delta=f"Class {xgb_fault_class}")
    with col3:
        st.metric("Ensemble Votes", f"{votes}/3",
                  delta=f"Score: {latest.get('ensemble_score', 0):.3f}")
    with col4:
        st.metric("LSTM Recon Error",
                  f"{latest.get('lstm_recon_error', 0):.5f}")
    with col5:
        st.metric("IF Anomaly Score",
                  f"{latest.get('if_score', 0):.3f}")
    with col6:
        st.metric("XGB Fault Prob",
                  f"{latest.get('xgb_score', 0):.3f}")

    st.markdown("---")

    # ── Multi-Class Probability Distribution (Latest Window) ──
    st.markdown("####  XGBoost Multi-Class Probability Distribution (Latest Window)")
    
    # Extract class probabilities if available
    # (This requires storing full probability matrix in predictions.csv — 
    # if not available, we'll show a placeholder)
    prob_cols = [f"xgb_prob_class_{i}" for i in range(9)]
    if all(col in preds_df.columns for col in prob_cols):
        latest_probs = [latest[col] for col in prob_cols]
    else:
        # Fallback: show which class was predicted with 100% confidence
        latest_probs = [0.0] * 9
        latest_probs[xgb_fault_class] = 1.0
    
    fig_prob = go.Figure()
    colors_bar = [FAULT_COLORS.get(i, NAVY) for i in range(9)]
    
    fig_prob.add_trace(go.Bar(
        x=[fault_class_names.get(i, f"Class {i}") for i in range(9)],
        y=latest_probs,
        marker=dict(color=colors_bar),
        text=[f"{p:.1%}" for p in latest_probs],
        textposition="outside",
    ))
    
    fig_prob.update_layout(
        title="Fault Class Probabilities — Latest Window",
        yaxis_title="Probability",
        xaxis_title="Fault Class",
        template="plotly_white",
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")

    # ── Ensemble score + Multi-Class Predictions Over Time ──
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        subplot_titles=[
            "Ensemble Anomaly Score",
            "XGBoost Multi-Class Predictions (0-8)",
            "Individual Model Scores"
        ],
        vertical_spacing=0.08,
        row_heights=[0.3, 0.25, 0.45]
    )

    ts = preds_df["timestamp"]
    fault_ts = preds_df[preds_df["true_label"] == 1]["timestamp"]

    # Shading for fault phase
    if len(fault_ts) > 0:
        for f_fig_row in [1, 2, 3]:
            fig.add_vrect(
                x0=fault_ts.min().to_pydatetime(), 
                x1=fault_ts.max().to_pydatetime(),
                fillcolor="rgba(204,0,0,0.08)", 
                line_width=0,
                row=f_fig_row, col=1
            )
            add_vertical_marker(
                fig,
                x=fault_onset,
                text="Fault Onset" if f_fig_row == 1 else "",
                color=RED,
                width=1.5,
                row=f_fig_row,
                col=1,
            )

    # Row 1: Ensemble score
    fig.add_trace(go.Scatter(
        x=ts, y=preds_df["ensemble_score"],
        name="Ensemble", 
        line=dict(color=NAVY, width=2), 
        fill="tozeroy",
        fillcolor="rgba(27,43,75,0.12)"
    ), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color=RED, line_width=1,
                  annotation_text="Alert Threshold", row=1, col=1)

    # Row 2: Multi-class predictions as color-coded scatter
    fault_class_series = preds_df["xgb_fault_class"].values
    fault_colors_series = [FAULT_COLORS.get(int(c), NAVY) for c in fault_class_series]
    
    fig.add_trace(go.Scatter(
        x=ts,
        y=fault_class_series,
        mode="markers",
        marker=dict(
            color=fault_colors_series,
            size=6,
            line=dict(width=0.5, color="white")
        ),
        name="XGBoost Class",
        text=[fault_class_names.get(int(c), f"Class {c}") for c in fault_class_series],
        hovertemplate="<b>%{text}</b><br>Time: %{x}<extra></extra>",
    ), row=2, col=1)
    
    fig.update_yaxes(
        title_text="Fault Class",
        tickmode="linear",
        tick0=0,
        dtick=1,
        range=[-0.5, 8.5],
        row=2, col=1
    )

    # Row 3: Individual model scores
    fig.add_trace(go.Scatter(
        x=ts, y=preds_df["if_score"],
        name="Isolation Forest", 
        line=dict(color=TEAL, width=1.5)
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=ts, y=preds_df["lstm_score"],
        name="LSTM Autoencoder", 
        line=dict(color=BLUE, width=1.5)
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=ts, y=preds_df["xgb_score"],
        name="XGBoost", 
        line=dict(color=AMBER, width=1.5)
    ), row=3, col=1)

    fig.update_layout(
        height=700, 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified"
    )
    fig.update_yaxes(range=[0, 1.1], row=1, col=1)
    fig.update_yaxes(range=[0, 1.1], row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

    # ── Prediction Detail Table ──
    st.markdown("#### Last 20 Windows — Prediction Detail")
    display_cols = [
        "timestamp", "true_label", "ensemble_pred", "ensemble_score",
        "xgb_fault_class", "fault_class_name", "xgb_score",
        "if_pred", "lstm_pred", "votes", "health_index"
    ]
    available_cols = [c for c in display_cols if c in preds_df.columns]
    recent = preds_df[available_cols].tail(20).copy()

    def colour_row(row):
        if row.get("ensemble_pred", 0) == 1:
            return ["background-color: #FFF0F0"] * len(row)
        return [""] * len(row)

    st.dataframe(
        recent.style.apply(colour_row, axis=1).format(
            {c: "{:.3f}" for c in ["ensemble_score", "xgb_score", "health_index"] 
             if c in recent.columns}
        ),
        use_container_width=True, height=400
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tab 2 — Sensor Trends (UNCHANGED)
# ═════════════════════════════════════════════════════════════════════════════

def tab_sensor_trends(raw_df):
    st.markdown("### 📈 Sensor Trends")

    if raw_df is None:
        st.info("Place the raw DCS Excel file in the data/ folder.")
        return

    fault_onset = pd.Timestamp(FAULT_ONSET_DATE)

    col1, col2 = st.columns([2, 1])
    with col1:
        sensor_type = st.radio("Sensor Type", ["Temperature", "Vibration", "Both"],
                               horizontal=True)
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(raw_df["Date"].min().date(), raw_df["Date"].max().date()),
        )

    if sensor_type == "Temperature":
        sensors_to_show = TEMPERATURE_SENSORS
    elif sensor_type == "Vibration":
        sensors_to_show = VIBRATION_SENSORS
    else:
        sensors_to_show = ALL_SENSORS

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
        (raw_df["Date"] >= pd.Timestamp(date_range[0])) &
        (raw_df["Date"] <= pd.Timestamp(date_range[1]))
    )
    plot_df = raw_df[mask]

    fig = go.Figure()

    for sensor in selected_sensors:
        label = SENSOR_LABELS.get(sensor, sensor)
        color = RED if sensor == FAULT_TEMP_SENSOR else None
        width = 2.5 if sensor == FAULT_TEMP_SENSOR else 1.5
        fig.add_trace(go.Scatter(
            x=plot_df["Date"], y=plot_df[sensor],
            name=label, line=dict(width=width, color=color),
        ))

    if fault_onset >= pd.Timestamp(date_range[0]) and fault_onset <= pd.Timestamp(date_range[1]):
        add_vertical_marker(
            fig,
            x=fault_onset,
            text="Fault Onset (Feb 28)",
            color=RED,
            width=2,
        )

    unit = "°C" if sensor_type == "Temperature" else "µm/s" if sensor_type == "Vibration" else ""
    fig.update_layout(
        title="Sensor Time-Series",
        xaxis_title="Date",
        yaxis_title=unit,
        template="plotly_white",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Statistics — Selected Sensors")
    fault_onset_str = FAULT_ONSET_DATE
    stats_rows = []
    for s in selected_sensors:
        norm_vals  = plot_df[plot_df["Date"] < fault_onset][s].dropna()
        fault_vals = plot_df[plot_df["Date"] >= fault_onset][s].dropna()
        stats_rows.append({
            "Sensor":       SENSOR_LABELS.get(s, s),
            "Normal Mean":  round(norm_vals.mean(), 2) if len(norm_vals) > 0 else "—",
            "Normal Std":   round(norm_vals.std(),  2) if len(norm_vals) > 0 else "—",
            "Fault Mean":   round(fault_vals.mean(), 2) if len(fault_vals) > 0 else "—",
            "Max Value":    round(plot_df[s].max(), 2),
            "Missing":      plot_df[s].isnull().sum(),
        })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Tab 3 — Fault Analysis (UPDATED FOR MULTI-CLASS)
# ═════════════════════════════════════════════════════════════════════════════

def tab_fault_analysis(raw_df, preds_df):
    st.markdown("### 🔴 Fault Analysis — Multi-Class Detection & Physics Insights")

    fault_class_names = load_fault_class_names()

    if preds_df is None:
        st.info("Run training to see fault analysis.")
        return

    # ── Overall Fault Distribution ──
    st.markdown("####  Fault Class Distribution (All Windows)")
    
    fault_counts = preds_df["xgb_fault_class"].value_counts().sort_index()
    
    fig_dist = go.Figure()
    colors_dist = [FAULT_COLORS.get(int(i), NAVY) for i in fault_counts.index]
    
    fig_dist.add_trace(go.Bar(
        x=[fault_class_names.get(int(i), f"Class {i}") for i in fault_counts.index],
        y=fault_counts.values,
        marker=dict(color=colors_dist),
        text=fault_counts.values,
        textposition="outside",
    ))
    
    fig_dist.update_layout(
        title="Detected Fault Classes Across All Windows",
        xaxis_title="Fault Type",
        yaxis_title="Count",
        template="plotly_white",
        height=350,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── Class-Specific Insights ──
    st.markdown("---")
    st.markdown("####  Physics-Based Fault Signatures")
    
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
    
    for class_id in sorted(fault_descriptions.keys()):
        color = FAULT_COLORS.get(class_id, NAVY)
        count = fault_counts.get(class_id, 0)
        st.markdown(f"""
        <div style='border-left:4px solid {color};padding:10px;margin:8px 0;background:white;border-radius:4px;'>
            <span class='fault-badge' style='background:{color};color:white;'>Class {class_id}</span>
            <b>{fault_class_names.get(class_id, f"Class {class_id}")}</b> — {count} windows detected<br>
            <span style='font-size:13px;color:#555;'>{fault_descriptions[class_id]}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── TI0731 Thermal Runaway Deep Dive (Class 1) ──
    if raw_df is not None:
        st.markdown("---")
        st.markdown("#### 🌡️ TI0731 Thermal Runaway — Detailed Progression")
        
        fault_onset = pd.Timestamp(FAULT_ONSET_DATE)
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
        ti_data = raw_df[["Date", FAULT_TEMP_SENSOR]].dropna()

        fig1 = go.Figure()
        normal_mask = ti_data["Date"] < fault_onset
        fault_mask  = ti_data["Date"] >= fault_onset

        fig1.add_trace(go.Scatter(
            x=ti_data[normal_mask]["Date"],
            y=ti_data[normal_mask][FAULT_TEMP_SENSOR],
            name="Normal Phase", line=dict(color=TEAL, width=2),
            fill="tozeroy", fillcolor="rgba(0,135,122,0.08)"
        ))
        fig1.add_trace(go.Scatter(
            x=ti_data[fault_mask]["Date"],
            y=ti_data[fault_mask][FAULT_TEMP_SENSOR],
            name="Fault Phase", line=dict(color=RED, width=2.5),
            fill="tozeroy", fillcolor="rgba(204,0,0,0.10)"
        ))
        add_vertical_marker(fig1, x=fault_onset, text=f"Fault Onset: {FAULT_ONSET_DATE}", color=RED, width=2)
        fig1.add_hline(y=165, line_dash="dot", line_color=AMBER, annotation_text="165°C Alert Threshold")
        
        fig1.update_layout(
            title="TI0731 — Thrust Bearing NDE 1 Temperature (Full Study Period)",
            yaxis_title="Temperature (°C)", 
            xaxis_title="Date",
            template="plotly_white", 
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig1, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# Tab 4 — Model Performance (UNCHANGED)
# ═════════════════════════════════════════════════════════════════════════════

def tab_model_performance(metrics_df):
    st.markdown("###  Model Performance")

    if metrics_df is None:
        st.info("Run training first to see model performance metrics.")
        return

    targets = {
        "recall": TARGET_RECALL,
        "precision": TARGET_PRECISION,
        "roc_auc": TARGET_ROC_AUC,
        "false_positive_rate": TARGET_MAX_FPR,
    }

    if "Ensemble" in metrics_df.index:
        ens = metrics_df.loc["Ensemble"]
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics_display = [
            (c1, "Recall",    "recall",    TARGET_RECALL,    True),
            (c2, "Precision", "precision", TARGET_PRECISION, True),
            (c3, "ROC-AUC",   "roc_auc",  TARGET_ROC_AUC,   True),
            (c4, "F1 Score",  "f1",        0.90,             True),
            (c5, "FPR",       "false_positive_rate", TARGET_MAX_FPR, False),
        ]
        for col, label, key, tgt, higher_is_better in metrics_display:
            val = ens.get(key, None)
            if val is not None and str(val) != "nan":
                val = float(val)
                if higher_is_better:
                    delta = "✓ Target Met" if val >= tgt else f"✗ Need ≥{tgt}"
                    dcolor = "normal" if val >= tgt else "inverse"
                else:
                    delta = "✓ Target Met" if val <= tgt else f"✗ Need ≤{tgt}"
                    dcolor = "normal" if val <= tgt else "inverse"
                col.metric(f"Ensemble {label}", f"{val:.4f}", delta=delta, delta_color=dcolor)

    st.markdown("---")

    models = ["Isolation Forest", "LSTM Autoencoder", "XGBoost", "Ensemble"]
    radar_metrics = ["recall", "precision", "f1"]
    radar_data = {}
    for m in models:
        if m in metrics_df.index:
            vals = []
            for met in radar_metrics:
                v = metrics_df.loc[m, met]
                vals.append(float(v) if str(v) != "nan" else 0.0)
            radar_data[m] = vals

    if radar_data:
        fig_radar = go.Figure()
        colors = [TEAL, BLUE, AMBER, NAVY]
        for i, (model, vals) in enumerate(radar_data.items()):
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_metrics + [radar_metrics[0]],
                fill="toself",
                name=model,
                line=dict(color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)],
                opacity=0.3 if i < len(radar_data)-1 else 0.5,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Model Comparison — Radar Chart",
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(fig_radar, use_container_width=True)
        with col2:
            st.markdown("#### Full Metrics Table")
            display_cols = ["recall", "precision", "f1", "roc_auc",
                            "false_positive_rate", "TP", "FP", "TN", "FN"]
            available = [c for c in display_cols if c in metrics_df.columns]
            st.dataframe(
                metrics_df[available].style.format(
                    {c: "{:.4f}" for c in ["recall", "precision", "f1",
                                            "roc_auc", "false_positive_rate"]}
                ).highlight_max(subset=["recall", "precision", "f1"],
                                color="rgba(0,135,122,0.2)"),
                use_container_width=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# Tab 5 — Feature Importance (UPDATED FOR MULTI-CLASS)
# ═════════════════════════════════════════════════════════════════════════════

def tab_feature_importance(fi_df):
    st.markdown("### 🔬 Feature Importance — XGBoost Multi-Class Classifier")
    st.caption("Trained on **augmented dataset** with 9 fault classes (Normal + 8 synthetic fault types)")

    if fi_df is None:
        st.info("Train the models first to see feature importance.")
        return

    top_n = st.slider("Show top N features", 5, len(fi_df), 20)
    top_df = fi_df.head(top_n)

    def domain_color(feat_name: str) -> str:
        if any(k in feat_name for k in ["vib_rms", "vib_peak", "vib_kurt", "vib_std",
                                          "vib_mean", "vib_crest", "pump_vib", "gb_vib",
                                          "motor_vib", "temp_rms", "temp_mean", "ti0731",
                                          "shutdown"]):
            return TEAL
        if any(k in feat_name for k in ["spectral", "band_energy", "peak_freq",
                                          "entropy", "centroid", "corr", "dominant"]):
            return BLUE
        return AMBER

    colors = [domain_color(f) for f in top_df["feature"]]

    fig = go.Figure(go.Bar(
        x=top_df["importance"],
        y=top_df["feature"],
        orientation="h",
        marker=dict(color=colors),
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
# Tab 6 — RUL Estimation (UNCHANGED)
# ═════════════════════════════════════════════════════════════════════════════

def tab_rul(preds_df, rul_info):
    st.markdown("### ⏱️ Remaining Useful Life (RUL) Estimation")
    st.markdown("*Based on LSTM Autoencoder reconstruction error trend — TI0731 Thrust Bearing NDE 1*")

    if preds_df is None:
        st.info("Run training to generate RUL estimates.")
        return

    if rul_info:
        rul_hours  = rul_info.get("rul_hours", None)
        rul_windows= rul_info.get("rul_windows", None)
        slope      = rul_info.get("trend_slope", 0)
        msg        = rul_info.get("message", "")

        col1, col2, col3 = st.columns(3)
        if rul_hours is not None and rul_hours > 0:
            col1.metric("Estimated RUL", f"{rul_hours:.0f} hours",
                        delta=f"~{rul_hours/24:.1f} days remaining")
        elif rul_hours == 0:
            col1.metric("Estimated RUL", "CRITICAL",
                        delta="⚠ Failure imminent", delta_color="inverse")
        col2.metric("Degradation Slope",  f"{slope:.6f}", delta="Error/window")
        col3.metric("Current Recon Error",
                    f"{rul_info.get('current_error', 0):.5f}")

        if rul_hours is not None and rul_hours < 48:
            st.markdown(f"""
            <div class='fault-alert'>
                <b>⚠ CRITICAL — Less than 48 hours of remaining useful life estimated!</b><br>
                {msg}<br>Immediate maintenance action required on Thrust Bearing NDE 1.
            </div>
            """, unsafe_allow_html=True)

    if "lstm_recon_error" in preds_df.columns:
        fig = go.Figure()
        fault_onset = pd.Timestamp(FAULT_ONSET_DATE)

        normal_mask = preds_df["true_label"] == 0
        fault_mask  = preds_df["true_label"] == 1

        fig.add_trace(go.Scatter(
            x=preds_df[normal_mask]["timestamp"],
            y=preds_df[normal_mask]["lstm_recon_error"],
            name="Normal Phase", line=dict(color=TEAL, width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=preds_df[fault_mask]["timestamp"],
            y=preds_df[fault_mask]["lstm_recon_error"],
            name="Fault Phase", line=dict(color=RED, width=2),
            fill="tozeroy", fillcolor="rgba(204,0,0,0.08)"
        ))

        if fault_mask.any():
            fault_errors = preds_df[fault_mask]["lstm_recon_error"].values
            x_idx = np.arange(len(fault_errors))
            if len(x_idx) > 1:
                slope_plot, intercept = np.polyfit(x_idx, fault_errors, 1)
                trend = slope_plot * x_idx + intercept

                n_future = 20
                x_future = np.arange(len(fault_errors), len(fault_errors) + n_future)
                ts_last  = preds_df[fault_mask]["timestamp"].iloc[-1]
                future_ts = pd.date_range(ts_last, periods=n_future+1, freq="12H")[1:]
                future_err = slope_plot * x_future + intercept

                fault_ts = preds_df[fault_mask]["timestamp"]
                fig.add_trace(go.Scatter(
                    x=list(fault_ts) + list(future_ts),
                    y=list(trend) + list(future_err),
                    name="Degradation Trend + Forecast",
                    line=dict(color=AMBER, width=2, dash="dash"),
                ))

        if rul_info:
            threshold = rul_info.get("critical_threshold", None)
            if threshold:
                fig.add_hline(y=threshold, line_dash="dot", line_color=RED,
                              annotation_text="Critical Threshold (2× anomaly threshold)")

        add_vertical_marker(fig, x=fault_onset, color=RED, text="Fault Onset", width=1.5)
        
        fig.update_layout(
            title="LSTM Reconstruction Error — Bearing Degradation Trajectory",
            xaxis_title="Date", yaxis_title="Reconstruction MSE",
            template="plotly_white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info("ℹ️ RUL estimate is based on linear extrapolation of the LSTM reconstruction "
            "error trend. As more fault-phase data becomes available, the estimate refines.")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 7 — Data Explorer (UPDATED FOR AUGMENTED DATASET)
# ═════════════════════════════════════════════════════════════════════════════

def tab_data_explorer(raw_df, sensor_summary, preds_df):
    st.markdown("### 🗄️ Data Explorer & Augmentation Statistics")

    fault_class_names = load_fault_class_names()

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Original Dataset")
        if raw_df is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Samples",   f"{len(raw_df):,}")
            c2.metric("Sensor Channels", "35")
            c3.metric("Date Range",      "97 days")
            c4.metric("Sample Interval", "15 min")
    
    with col2:
        st.markdown("#### 🧬 Augmented Training Dataset")
        if preds_df is not None:
            fault_counts = preds_df["xgb_fault_class"].value_counts()
            total_windows = len(preds_df)
            real_windows = fault_counts.get(0, 0) + fault_counts.get(1, 0)
            synth_windows = total_windows - real_windows
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Windows", f"{total_windows:,}")
            c2.metric("Real Data", f"{real_windows:,}")
            c3.metric("Synthetic", f"{synth_windows:,}")
            c4.metric("Fault Classes", "9")

    st.markdown("---")

    tab_a, tab_b, tab_c, tab_d = st.tabs([
        "Raw Data Preview", 
        "Sensor Summary", 
        "Correlation Heatmap",
        "Augmentation Breakdown"
    ])

    with tab_a:
        st.markdown("#### Raw DCS Data (first 200 rows)")
        if raw_df is not None:
            display_cols = ["Date"] + [c for c in ALL_SENSORS if c in raw_df.columns]
            st.dataframe(
                raw_df[display_cols].head(200).rename(columns=SENSOR_LABELS),
                use_container_width=True, height=400
            )
            csv = raw_df[display_cols].to_csv(index=False)
            st.download_button("⬇️ Download Full Dataset (CSV)",
                               data=csv, file_name="hp_wipa_dcs_data.csv",
                               mime="text/csv")

    with tab_b:
        if sensor_summary is not None:
            st.markdown("#### Sensor Statistics — Normal vs Fault Phase")
            st.dataframe(sensor_summary.style.format(
                {c: "{:.2f}" for c in sensor_summary.select_dtypes("number").columns}
            ), use_container_width=True)

    with tab_c:
        st.markdown("#### Sensor Correlation Matrix")
        if raw_df is not None:
            sensors_for_corr = [c for c in VIBRATION_SENSORS[:8] + TEMPERATURE_SENSORS[:8]
                                if c in raw_df.columns]
            corr_matrix = raw_df[sensors_for_corr].corr()
            labels = [SENSOR_LABELS.get(c, c) for c in sensors_for_corr]

            fig = px.imshow(
                corr_matrix.values,
                x=labels, y=labels,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Sensor Correlation Heatmap",
                aspect="auto"
            )
            fig.update_layout(height=520, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_d:
        st.markdown("#### 🧬 Physics-Based Data Augmentation Breakdown")
        st.markdown("""
        The training dataset was augmented using **physics-based synthetic fault injection**
        to create labeled examples for fault classes not present in the real data.
        """)
        
        if preds_df is not None:
            fault_counts = preds_df["xgb_fault_class"].value_counts().sort_index()
            
            aug_df = pd.DataFrame({
                "Class": [int(i) for i in fault_counts.index],
                "Fault Type": [fault_class_names.get(int(i), f"Class {i}") for i in fault_counts.index],
                "Count": fault_counts.values,
                "Source": ["Real Data" if i in [0, 1] else "Synthetic" for i in fault_counts.index],
                "Physics Basis": [
                    "Baseline operation" if i == 0 else
                    "TI0731 thermal runaway (confirmed)" if i == 1 else
                    "Bearing BPFO harmonics + impulses" if i == 2 else
                    "1× RPM centrifugal force" if i == 3 else
                    "2× RPM misalignment signature" if i == 4 else
                    "Vane-pass frequency + broadband" if i == 5 else
                    "Seal friction heat + sub-sync vibration" if i == 6 else
                    "GMF + ±1× sidebands" if i == 7 else
                    "BPFI/BPFO bearing defect frequencies" if i == 8 else
                    "Unknown"
                    for i in fault_counts.index
                ]
            })
            
            st.dataframe(
                aug_df.style.apply(
                    lambda row: [
                        f"background-color: {FAULT_COLORS.get(row['Class'], NAVY)}30"
                    ] * len(row),
                    axis=1
                ),
                use_container_width=True
            )
            
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=[fault_class_names.get(int(i), f"Class {i}") for i in fault_counts.index],
                values=fault_counts.values,
                marker=dict(colors=[FAULT_COLORS.get(int(i), NAVY) for i in fault_counts.index]),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
            )])
            
            fig_pie.update_layout(
                title="Training Dataset Composition by Fault Class",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.info("""
            **Augmentation Methodology:**  
            - Classes 2-8 were generated by injecting physics-based sensor signatures into normal windows
            - Each fault type uses domain-specific parameters from Sulzer BB5 datasheets
            - Severity levels: Mild (8%), Moderate (20%), Severe (40%)
            - SMOTE resampling applied during XGBoost training to balance class distribution
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
            AI-Based Multi-Class Fault Detection & RUL Estimation — Sulzer BB5 API 610</p>
    </div>
    """, unsafe_allow_html=True)

    # Load all data
    with st.spinner("Loading data..."):
        preds_df      = load_predictions()
        raw_df        = load_raw_sensor_data()
        metrics_df    = load_metrics()
        fi_df         = load_feature_importance()
        rul_info      = load_rul()
        sensor_summary= load_sensor_summary()

    # Navigation tabs
    tabs = st.tabs([
        "⚙️ Health Monitor",
        "📈 Sensor Trends",
        "🔴 Fault Analysis",
        "📊 Model Performance",
        "🔬 Feature Importance",
        "⏱️ RUL Estimation",
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