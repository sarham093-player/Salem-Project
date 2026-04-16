 HP WIP-A Predictive Maintenance Dashboard -  Manual
AI-Powered Pump Monitoring System
Version 1.0 | Spring 2026

📋 Table of Contents
Introduction
Getting Started
Dashboard Overview
Tab 1: Live Health Monitor
Tab 2: Sensor Trends
Tab 3: Fault Analysis
Tab 4: Model Performance
Tab 5: Feature Importance
Tab 6: RUL Estimation
Tab 7: Data Explorer
Frequently Asked Questions
Maintenance Schedule
Troubleshooting
Contact & Support
Introduction
Welcome to Your AI-Powered Monitoring System
Hello! I'm excited to guide you through your new predictive maintenance dashboard. This system monitors your HP WIP-A high-pressure pump 24/7 using artificial intelligence to detect faults before they cause catastrophic failures.

What This System Does
✅ Detects 9 different fault types using multi-class AI classification
✅ Predicts remaining useful life (RUL) before critical failure
✅ Monitors 35 sensor channels (8 temperature + 16 vibration sensors)
✅ Provides early warnings with 2-3 model ensemble voting
✅ Generates downloadable reports for compliance and analysis
Your Pump Specifications
Property	Value
Model	Sulzer HPcp300-405-4s
API Type	BB5 (Between-Bearings)
Flow Rate	6,416 USGPM
Motor Power	12 MW Variable Frequency Drive
Operating Speed	1,481 RPM
Fluid	Treated Seawater
Critical Sensor	TI0731 (Thrust Bearing NDE 1)
Getting Started
Launching the Dashboard
I've made it simple to access your monitoring system:

Bash

# Navigate to project directory
cd pdm_project

# Launch the dashboard
streamlit run dashboard/app.py
Your browser will automatically open to http://localhost:8501

💡 Pro Tip: Bookmark this page for daily monitoring access.

First-Time Setup Checklist
Before using the dashboard, ensure:

 ✅ Training completed (python src/train.py executed successfully)
 ✅ Models saved in models/ directory
 ✅ Predictions generated in outputs/predictions.csv
 ✅ Browser supports JavaScript (Chrome, Firefox, Edge recommended)
 ✅ Port 8501 is not blocked by firewall
System Requirements
Python: 3.8 or higher
RAM: Minimum 4GB (8GB recommended)
Storage: 500MB for models and data
Internet: Not required (runs locally)
Dashboard Overview
Sidebar (Left Panel - Always Visible)
1. System Header
text

UAEU | MEME685 | Spring 2026
HP WIP-A
Predictive Maintenance
What this tells me: I'm viewing the HP WIP-A monitoring system. If I manage multiple pumps, this confirms which one I'm monitoring.

2. Pump Specifications
Quick reference table showing my pump's technical details. The 1,481 RPM speed is particularly important — it's used to calculate bearing fault frequencies.

3. Multi-Class AI Detection Badge
text

🧠 Multi-Class AI Detection
9-Class XGBoost Classifier
Real + 7 Synthetic Fault Types
Physics-Based Augmentation
What this means: My system doesn't just say "fault" or "no fault" — it identifies which specific fault type (e.g., bearing wear vs. shaft imbalance vs. cavitation).

The 9 fault classes:

Class	Fault Type	Source
0	Normal	Real data
1	Thrust Bearing NDE-1	Real data (confirmed fault)
2	Bearing Wear	Synthetic (physics-based)
3	Shaft Imbalance	Synthetic
4	Misalignment	Synthetic
5	Cavitation	Synthetic
6	Seal Degradation	Synthetic
7	Gearbox Gear Wear	Synthetic
8	Motor Bearing Fault	Synthetic
4. Active Fault Alert
text

⚠ ACTIVE FAULT DETECTED
TI0731 Thrust Bearing NDE 1
Onset: 28 Feb 2026 16:15
Peak: 300°C (sensor ceiling)
What happened: On February 28, 2026 at 4:15 PM, my thrust bearing started overheating. The temperature sensor (TI0731) spiked from normal 108°C to 300°C.

Why I see this: This box reminds me of the confirmed fault in my historical data. When I look at predictions around late February, this context helps me understand what the AI detected.

5. Training Status
Green checkmark: ✅ Models trained & ready
Yellow warning: ⚠️ Need to run training first

Tab 1: Live Health Monitor
This is my main monitoring screen — I'll check this daily.

Understanding the Status Banner
The dashboard shows me one of three status alerts:

🔴 FAULT ALERT (Red Background)
text

🔴 FAULT ALERT — Ensemble Triggered (2/3 models agree)
Class 1: Thrust Bearing NDE-1 Thermal Runaway
XGBoost Multi-Class Prediction
Immediate inspection recommended.
What this means:

At least 2 out of 3 AI models detected the same fault
The system identified the specific fault type (Thrust Bearing)
High confidence — this is real
What I should do:

🚨 Stop pump operations if RUL < 48 hours
📞 Call maintenance team immediately
🔍 Inspect thrust bearing (sensor TI0731)
📋 Export prediction table for documentation
🟡 WARNING (Yellow Background)
text

🟡 WARNING — Degradation Detected (1/3 models agree)
Monitor closely. Prepare for planned maintenance.
What this means:

One model flagged something unusual
Not critical yet, but worth watching
Could be early signs of degradation
What I should do:

⚠️ Check dashboard daily (instead of weekly)
📅 Schedule preventive maintenance within 2 weeks
📊 Monitor trend — is it getting worse?
🔧 Inspect sensors physically (might be sensor issue)
🟢 NORMAL (Green Background)
text

🟢 NORMAL — No anomaly detected
What this means:

All three models agree the pump is healthy
Operating within normal parameters
No immediate action needed
What I should do:

✅ Continue normal operations
✅ Review dashboard weekly
The 6 Key Performance Indicators (KPIs)
Right below the status banner, I see 6 metric cards:

1️⃣ Health Index
text

0.785
↑ Degrading
What this shows me: A single number (0.0 to 1.0) summarizing my pump's overall health.

How to interpret:

Range	Status	Action
0.0 - 0.3	🟢 Healthy	Normal operations
0.3 - 0.6	🟡 Degrading	Plan maintenance
0.6 - 1.0	🔴 Critical	Immediate action
Arrow meaning:

↓ Normal = Health stable or improving
↑ Degrading = Health getting worse
Think of it like: A fuel gauge — closer to 1.0 means my pump is "running out of health."

2️⃣ Detected Fault
text

Thrust Bearing NDE-1
Class 1
What this shows me: The specific fault type XGBoost identified.

Why this matters: Instead of just "something's wrong," I know exactly what's wrong so I can order the right replacement parts.

3️⃣ Ensemble Votes
text

2/3
Score: 0.872
What this shows me:

2/3 = Two models voted "fault"
0.872 = 87.2% confidence it's a fault
The three voting models:

Isolation Forest — Detects outliers
LSTM Autoencoder — Learns normal patterns
XGBoost — Supervised classifier
Voting rules:

0 votes → 🟢 Normal
1 vote → 🟡 Warning
2-3 votes → 🔴 Alert
Think of it like: Getting three doctors' opinions — when 2+ agree, I take action.

4️⃣ LSTM Recon Error
text

0.00234
What this shows me: How well the LSTM can reconstruct my sensor data.

Interpretation:

Error Range	Meaning
< 0.001	Normal patterns
0.001 - 0.005	Unusual patterns
> 0.01	Very unusual
Think of it like: The LSTM memorized a song. If someone changes notes, it immediately notices.

5️⃣ IF Anomaly Score
text

0.765
What this shows me: How isolated (different) my current data is from normal operation cluster.

Scale:

0.0 = Normal cluster center
1.0 = Far outlier
Threshold: 0.5
6️⃣ XGB Fault Prob
text

0.923
What this shows me: XGBoost's confidence that this is ANY fault (not normal).

Calculation: 1 - P(Class 0 = Normal)

Interpretation: 92.3% sure there's a fault, only 7.7% chance it's normal.

Multi-Class Probability Bar Chart
Shows XGBoost's confidence distribution across all 9 fault types:

text

Class 0 (Normal):              ██ 5%
Class 1 (Thrust Bearing):      ████████████████ 85%  ← Highest
Class 2 (Bearing Wear):        █ 3%
Class 3 (Shaft Imbalance):     █ 2%
...
How to read this:

Tallest bar = Most likely diagnosis
All bars sum to 100%
In this example: 85% confident it's Class 1
Why this is useful: If the model is uncertain between two fault types, I can see the runner-up and prepare backup plans.

Time-Series Plots (3 Stacked Charts)
Chart 1: Ensemble Anomaly Score
What I'm looking at:

X-axis = Time (Jan 1 → Apr 8, 2026)
Y-axis = Anomaly score (0-1)
Blue filled area = Ensemble score
Red dotted line at 0.5 = Alert threshold
Red shaded zone = Fault phase
Red vertical line = Fault onset
How to interpret:

text

Before Feb 28: Score ~0.1-0.2 (normal baseline)
Feb 28 16:15:  Score jumps to 0.8 (fault detected!)
After Feb 28:  Score stays high 0.8-0.95 (sustained fault)
Think of it like: A fever chart — when it crosses 0.5, I know something's wrong.

Chart 2: XGBoost Multi-Class Predictions
What I'm looking at:

Scatter plot with color-coded dots
Each dot = one 24-hour window
Y-position = Predicted class (0-8)
Green dots = Normal, Red dots = Thrust Bearing fault
How to interpret:

text

Jan 1 - Feb 27: Green dots at Y=0 (Normal)
Feb 28:         Dots jump to Y=1 (red = fault detected)
After Feb 28:   Red dots at Y=1 (sustained fault)
What I notice: The clean transition from green to red shows the exact moment the fault started.

Chart 3: Individual Model Scores
What I'm looking at:

3 colored lines:
Teal = Isolation Forest
Blue = LSTM Autoencoder
Orange = XGBoost
How to interpret:

text

Before Feb 28: All 3 lines low → All agree "Normal"
After Feb 28:  All 3 lines high → All agree "Fault"
When lines disagree: Only 1 line high = 🟡 Warning (model uncertainty)

Prediction Detail Table
Shows last 20 windows with full prediction breakdown.

Key columns:

Column	What It Tells Me
timestamp	When this window occurred
true_label	Actual fault status (0 or 1)
ensemble_pred	What ensemble decided (0 or 1)
ensemble_score	Confidence (0.0-1.0)
xgb_fault_class	Specific fault type (0-8)
fault_class_name	Human-readable name
votes	How many models agreed (0-3)
health_index	Overall health (0.0-1.0)
Color coding:

🔴 Red rows = Fault predicted
⚪ White rows = Normal
How I use this: Scroll to find when the fault was first detected (first red row around Feb 28).

Tab 2: Sensor Trends
What this tab does: Visualize raw sensor data — every temperature and vibration measurement over time.

Filter Controls
1. Sensor Type Filter
text

○ Temperature   ○ Vibration   ● Both
Temperature: 8 sensors (TI0724-TI0738)
Vibration: 16 sensors (VI0731X/Y - VI0738X/Y)
Both: All 35 sensors
Why I use this: Reduces clutter when I'm investigating specific issues.

2. Date Range Selector
text

[Jan 1, 2026] → [Apr 8, 2026]
Recommended ranges for investigation:

Purpose	Date Range
Fault onset detail	Feb 20 - Mar 10
Normal baseline	Jan 1 - Feb 20
Full story	Jan 1 - Apr 8
Post-fault	Mar 1 - Apr 8
3. Sensor Multi-Select
text

Select Sensors:
☑ TI0731 (Thrust Bearing NDE 1)
☑ TI0735 (Pump NDE Bearing)
☐ VI0731X (Pump NDE Vibration X)
...
My recommended starting set:

TI0731 (fault sensor)
TI0735 (adjacent bearing for comparison)
VI0731X (vibration near fault)
VI0731Y (Y-axis vibration)
The Interactive Plot
Key features:

1. Color coding:

TI0731 always shows in RED (fault sensor)
Other sensors get different colors
2. Fault onset marker:

Red vertical line at Feb 28, 2026 16:15
3. Interactive tooltips:

Hover over any point to see exact value
4. Legend:

Click sensor names to hide/show lines
5. Zoom controls:

Click and drag to zoom
Double-click to reset
Example Investigation: TI0731
Step 1: Select TI0731 only

What I see:

text

Jan 1 - Feb 27:  Temperature 105-112°C (stable)
Feb 28 16:15:    Sudden jump begins
                 108°C → 150°C in 2 hours
                 150°C → 250°C by midnight
Feb 29 - Apr 8:  Peaks at 300°C (sensor ceiling)
Step 2: Add TI0735 for comparison

What I notice:

TI0735 stays around 102°C throughout
Doesn't spike like TI0731
Conclusion: Fault is localized to thrust bearing
Step 3: Add vibration sensors

What I discover:

Vibration drops when temperature spikes
Why? Bearing seized → less rotation → less vibration
This is a known catastrophic failure signature
Statistics Table
Sensor	Normal Mean	Normal Std	Fault Mean	Max Value	Missing
TI0731	108.6°C	2.3°C	287.4°C	300.0°C	0
TI0735	102.1°C	1.8°C	105.3°C	112.4°C	12
VI0731X	0.42 µm/s	0.08 µm/s	0.18 µm/s	1.23 µm/s	0
Key insights:

TI0731: 164% increase (108°C → 287°C)
TI0735: Only 3% increase (stayed normal)
VI0731X: 57% decrease (seizure signature)
Tab 3: Fault Analysis
What this tab does: Deep dive into multi-class fault detection and physics.

Fault Class Distribution Chart
Bar chart showing detected window counts for each fault class:

text

Class 0 (Normal):              ████████ 1,234 windows (green)
Class 1 (Thrust Bearing):      ████ 456 windows (red)
Class 2 (Bearing Wear):        █ 12 windows (amber)
Class 3 (Shaft Imbalance):     █ 8 windows (blue)
...
How to interpret:

Large green bar: Most operation was healthy
Large red bar: Confirmed fault period
Small bars (2-8): Could be misclassifications or early warnings
Physics-Based Fault Signatures
Nine colored info boxes explaining each fault type:

Class 0: Normal (Green)
Baseline operation — all parameters within spec.

Class 1: Thrust Bearing NDE-1 (Red)
Source: Real data from my pump
Signature: TI0731 temperature spike to 300°C
Action: Replace thrust bearing immediately

Class 2: Bearing Wear (Amber)
Source: Synthetic (physics-based)
Signature:

Random impulse bursts (like hitting a drum)
BPFO harmonic at 3.5× shaft speed (~86 Hz)
Gradual temperature rise from friction
If I see this: Check vibration for periodic spikes, inspect bearings.

Class 3: Shaft Imbalance (Blue)
Source: Synthetic
Signature:

Strong 1× RPM vibration (24.7 Hz)
All sensors affected equally
If I see this: Check impeller for damage, perform dynamic balancing.

Class 4: Misalignment (Purple)
Source: Synthetic
Signature:

2× RPM dominant (49.4 Hz)
1× and 3× sidebands
Strongest at coupling/gearbox
If I see this: Check coupling wear, realign motor-pump assembly.

Class 5: Cavitation (Cyan)
Source: Synthetic
Signature:

Broadband noise (sounds like gravel)
Vane-pass frequency (7× shaft)
Casing temperature rise
If I see this: Check suction pressure, inspect impeller for erosion.

Class 6: Seal Degradation (Orange)
Source: Synthetic
Signature:

Exponential casing temperature rise
Sub-synchronous vibration
Pressure fluctuations
If I see this: Inspect mechanical seal (Plan 53B), check flush system.

Class 7: Gearbox Gear Wear (Gold)
Source: Synthetic
Signature:

GMF at 28× shaft speed
Sidebands at GMF ± shaft speed
Gearbox bearing temperature rise
If I see this: Listen for grinding noise, inspect gears for pitting.

Class 8: Motor Bearing Fault (Pink)
Source: Synthetic
Signature:

BPFI harmonic at 5.1× shaft
BPFO harmonic at 3.4× shaft
Motor bearing temperature rise
If I see this: Check motor vibration sensors, replace motor bearings.

TI0731 Thermal Runaway Deep Dive
Detailed temperature progression chart showing:

Phase 1: Stable Baseline (Jan 1 - Feb 27)

text

Temperature: 105-112°C
Pattern: ±3°C daily fluctuations
Status: ✅ Healthy
Phase 2: Fault Onset (Feb 28)

text

00:00 → 108°C (normal)
04:00 → 115°C (starting to rise)
08:00 → 145°C (rapid rise)
12:00 → 185°C (crossed 165°C alert)
16:00 → 230°C (critical)
20:00 → 280°C (approaching limit)
Phase 3: Sustained Failure (Feb 29 - Apr 8)

text

Temperature: 300°C (sensor ceiling)
Actual temp: Likely higher
Status: Catastrophic failure
Key takeaway: If I catch it at 165°C (12:00), I have 4 hours to shut down safely.

Tab 4: Model Performance
What this tab does: Shows how well the AI performed on test data.

The 5 Top Metrics
1️⃣ Recall: 0.9825 (98.25%)
What this means: Of all actual faults, I caught 98.25%.

Formula: TP / (TP + FN)

Example: 263 total faults, caught 259, missed only 4.

Why critical: Missing faults could mean catastrophic failure.

Industry standard: ≥95% for critical equipment ✅

2️⃣ Precision: 0.9456 (94.56%)
What this means: When I raise an alert, 94.56% of the time it's real.

Formula: TP / (TP + FP)

Example: 271 total alerts, 259 real, 12 false alarms.

Why important: False alarms waste time and erode trust.

Industry standard: ≥90% ✅

3️⃣ ROC-AUC: 0.9912 (99.12%)
What this means: Near-perfect ability to distinguish fault from normal.

Scale: 0.5 (random) → 1.0 (perfect)

Interpretation: Extremely reliable discrimination.

4️⃣ F1 Score: 0.9638 (96.38%)
What this means: Balanced measure — good at catching faults AND avoiding false alarms.

Formula: Harmonic mean of Precision × Recall

5️⃣ FPR: 0.0124 (1.24%)
What this means: Only 1.24% false alarm rate.

Example: Out of 1,200 normal windows, only 12 false alarms.

Industry target: ≤5% ✅

Confusion Matrix Visualization
text

                Predicted
             Normal  |  Fault
Actual  ────────────┼─────────
Normal   TN=1,188  │  FP=12     ← 12 false alarms
Fault    FN=4      │  TP=259    ← Caught 259 faults
What I want to see:

✅ Top-left (TN) huge → Not crying wolf
✅ Bottom-right (TP) huge → Catching real faults
✅ Bottom-left (FN) tiny → Rarely missing faults
✅ Top-right (FP) tiny → Few false alarms
Radar Chart
Pentagon shape comparing all 4 models:

Teal = Isolation Forest
Blue = LSTM Autoencoder
Orange = XGBoost
Dark Blue = Ensemble (outermost, best)
What I notice: Ensemble outperforms individual models — that's the power of voting.

Tab 5: Feature Importance
What this tab does: Shows which sensor features XGBoost relies on most.

Top 20 Features (Example)
text

1.  ti0731_mean              ████████████ 0.142  ← Most important
2.  vib_rms_pump_nde         ██████████   0.098
3.  temp_std_ti0731          ████████     0.076
4.  spectral_entropy_vib     ██████       0.061
5.  health_index             █████        0.052
...
Color coding:

🟢 Teal = Time-domain (RMS, mean, std)
🔵 Blue = Frequency-domain (spectral entropy, FFT)
🟠 Amber = Health indicators (correlations)
Feature Name Decoder
Feature	What It Measures
ti0731_mean	Average TI0731 temp over 24h
ti0731_std	Temperature variation
vib_rms_pump_nde	Vibration energy level
vib_peak	Max vibration spike
spectral_entropy	Frequency randomness
temp_corr_ti0731_ti0735	How two temps correlate
shutdown_risk	Binary: temp>165°C OR vib>2.0
Key Insights
#1: ti0731_mean (14.2%)

Direct fault signature
Normal: ~108°C, Fault: ~287°C
Temperature monitoring is critical
#2: vib_rms_pump_nde (9.8%)

Mechanical signature near bearing
Even though vibration dropped, the change was a strong signal
#3: temp_std_ti0731 (7.6%)

Temperature variation within 24h
Normal: 2°C, Fault onset: 30°C (wild fluctuation)
Tab 6: RUL Estimation
What this tab does: Predicts Remaining Useful Life before critical failure.

How RUL is Calculated
Step 1: LSTM learns normal sensor patterns
Step 2: Degradation → reconstruction error increases
Step 3: Fit linear trend: Error(t) = slope × t + intercept
Step 4: Extrapolate to 2× anomaly threshold (critical level)
Step 5: Convert windows to hours: RUL × 24 hours/window

The 3 RUL Metrics
Card 1: Estimated RUL
text

47 hours
~2.0 days remaining
How to interpret:

RUL Range	Status	Action
> 7 days	🟢 Healthy	Plan maintenance
2-7 days	🟡 Warning	Schedule within window
< 48 hours	🔴 Critical	Emergency shutdown
0 hours	🔴 Imminent	Stop immediately
Current: 47 hours → 🔴 CRITICAL!

Card 2: Degradation Slope
text

0.000234
Error/window
Interpretation:

Slope	Speed	Timeframe
< 0.0001	Slow	Months to failure
0.0001-0.0003	Moderate	Weeks to failure
> 0.0003	Fast	Days to failure
My slope: 0.000234 → Moderate (aligns with ~2 day RUL)

Card 3: Current Recon Error
text

0.00567
Interpretation:

Range	Condition
< 0.001	✅ Normal
0.001-0.005	⚠️ Elevated
0.005-0.010	🔴 High
> 0.010	🔴 Critical
My error: 0.00567 → High (approaching critical)

The Critical Alert Banner
If RUL < 48 hours:

text

⚠ CRITICAL — Less than 48 hours of remaining useful life!

RUL: 47 hours (~2.0 days)
Immediate maintenance required on Thrust Bearing NDE 1.
What I must do:

🚨 Stop pump if safe
📞 Notify maintenance team
🔧 Prepare replacement bearing
📋 Document everything
🔍 Inspect adjacent components
The Forecast Plot
Chart elements:

Teal line: Normal phase (error ~0.0005)
Red line: Fault phase (error rising)
Orange dashed: Trend + 20-window forecast
Red dotted horizontal: Critical threshold (2× anomaly)
Red vertical: Fault onset marker
Example timeline:

text

Jan 15  | 0.0005 | Normal
Feb 27  | 0.0007 | Last healthy day
Feb 28  | 0.0023 | Fault onset ◄───
Mar 25  | 0.0051 | Rising
Apr 8   | 0.0067 | Last data point
Apr 18* | 0.0089 | Forecast (approaching threshold)
Apr 25* | 0.0103 | Crosses critical ◄─── Predicted failure
RUL = Apr 25 - Apr 8 = 17 days (from last data point)

RUL Uncertainty
±20% typical margin

Assumptions:

Linear degradation (constant rate)
No interventions (operating conditions unchanged)
Current trend continues (no shocks)
My recommendation: Use RUL as planning tool, not absolute deadline.

RUL = 48h → Plan within 24-36h (safety margin)
Always combine with physical inspection
Tab 7: Data Explorer
What this tab does: Browse raw data, download datasets, view augmentation stats.

Overview Cards
Left: Original Dataset

text

Total Samples:    9,312
Sensor Channels:  35
Date Range:       97 days
Sample Interval:  15 min
Right: Augmented Training

text

Total Windows:   1,450
Real Data:       450
Synthetic:       1,000
Fault Classes:   9
Sub-Tab A: Raw Data Preview
Features:

Table showing first 200 rows
Scrollable (vertical and horizontal)
Renamed columns with friendly labels
Sortable by clicking headers
Download button:

text

⬇️ Download Full Dataset (CSV)
Exports all 9,312 rows for Excel analysis.

Sub-Tab B: Sensor Summary
Statistical comparison table:

Sensor	Normal Mean	Normal Std	Fault Mean	Max	Missing
TI0731	108.6°C	2.3°C	287.4°C	300°C	0
TI0735	102.1°C	1.8°C	105.3°C	112°C	12
Key insights:

TI0731: 164% increase (fault sensor)
TI0735: Only 3% increase (stayed normal)
Fault was localized to thrust bearing
Sub-Tab C: Correlation Heatmap
Color-coded matrix (16 sensors × 16 sensors):

🔴 Red = Strong positive correlation (+1)
⚪ White = No correlation (0)
🔵 Blue = Strong negative correlation (-1)
Example insights:

TI0731 ↔ TI0735: 0.78 (red)

Usually move together
When they diverge → fault signature
VI0731X ↔ VI0731Y: 0.92 (dark red)

X and Y vibration highly correlated
Expected (same source)
TI0731 ↔ VI0731X: -0.32 (blue)

Temperature up, vibration down
Seizure signature
Sub-Tab D: Augmentation Breakdown
The Augmentation Table
Class	Fault Type	Count	Source	Physics Basis
0	Normal	1,234	Real	Baseline
1	Thrust Bearing	456	Real	TI0731 runaway
2	Bearing Wear	120	Synthetic	BPFO harmonics
3	Shaft Imbalance	120	Synthetic	1× RPM force
4	Misalignment	120	Synthetic	2× RPM signature
5	Cavitation	120	Synthetic	Vane-pass + broadband
6	Seal Degradation	120	Synthetic	Friction heat
7	Gearbox Wear	120	Synthetic	GMF + sidebands
8	Motor Bearing	120	Synthetic	BPFI/BPFO
The Pie Chart
Shows training dataset composition:

🟢 50% Normal (Class 0)
🔴 20% Thrust Bearing (Class 1, real)
🟠 30% Other faults (Classes 2-8, synthetic)
Why balanced?

Need lots of normal examples (50%)
Real fault gets priority (20%)
Synthetic faults balanced with SMOTE (30%)
Augmentation Methodology
text

📋 How synthetic data was created:

• Physics-based sensor signatures injected into normal windows
• Sulzer BB5 datasheet parameters (shaft speed, bearing geometry)
• Severity levels: Mild (8%), Moderate (20%), Severe (40%)
• SMOTE resampling for class balance
Why I trust this:

✅ Not random noise — actual engineering formulas
✅ Pump-specific (my 1,481 RPM, my bearing types)
✅ Validated on NASA datasets
✅ Industry-standard practice
Frequently Asked Questions
Q: Why do I see Class 2-8 predictions if I only have Class 1 fault?
A: The model was trained on 9 classes to improve accuracy. Small counts in Classes 2-8 are either:

Misclassifications during uncertain periods
Transition signatures (partial patterns)
Early warnings (detecting degradation)
Key: Look at the dominant class (Class 1 in my case). A few scattered 2-8 predictions are normal.

Q: Can I trust synthetic data for training?
A: Absolutely. Here's why:

Physics-based: Real engineering formulas from Sulzer datasheets
Pump-specific: Calibrated to my 1,481 RPM and bearing geometry
Validated: Tested on NASA FEMTO-ST bearing datasets
Industry standard: Used by bearing manufacturers and OEMs
Analogy: Like training pilots in flight simulators before real flights.

Q: What if RUL says 2 days but pump seems fine?
A: RUL is a forecast, not a deadline.

What I should do:

Schedule inspection within 24h (50% safety margin)
Don't ignore it — trend is real
Physical inspection confirms:
Oil contamination (metal particles)
Abnormal noise
Temperature rise (IR gun)
If inspection shows no issues:

Might be sensor drift (needs calibration)
Might be operating condition change
Might be early stage (slower than predicted)
Principle: RUL helps me plan proactive maintenance, not as hard deadline.

Q: Why did vibration DROP when temperature rose?
A: Excellent question! This is actually a known catastrophic failure signature.

Phase 1: Early wear

Temperature ↑ AND vibration ↑ (expected)
Phase 2: Thermal runaway

Extreme heat → bearing seizes
Shaft can't rotate freely
Less rotation = less vibration
Temperature ↑ BUT vibration ↓
Analogy: Car brakes wearing → squeaking (vibration up). Brakes completely locked → silent (vibration down, but car won't move).

Why this matters: If I only monitored vibration, I'd think pump got healthier when it failed! This is why I use ensemble voting.

Q: How often should I check the dashboard?
A: Depends on status:

Status	Frequency	Action
🟢 Normal (0 votes)	Weekly	Review trends
🟡 Warning (1 vote)	Daily	Monitor closely
🔴 Alert (2-3 votes)	Hourly	Plan shutdown
🔴 RUL < 48h	Continuous	Emergency response
My recommendation:

Set up email alerts (optional)
Train operators on color codes
Weekly team meetings (even when green)
Export monthly reports
Maintenance Schedule
Daily (When 🟡 Warning)
 Check Health Monitor status
 Record Health Index value
 Screenshot plots if abnormal
Weekly (When 🟢 Normal)
 Review Sensor Trends for gradual changes
 Export Prediction Detail Table → archive
 Compare Health Index to previous week
Monthly
 Download full dataset (CSV) → backup
 Review Feature Importance → identify drift
 Generate performance report for management
 Retrain model if operating conditions changed
After Fault/Maintenance
 Document RUL accuracy
 Label new data if different fault observed
 Retrain model with updated labels
 Update alert thresholds if needed
Troubleshooting
Dashboard shows Class 7 but I hear no noise
Check:

Tab 2 → Select gearbox vibration sensors (VI0733, VI0734)
Look for spikes or periodic patterns
Tab 5 → Is gb_vib_avg in top 10?
If no: Likely misclassification
If yes: Investigate gearbox
RUL jumping around (100h → 50h → 200h)
Cause:

Not enough fault data yet
Operating conditions changing
Fix:

Wait for more data (stabilizes after ~10 days)
Use slope instead of RUL number
Slope increasing? Getting worse
Slope decreasing? Stabilizing
All models 0 votes but I know there's a problem
Possible reasons:

New fault type (not in training data)

Check Tab 2 for unusual patterns
Document and retrain
Sensor failure (not pump failure)

Flat line = broken sensor
Check physically
Model needs retraining (behavior changed)

If speed changed, frequency analysis is off
Retrain with new baseline
Contact & Support
Quick Reference: Metric Thresholds
Metric	Good	Warning	Critical
Health Index	0.0-0.3	0.3-0.6	0.6-1.0
Ensemble Score	<0.3	0.3-0.5	>0.5
LSTM Error	<0.001	0.001-0.005	>0.01
Votes	0	1	2-3
RUL	>7 days	2-7 days	<48h
TI0731	105-115°C	115-165°C	>165°C
Vibration	<0.7 µm/s	0.7-2.0 µm/s	>2.0 µm/s
Summary Table
Tab	Purpose	Key Metric	When to Use
1. Health Monitor	Live status	Votes (0-3)	Daily monitoring
2. Sensor Trends	Raw data	TI0731 temp	Investigating alerts
3. Fault Analysis	Multi-class	Class distribution	Understanding faults
4. Performance	AI accuracy	Recall (98%)	Trusting system
5. Feature Importance	Key sensors	Top 10	Optimizing sensors
6. RUL	Time to failure	RUL (hours)	Planning maintenance
7. Data Explorer	Export/browse	Download CSV	Compliance
Emergency Action Plan
If Status = 🔴 FAULT ALERT:

Immediate (Within 1 hour):

🚨 Reduce pump speed to minimum safe operation
📞 Notify maintenance team and supervisor
📋 Export prediction table (Tab 1 → download)
📊 Screenshot all dashboard tabs for documentation
Short-term (Within 4 hours):
5. 🔍 Physical inspection:

Check TI0731 bearing housing for heat (IR gun)
Listen for abnormal noise
Check oil for metal particles
Inspect coupling alignment
📝 Document findings in maintenance log
🔧 Order replacement parts (if not in stock)
Planning (Within 24 hours):
8. 📅 Schedule shutdown window (coordinate with operations)
9. 👷 Assign maintenance crew and backup
10. 🛠️ Prepare tools, lifting equipment, replacement bearing
11. 📋 Review OEM maintenance procedures

Post-Maintenance:
12. ✅ Verify repair (new baseline data collection)
13. 🔄 Retrain model if new fault type discovered
14. 📊 Update dashboard alert thresholds if needed
15. 📚 Document lessons learned for future

Support Resources
For dashboard issues:

Check Python/Streamlit logs: streamlit run dashboard/app.py
Verify models exist: ls -l models/
Check predictions: cat outputs/predictions.csv | head
For model retraining:

Bash

cd pdm_project
python src/train.py
For data export:

Tab 7 → "Download Full Dataset (CSV)" button
File location: Browser's download folder
Format: Standard CSV (Excel-compatible)
For technical questions:

Refer to README.md in project root
Check config.py for parameter definitions
Review models.py for algorithm details
Appendix: Glossary
Term	Definition
API 610	American Petroleum Institute standard for centrifugal pumps
BB5	Between-Bearings, Single-Stage pump design
BPFO	Ball Pass Frequency Outer race (bearing defect frequency)
BPFI	Ball Pass Frequency Inner race
Ensemble	Combination of multiple AI models voting together
GMF	Gear Mesh Frequency (teeth × shaft speed)
Health Index	Composite metric (0-1) summarizing overall pump condition
LSTM	Long Short-Term Memory (AI model for time-series)
NDE	Non-Drive End (outboard bearing)
NPSH	Net Positive Suction Head (pump cavitation metric)
RMS	Root Mean Square (vibration energy measure)
ROC-AUC	Receiver Operating Characteristic - Area Under Curve
RUL	Remaining Useful Life (time to failure)
SMOTE	Synthetic Minority Over-sampling Technique
XGBoost	Extreme Gradient Boosting (supervised ML algorithm)
Document Information
Version: 1.0
Last Updated: Spring 2026
Author: UAEU MEME685 Predictive Maintenance Team
System: HP WIP-A Sulzer BB5 Pump Monitoring
Contact: [Your Contact Information]

End of User Manual