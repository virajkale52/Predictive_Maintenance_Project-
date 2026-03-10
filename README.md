# Predictive_Maintenance_Project-
# ⚡ DYNAMO — Motor Intelligence Platform
### by PredictX · Predictive Maintenance Intelligence

![Python](https://img.shields.io/badge/Python-3.9+-dc2626?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-f97316?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-fbbf24?style=flat-square&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-06b6d4?style=flat-square&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## 🔴 Live Demo

👉 **[Launch DYNAMO on Streamlit Cloud](https://predictx.streamlit.app/)**

> Upload the sample CSV from the sidebar to try it instantly — no setup required.

---

## 📌 What is DYNAMO?

**DYNAMO** is an industrial-grade predictive maintenance platform for electric motors, built under the **PredictX** platform brand.

It ingests motor sensor data (CSV), trains a logistic regression fault detector, and delivers:
- Real-time fault probability
- Remaining Useful Life (RUL) estimation
- Tipping point analysis per parameter
- Monte Carlo risk simulation
- Auto-generated IEC 60034 maintenance work orders

Built as a learning project — **v1 uses Logistic Regression**. Future versions will upgrade to Random Forest, XGBoost and LSTM for time-series fault prediction.

---

## 🖥️ App Sections

| Slide | Name | Description |
|-------|------|-------------|
| S01 | Motor Data Ingestion | Upload your motor sensor CSV |
| S02 | Nameplate Readings   | Dataset overview — cycles, channels, fault rate |
| S03 | Live Winding Stream  | Simulated live parameter feed with anomaly detection |
| S04 | Correlation Matrix   | Heatmap of all sensor parameter correlations |
| S05 | Model Configuration  | Select features and fault label column |
| S06 | Bearing Trend Monitor | Time-series trend with fault event markers |
| S07 | Training Engine      | Train logistic regression with configurable test split |
| S08 | Diagnostic Metrics   | Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC |
| S09 | Fault Driver Ranking | Feature importance from model coefficients |
| S10 | RUL Estimation       | Degradation curve and remaining useful life |
| S11 | Live Fault Scan      | Real-time prediction via sliders or manual entry |
| S12 | Machine Health Report | Full motor scorecard with maintenance verdict |
| S13 | Scenario Engine      | Tipping points · Monte Carlo · Work Order Generator |
| S14 | AI Diagnostic Assistant | Claude-powered motor diagnostic chat |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/dynamo-predictx.git
cd dynamo-predictx
```

### 2. Install dependencies
```bash
pip install streamlit scikit-learn plotly pandas numpy requests streamlit-lottie
```

### 3. Run the app
```bash
streamlit run pm_1.py
```

### 4. Upload data
Use the **sample CSV** available in the sidebar, or upload your own motor sensor dataset.

---

## 📂 Sample CSV Format

Your CSV needs numeric sensor columns + one binary fault label column.

```csv
cycle_no, winding_temp_C, vibration_mm_s, stator_current_A, shaft_rpm, torque_Nm, insulation_MOhm, failure
1, 72, 1.8, 36, 1490, 175, 210, 0
2, 108, 8.5, 64, 2960, 365, 42, 1
...
```

| Column             | Unit  | Description |
|--------------------|-------
|-------------|
| `winding_temp_C`   | °C   | Stator winding temperature |
| `vibration_mm_s`   | mm/s | Bearing vibration RMS (ISO 10816) |
| `stator_current_A` | A    | Phase current draw |
| `shaft_rpm`        | RPM  | Rotor shaft speed |
| `torque_Nm`        | Nm   | Output torque |
| `insulation_MOhm`  | MΩ   | Winding insulation resistance |
| `failure`          | 0 / 1 | 0 = Healthy · 1 = Fault |

---

## 🧠 ML Pipeline

```
CSV Upload → Feature Selection → Train/Test Split
     ↓
StandardScaler → LogisticRegression → Predict Proba
     ↓
Fault Probability → RUL Estimation → Health Report
     ↓
Tipping Points → Monte Carlo (5,000 runs) → Work Order
```

**Current model:** Logistic Regression (`scikit-learn`)  
**Planned upgrades:** Random Forest · XGBoost · LSTM (time-series)

---

## 🏭 Standards Compliance

| Standard | Scope |
|----------|-------|
| IEC 60034 | Rotating electrical machines |
| NEMA MG-1 | Motor and generator construction |
| ISO 10816 | Vibration severity evaluation |
| IEEE 43 | Insulation resistance testing |
| IEC 60085 | Insulation thermal classification |

---

## 🛠️ Tech Stack

- **Frontend** — Streamlit + custom CSS (Bebas Neue · Teko · Courier Prime)
- **ML** — Scikit-learn (LogisticRegression, StandardScaler)
- **Visualisation** — Plotly (charts, gauge, heatmap, ROC)
- **Animations** — Pure CSS (motor rings, orbiting dots, piston arms)
- **AI Chat** — Anthropic Claude API (S14)
- **Data** — Pandas · NumPy

---

## 📊 Features Breakdown

### 🎯 Tipping Point Finder
Sweeps each parameter across its full range while holding others at mean. Finds the exact sensor value where fault probability crosses **40% (CAUTION)** and **70% (CRITICAL)**.

### 🎲 Monte Carlo Simulation
Runs **5,000 randomised scenarios** with Gaussian noise (±1 std dev) on all parameters simultaneously. Returns healthy / caution / critical distribution and P50/P95 risk percentiles.

### 📋 Work Order Generator
Auto-generates a structured **IEC 60034 / NEMA MG-1 maintenance work order** with prioritised tasks derived from fault probability, tipping point proximity and Monte Carlo risk. Downloadable as `.TXT` and `.CSV`.

---

## 🗺️ Roadmap

- [x] v1 — Logistic Regression fault detector
- [x] Tipping point analysis
- [x] Monte Carlo simulation
- [x] IEC 60034 work order generator
- [x] AI diagnostic assistant (Claude API)
- [ ] v2 — Random Forest + XGBoost
- [ ] v3 — LSTM time-series fault prediction
- [ ] Real-time MODBUS TCP / OPC-UA data ingestion
- [ ] Multi-motor fleet dashboard
- [ ] PredictX — next product (pump cavitation / gearbox wear)

---

## 📁 Project Structure

```
dynamo-predictx/
│
├── pm_1.py              # Main Streamlit application
├── README.md            # This file
└── dynamo_sample.csv    # Sample motor dataset (optional)
```

---

## 👤 Author
Viraj Vijay Kale

> *This is my first ML project. DYNAMO is v1 of the PredictX platform — designed to grow.*



---

<div align="center">
  <sub>DYNAMO · PredictX · IEC 60034 · NEMA MG-1 · ISO 10816 · Streamlit · Scikit-learn · Plotly</sub>
</div>
