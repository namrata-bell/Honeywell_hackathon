# 🛡️ AI-Powered Behavioral Anomaly Detection for Cybersecurity

An intelligent cybersecurity system that detects behavioral anomalies from access logs using machine learning, behavioral profiling, and explainable AI. The system identifies suspicious user/device activities, assigns risk scores, classifies attack types, and provides human-readable explanations through an interactive SOC dashboard.

---

## 📌 Overview

Traditional rule-based security systems struggle to detect sophisticated insider threats and previously unseen attacks.

This project builds an **AI-powered anomaly detection pipeline** that learns normal user behavior and flags deviations using **unsupervised learning**, while also classifying known attack patterns using supervised learning.

The project includes:

- Synthetic enterprise log generation
- Behavioral feature engineering
- Baseline profile creation
- Isolation Forest anomaly detection
- Random Forest attack classification
- Explainability module
- Streamlit SOC Dashboard

---

## 🚀 Features

- ✅ Synthetic enterprise access log generation
- ✅ User behavioral profiling
- ✅ Risk scoring using Isolation Forest
- ✅ Attack type prediction
- ✅ Explainable AI (human-readable alerts)
- ✅ Cold-start detection
- ✅ Concept drift handling
- ✅ Interactive SOC dashboard
- ✅ Ranked security alerts

---

# 🏗️ Project Architecture

```
Synthetic Log Generator
        │
        ▼
synthetic_access_logs.csv
        │
        ▼
Behavioral Feature Engineering
        │
        ▼
Baseline User Profiles
        │
        ▼
Isolation Forest
(Unsupervised Anomaly Detection)
        │
        ├────────► Risk Score
        │
        ▼
Random Forest
(Attack Classification)
        │
        ▼
Explainability Engine
        │
        ▼
SOC Dashboard (Streamlit)
```

---

# 📂 Project Structure

```
cyber-anomaly/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── synthetic_access_logs.csv
│   ├── features.csv
│   ├── scored_features.csv
│   └── explained_alerts.csv
│
├── data_gen/
│   └── generate_synthetic_logs.py
│
├── explainability/
│   └── explain.py
│
├── models/
│   ├── baseline_profile.py
│   ├── sequence_detector.py
│   ├── isolation_forest.joblib
│   ├── anomaly_classifier.joblib
│   └── scaler.joblib
│
├── report.md
└── README.md
```

---

# 🧠 Machine Learning Pipeline

## 1. Synthetic Data Generation

A realistic enterprise access log dataset is generated containing information such as:

- User ID
- Device ID
- Timestamp
- Geo location
- Resource accessed
- Session information
- Labels for evaluation

The generated dataset includes both normal activities and injected attack scenarios.

---

## 2. Behavioral Profiling

For every entity (user/device), the system builds a behavioral baseline including:

- Frequently accessed resources
- Typical login hours
- Common geographic locations
- Device usage patterns
- Historical activity profile

This baseline represents the user's normal behavior.

---

## 3. Feature Engineering

Behavioral features are extracted including:

- Resource novelty
- Geo deviation
- Time deviation
- Device deviation
- Session statistics
- Historical behavioral differences

These features become the input to the anomaly detection model.

---

## 4. Anomaly Detection

The primary anomaly detector is **Isolation Forest**.

### Why Isolation Forest?

- Works without labeled anomalies
- Excellent for highly imbalanced datasets
- Fast training
- Low computational cost
- Effective for unknown attack detection

The model assigns every session a **Risk Score (0–100)**.

Higher score → More suspicious behavior.

---

## 5. Attack Classification

Once a session is flagged as suspicious, a **Random Forest classifier** predicts the likely attack category.

Example attack types include:

- Credential misuse
- Insider threat
- Impossible travel
- Privilege abuse
- Low-and-slow exfiltration
- Unauthorized resource access

---

## 6. Explainable AI

Instead of showing only a risk score, the system generates analyst-friendly explanations.

Example:

> High risk detected due to impossible travel, previously unseen device, and abnormal resource access.

This helps SOC analysts understand why an alert was generated.

---

# 📊 Dashboard

The Streamlit dashboard provides:

### Alert Queue

- Ranked alerts
- Risk score
- Attack type
- Explanation
- Entity information

### Risk Distribution

Visualizes

- Risk score histogram
- Normal vs anomalous sessions

### Entity History

Displays

- Historical activity
- Risk score trend
- Previous alerts
- Behavioral timeline

### Security Metrics

- Total sessions
- Flagged alerts
- Cold-start sessions
- Ground truth anomalies

---

# 📈 Model Performance

Performance on the generated dataset:

| Alert Budget | Precision | Recall | ROC-AUC |
|--------------|----------:|-------:|--------:|
| 0.5% | 97.8% | 5.6% | 0.945 |
| 1.0% | 92.9% | 10.7% | 0.945 |
| 2.0% | 86.9% | 20.0% | 0.945 |
| 5.0% | 74.0% | 42.7% | 0.945 |

Attack classification achieves approximately **99% accuracy** on labeled anomaly classes.

---

# 🛠️ Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-learn
- Isolation Forest
- Random Forest

### Data Processing

- Pandas
- NumPy

### Visualization

- Streamlit
- Plotly

### Model Persistence

- Joblib

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/cyber-anomaly.git

cd cyber-anomaly
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1 — Generate Dataset

```bash
python data_gen/generate_synthetic_logs.py
```

---

## Step 2 — Create Behavioral Profiles

```bash
python models/baseline_profile.py
```

---

## Step 3 — Train Models & Score Events

```bash
python models/sequence_detector.py
```

---

## Step 4 — Generate Explanations

```bash
python explainability/explain.py
```

---

## Step 5 — Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 📌 Key Highlights

- Behavioral anomaly detection instead of static rule matching
- Learns normal user behavior automatically
- Detects previously unseen attacks
- Handles highly imbalanced cybersecurity datasets
- Human-readable alert explanations
- Interactive SOC analyst dashboard
- Modular ML pipeline suitable for production extension

---

# 🔮 Future Improvements

- Deep learning sequence models (LSTM/Transformer)
- Online model retraining for concept drift
- SHAP explanations for anomaly scoring
- Graph Neural Networks for user-resource relationships
- SIEM integration (Splunk, ELK, Microsoft Sentinel)
- Real-time Kafka event streaming
- Docker and Kubernetes deployment
- REST API for inference
- Continuous monitoring and alerting

---

# 👨‍💻 Author: Namrata

Developed as an AI-powered cybersecurity behavioral anomaly detection system demonstrating the application of machine learning, behavioral analytics, explainable AI, and interactive security visualization for modern Security Operations Centers (SOC).

---
