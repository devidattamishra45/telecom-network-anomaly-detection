# 📡 Telecom Network Anomaly Detection System

An end-to-end, unsupervised Machine Learning system engineered to detect anomalous operational behaviors in high-throughput telecommunication networks using **Isolation Forest** (Scikit-Learn).

> **Academic & Interview Demonstration Note:**  
> This project is designed specifically for demonstration in a **Centre for Development of Telematics (C-DOT)** internship interview. All telemetry data is **synthetically simulated** using NumPy and Pandas to realistically model telecom phenomena (bufferbloat, link drops, DDoS, SYN floods, retransmission storms). **It does not use proprietary or confidential C-DOT network data.**

---

## 📑 Table of Contents
1. [Problem Statement & Telecom Context](#1-problem-statement--telecom-context)
2. [Machine Learning Methodology](#2-machine-learning-methodology)
3. [Input Features (8 Telecom Telemetry Metrics)](#3-input-features-8-telecom-telemetry-metrics)
4. [Dataset Architecture & Simulated Failure Modes](#4-dataset-architecture--simulated-failure-modes)
5. [Model Evaluation & Authentic Performance](#5-model-evaluation--authentic-performance)
6. [Rule-Based Diagnostic Indicators vs. Model Decision](#6-rule-based-diagnostic-indicators-vs-model-decision)
7. [System Architecture & Project Structure](#7-system-architecture--project-structure)
8. [Quick Start & Local Execution](#8-quick-start--local-execution)
9. [C-DOT Interview Guide: How to Present This Project](#9-c-dot-interview-guide-how-to-present-this-project)

---

## 1. Problem Statement & Telecom Context

Modern telecommunication infrastructures (such as those developed by **C-DOT**, including 4G/5G core networks, GPON optical line terminals, and broadband edge routers) transport gigabits of traffic every second. 

In such networks:
- **Normal traffic comprises >99.9% of flows.**
- **Faults and attacks are rare, heterogeneous, and unlabelled in real time.**
- Manual rule sets and static thresholds fail when traffic patterns change dynamically (e.g., flash crowds, new streaming protocols).
- Traditional supervised ML fails because labelled anomaly datasets are scarce, biased, or unavailable for zero-day faults.

### The Solution:
We formulate anomaly detection as an **unsupervised semi-supervised learning task**:
1. We train an **Isolation Forest** model exclusively on **uncontaminated normal operational traffic**.
2. The model learns the multi-dimensional manifold of healthy network behavior.
3. In production, any packet flow that departs from this manifold is flagged with an anomaly score and prioritized for NOC engineering triage.

---

## 2. Machine Learning Methodology

### Why Isolation Forest?
Isolation Forest (Liu et al.) is uniquely suited for telecom telemetry because:
1. **Few and Different:** Anomalies are numerically distant from normal clusters and sparse in feature space.
2. **Recursive Random Partitioning:** An ensemble of isolation trees (`iTrees`) splits features randomly. Outliers are isolated near the root of trees (short path length $h(x)$), while normal instances require deep recursive partitions.
3. **Linear Time Complexity $O(n \log n)$:** Extremely lightweight and fast enough to be deployed near line-rate at edge routers or stream-processing engines.
4. **No Distance Metrics:** Unlike $k$-means, KNN, or DBSCAN, Isolation Forest does not calculate pairwise Euclidean distances, avoiding the curse of dimensionality and expensive matrix operations.

```
                  Isolation Tree (iTree) Concept:
                  
               [Root: Latency Split = 120ms]
                     /                 \
            [Latency > 120ms]      [Latency <= 120ms]
                   /                       \
        *ANOMALY ISOLATED*         [Packet Loss Split = 2%]
          (Path Length = 1)               /          \
                                  [Loss > 2%]     [Deep Splits...]
                                       |                 |
                                   *Outlier*        *NORMAL DATA*
                                                (Path Length = 14)
```

---

## 3. Input Features (8 Telecom Telemetry Metrics)

Our system monitors 8 standard telecom QoS and device health metrics:

| # | Feature Name | Description | Normal Operating Range | Unit |
|---|---|---|---|---|
| 1 | `latency_ms` | End-to-end packet transmission delay | $15 - 45$ | ms |
| 2 | `packet_loss_pct` | Dropped packet percentage | $< 0.5\%$ | % |
| 3 | `packets_per_sec` | Throughput in packet arrival rate | $1,500 - 4,500$ | pps |
| 4 | `data_rate_mbps` | Bandwidth throughput rate | $25 - 85$ | Mbps |
| 5 | `active_connections` | Concurrent TCP/UDP sessions | $250 - 750$ | flows |
| 6 | `retransmission_rate` | TCP segments resent due to drop/timeout | $< 1.5\%$ | % |
| 7 | `avg_packet_size` | Average Ethernet/IP frame payload size | $600 - 1,050$ | bytes |
| 8 | `cpu_utilization_pct` | Router line-card CPU load | $25 - 55\%$ | % |

---

## 4. Dataset Architecture & Simulated Failure Modes

The dataset is generated in `generate_data.py` using calibrated statistical distributions (Gaussian, Poisson, Exponential) seeded for 100% reproducibility (`random_state=42`).

### 1. Training Set (`data/train_normal.csv`)
- **8,000 observations** of pure normal network traffic.
- **Strictly uncontaminated:** No anomalies and zero anomaly labels are provided during training.

### 2. Held-Out Evaluation Set (`data/test_mixed.csv`)
- **2,300 observations**: 2,000 normal flows (87.0%) + 300 simulated anomalies (13.0%).
- Used solely for objective testing and validation.

### 5 Realistic Simulated Telecom Anomaly Scenarios:
1. **High Latency Congestion (Bufferbloat):** Latency spikes to $200 - 850\text{ ms}$, packet loss rises to $3 - 12\%$, CPU load elevated.
2. **Sudden Packet Loss (Link Degradation):** Fiber optic bend / line degradation causing $15 - 48\%$ packet loss, retransmissions surge to $12 - 32\%$, throughput drops to $< 12\text{ Mbps}$.
3. **Volumetric Traffic Spike (DDoS):** Packets per second surges to $18,000 - 52,000\text{ pps}$, bandwidth spikes to $250 - 750\text{ Mbps}$, CPU pegged at $88 - 99.8\%$.
4. **Abnormal Connection Surge (SYN Flood / Scan):** Active connection table floods to $4,500 - 22,000$ concurrent sessions with tiny packet sizes ($64 - 128\text{ bytes}$) and CPU $> 85\%$.
5. **High Retransmission Storm (NIC / Interface Fault):** TCP retransmission jumps to $22 - 58\%$, latency climbs, severe channel degradation.

---

## 5. Model Evaluation & Authentic Performance

> [!IMPORTANT]
> **Zero Fabrication Policy:** All figures below are dynamically computed on the held-out 2,300-sample test set (`models/evaluation_metrics.json`).

| Metric | Authentic Result | Telecom Interpretation |
|---|---|---|
| **Precision** | **85.96%** | When the system sounds an alarm, ~86% are genuine critical anomalies. |
| **Recall / Detection Rate** | **100.00%** | Caught **300 out of 300** simulated anomalous telecom events. |
| **F1-Score** | **92.45%** | Harmonic balance between alarm reliability and fault coverage. |
| **False Positive Rate (FPR)** | **2.45%** | Only 49 false alarms out of 2,000 normal flows (critical for avoiding NOC alarm fatigue). |
| **Specificity** | **97.55%** | Correctly identifies nominal traffic without interruption. |

### Confusion Matrix
```
                         Predicted Normal (0)    Predicted Anomaly (1)
Actual Normal (0)                1,951                    49   [False Positives / 2.45%]
Actual Anomaly (1)                   0                   300   [True Positives / 100%]
```

### Detection Rate by Scenario
- **Abnormal Connection Surge:** 60/60 caught (100.0%)
- **High Latency Congestion:** 60/60 caught (100.0%)
- **High Retransmission Storm:** 60/60 caught (100.0%)
- **Sudden Packet Loss:** 60/60 caught (100.0%)
- **Traffic Spike / DDoS:** 60/60 caught (100.0%)

---

## 6. Rule-Based Diagnostic Indicators vs. Model Decision

A common problem in enterprise AI is "black-box" decision making: a model flags a packet as an anomaly, but the network engineer doesn't know *why*.

Our system cleanly decouples:
1. **Model Prediction (Isolation Forest):** Non-linear multi-metric isolation decision. Flags that the point as a whole is structurally abnormal.
2. **Diagnostic Indicators (Baseline Attribution):** Compares each input metric against precomputed normal baseline distributions (Mean, Std, 5th, 95th, 99th percentiles).
   - Computes metric Z-score: $Z = \frac{x - \mu_{\text{normal}}}{\sigma_{\text{normal}}}$.
   - Flags deviations:
     - **Nominal ($|Z| < 2.5\sigma$):** Within normal operational bounds.
     - **Moderate Deviation ($2.5\sigma \le |Z| < 4\sigma$):** Alert worthy.
     - **Critical Outlier ($|Z| \ge 4\sigma$ or $> P_{99} \times 1.5$):** Severe deviation.
   - Provides plain-English engineering triage findings (e.g., *"Elevated by +8.2σ above baseline. Possible optical link degradation"*).

---

## 7. System Architecture & Project Structure

```
telecom-network-anomaly-detection/
├── app.py                      # Professional Streamlit NOC dashboard
├── model.py                    # TelecomAnomalyDetector class & diagnostic engine
├── train_model.py              # Unsupervised training & authentic test evaluation
├── generate_data.py            # Reproducible synthetic dataset generator
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation & interview guide
├── data/
│   ├── train_normal.csv        # 8,000 normal baseline flows (uncontaminated)
│   ├── test_mixed.csv          # 2,300 test flows (normal + 5 anomaly scenarios)
│   └── baseline_stats.json     # Precomputed normal distribution benchmarks
├── models/
│   ├── isolation_forest_model.joblib # Serialized model pipeline (Scaler + Model + Metadata)
│   └── evaluation_metrics.json       # Authentic evaluation metrics
└── tests/
    ├── __init__.py
    └── test_pipeline.py        # 6 comprehensive unit tests (pytest)
```

---

## 8. Quick Start & Local Execution

### 1. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Datasets
```bash
python generate_data.py
```

### 3. Train the Isolation Forest Model & Evaluate
```bash
python train_model.py
```

### 4. Run the Unit Test Suite
```bash
pytest tests/ -v
```

### 5. Launch the Streamlit NOC Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 9. C-DOT Interview Guide: How to Present This Project

### 30-Second Elevator Pitch
> *"I built an unsupervised Telecom Network Anomaly Detection System using Isolation Forest. In telecom environments like C-DOT's routers and 4G/5G core, traffic is 99.9% normal, and collecting labeled datasets for real-time faults is impractical. My system trains exclusively on normal baseline traffic to learn healthy behavior across 8 telemetry features—including latency, packet loss, retransmission rates, and CPU utilization. On a held-out test set covering 5 realistic telecom failure modes (such as DDoS spikes, fiber link degradation, and SYN floods), the system achieved a 100% anomaly detection rate with an authentic false alarm rate of just 2.45%. I also developed a Streamlit NOC dashboard that clearly separates the ML prediction from rule-based diagnostic indicators to help network engineers diagnose root causes immediately."*

### Top 5 Interview Questions & Confident Answers

#### Q1: Why did you choose Isolation Forest instead of DBSCAN or Autoencoders?
- **Answer:** *"DBSCAN relies on pairwise Euclidean distance calculations, which scale as $O(n^2)$ and struggle in higher dimensions due to distance concentration. Autoencoders are powerful but computationally heavy, require GPU acceleration for real-time inference, and are sensitive to hyperparameter tuning. Isolation Forest has an $O(n \log n)$ time complexity, low memory overhead, and isolates outliers near the root of shallow trees. In telecom edge routers, low CPU overhead and sub-millisecond inference are essential."*

#### Q2: Why train only on normal data? Isn't that semi-supervised?
- **Answer:** *"Yes, this is often referred to as semi-supervised or one-class anomaly detection. In telecom operations, clean normal data is abundant during steady-state operations, whereas anomalies are rare, diverse, and unpredictable. If we trained a supervised classifier on known faults, it would fail to detect novel zero-day attacks or unforeseen physical fiber faults. Training on normal data ensures the model learns the envelope of healthy network behavior."*

#### Q3: How do you handle false positives in a telecom NOC?
- **Answer:** *"In telecom NOCs, alert fatigue is a major operational risk. If a system triggers 20% false alarms, operators turn off notifications. We control this via the `contamination` parameter and decision offset. By calibrating the decision threshold on normal traffic, we achieved a False Positive Rate of only 2.45% (97.55% specificity), meaning normal traffic passes through with minimal false alarms."*

#### Q4: What are the 'Contributing Diagnostic Indicators' and why are they separated from the model?
- **Answer:** *"Isolation Forest makes a holistic, multi-dimensional decision—it detects complex joint interactions (e.g., normal throughput combined with abnormal connection counts and small packet sizes). However, operators need actionable root cause hints. Our diagnostic layer computes Z-score standard deviations and percentile deviations for each metric against the normal baseline. We explicitly distinguish this rule-based triage layer from the ML model decision so operators know that the ML model made the anomaly classification, while the diagnostic indicators provide clinical engineering context for triage."*

#### Q5: How would this scale to a real C-DOT 100 Gbps core router?
- **Answer:** *"At 100 Gbps, inspecting every single packet payload at line rate with software ML is impossible. In production C-DOT architecture, this system would be fed by IPFIX, NetFlow, or sFlow sampled telemetry aggregated in 1-second or 10-second flow windows, or implemented on P4 programmable data planes (FPGA/ASIC) where shallow decision tree logic can be evaluated at hardware line rate."*
