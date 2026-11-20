"""
=============================================================================
TELECOM NETWORK ANOMALY DETECTION SYSTEM
Module: generate_data.py
-----------------------------------------------------------------------------
DISCLAIMER:
This dataset is SYNTHETICALLY SIMULATED using NumPy and Pandas for academic,
research, and evaluation purposes.
IT DOES NOT REPRESENT REAL C-DOT NETWORK INFRASTRUCTURE OR CONFIDENTIAL TELECOM DATA.

Features:
1. latency_ms: End-to-end network latency in milliseconds.
2. packet_loss_pct: Percentage of dropped packets (0 - 100%).
3. packets_per_sec: Packet arrival rate per second (pps).
4. data_rate_mbps: Bandwidth throughput in Megabits per second (Mbps).
5. active_connections: Number of concurrent active TCP/UDP connections.
6. retransmission_rate: Packet retransmission rate percentage (0 - 100%).
7. avg_packet_size: Average Ethernet/IP frame payload size in bytes.
8. cpu_utilization_pct: Telecom node/router CPU utilization percentage (0 - 100%).
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd

# Define standard feature column names
FEATURE_COLUMNS = [
    "latency_ms",
    "packet_loss_pct",
    "packets_per_sec",
    "data_rate_mbps",
    "active_connections",
    "retransmission_rate",
    "avg_packet_size",
    "cpu_utilization_pct"
]

def generate_normal_traffic(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic normal operational traffic for a high-capacity telecom router/node.
    
    Statistical characteristics reflect healthy network operations:
    - Low to moderate latency (15ms - 45ms)
    - Negligible packet loss (< 0.5%)
    - Steady packets per second (1500 - 4500 pps)
    - Balanced data rate (25 - 85 Mbps)
    - Moderate active connection pool (250 - 750)
    - Minimal TCP retransmissions (< 1.5%)
    - Standard Internet MTU distribution (avg ~ 600 - 1050 bytes)
    - Stable CPU utilization (25% - 55%)
    """
    rng = np.random.RandomState(seed)

    # 1. Latency (ms): Gamma/Gaussian distribution around ~28ms with low jitter
    latency = rng.normal(loc=28.0, scale=4.5, size=n_samples)
    latency = np.clip(latency, 12.0, 50.0)

    # 2. Packet Loss (%): Exponential distribution skewed heavily towards 0.0%
    packet_loss = rng.exponential(scale=0.08, size=n_samples)
    packet_loss = np.clip(packet_loss, 0.0, 0.6)

    # 3. Packets per second (pps): Normal telecom flow
    pps = rng.normal(loc=2800.0, scale=450.0, size=n_samples)
    pps = np.clip(pps, 1400.0, 4800.0)

    # 4. Data Transfer Rate (Mbps): Throughput proportional to pps
    data_rate = (pps * rng.normal(loc=820, scale=50, size=n_samples) * 8) / 1e6
    data_rate = np.clip(data_rate, 15.0, 95.0)

    # 5. Active Connections: Healthy concurrent user sessions
    connections = rng.normal(loc=480.0, scale=80.0, size=n_samples)
    connections = np.clip(connections, 200, 850).astype(int)

    # 6. Retransmission Rate (%): Standard TCP retransmissions under clean line conditions
    retransmission = rng.exponential(scale=0.25, size=n_samples)
    retransmission = np.clip(retransmission, 0.01, 1.4)

    # 7. Average Packet Size (bytes): Standard web/VoIP/video mixed distribution
    packet_size = rng.normal(loc=780.0, scale=110.0, size=n_samples)
    packet_size = np.clip(packet_size, 450.0, 1250.0)

    # 8. CPU Utilization (%): Balanced processing load on router line-cards
    cpu = rng.normal(loc=38.0, scale=6.5, size=n_samples)
    cpu = np.clip(cpu, 18.0, 58.0)

    df = pd.DataFrame({
        "latency_ms": np.round(latency, 2),
        "packet_loss_pct": np.round(packet_loss, 3),
        "packets_per_sec": np.round(pps, 1),
        "data_rate_mbps": np.round(data_rate, 2),
        "active_connections": connections,
        "retransmission_rate": np.round(retransmission, 3),
        "avg_packet_size": np.round(packet_size, 1),
        "cpu_utilization_pct": np.round(cpu, 2),
    })

    return df

def generate_anomalous_traffic(n_samples: int, seed: int = 101) -> pd.DataFrame:
    """
    Generates synthetic anomalies mimicking 5 realistic telecom network failure scenarios:
    1. High Latency Congestion (Bufferbloat / Routing degradation)
    2. Sudden Packet Loss (Fiber attenuation / Physical link degradation)
    3. Traffic Spikes / DDoS (Volumetric flood)
    4. Abnormal Connection Surge (SYN Flood / Port scanning storm)
    5. High Retransmission Rate (TCP congestion collapse / faulty network interface card)
    """
    rng = np.random.RandomState(seed)
    
    # Divide the anomalous samples roughly equally across the 5 scenarios
    scenario_counts = [n_samples // 5] * 5
    # Allocate remainder to the last scenario
    scenario_counts[-1] += (n_samples - sum(scenario_counts))

    anomaly_dfs = []

    # Scenario 1: High Latency Congestion
    n1 = scenario_counts[0]
    df1 = generate_normal_traffic(n1, seed=seed + 1)
    df1["latency_ms"] = rng.uniform(220.0, 850.0, size=n1).round(2)
    df1["packet_loss_pct"] = rng.uniform(3.0, 12.0, size=n1).round(3)
    df1["retransmission_rate"] = rng.uniform(4.0, 15.0, size=n1).round(3)
    df1["cpu_utilization_pct"] = rng.uniform(65.0, 88.0, size=n1).round(2)
    df1["anomaly_type"] = "High Latency Congestion"
    anomaly_dfs.append(df1)

    # Scenario 2: Sudden Packet Loss / Link Degradation
    n2 = scenario_counts[1]
    df2 = generate_normal_traffic(n2, seed=seed + 2)
    df2["packet_loss_pct"] = rng.uniform(15.0, 48.0, size=n2).round(3)
    df2["retransmission_rate"] = rng.uniform(12.0, 32.0, size=n2).round(3)
    # Throughput throttles due to extreme packet drops
    df2["data_rate_mbps"] = rng.uniform(1.5, 12.0, size=n2).round(2)
    df2["packets_per_sec"] = rng.uniform(500.0, 1500.0, size=n2).round(1)
    df2["anomaly_type"] = "Sudden Packet Loss"
    anomaly_dfs.append(df2)

    # Scenario 3: Volumetric Traffic Spike / DDoS Flood
    n3 = scenario_counts[2]
    df3 = generate_normal_traffic(n3, seed=seed + 3)
    df3["packets_per_sec"] = rng.uniform(18000.0, 52000.0, size=n3).round(1)
    df3["data_rate_mbps"] = rng.uniform(250.0, 750.0, size=n3).round(2)
    df3["cpu_utilization_pct"] = rng.uniform(88.0, 99.8, size=n3).round(2)
    df3["latency_ms"] = rng.uniform(80.0, 240.0, size=n3).round(2)
    df3["anomaly_type"] = "Traffic Spike / DDoS"
    anomaly_dfs.append(df3)

    # Scenario 4: Abnormal Connection Surge / SYN Flood
    n4 = scenario_counts[3]
    df4 = generate_normal_traffic(n4, seed=seed + 4)
    df4["active_connections"] = rng.randint(4500, 22000, size=n4)
    df4["avg_packet_size"] = rng.uniform(64.0, 128.0, size=n4).round(1)  # small SYN packets
    df4["cpu_utilization_pct"] = rng.uniform(82.0, 97.0, size=n4).round(2)
    df4["packets_per_sec"] = rng.uniform(12000.0, 30000.0, size=n4).round(1)
    df4["anomaly_type"] = "Abnormal Connection Surge"
    anomaly_dfs.append(df4)

    # Scenario 5: High Retransmission Storm
    n5 = scenario_counts[4]
    df5 = generate_normal_traffic(n5, seed=seed + 5)
    df5["retransmission_rate"] = rng.uniform(22.0, 58.0, size=n5).round(3)
    df5["latency_ms"] = rng.uniform(90.0, 310.0, size=n5).round(2)
    df5["packet_loss_pct"] = rng.uniform(6.0, 20.0, size=n5).round(3)
    df5["anomaly_type"] = "High Retransmission Storm"
    anomaly_dfs.append(df5)

    df_anomalies = pd.concat(anomaly_dfs, ignore_index=True)
    df_anomalies["is_anomaly"] = 1
    return df_anomalies

def generate_and_save_datasets(
    output_dir: str = "data",
    n_train_normal: int = 8000,
    n_test_normal: int = 2000,
    n_test_anomalies: int = 300,
    seed: int = 42
):
    """
    Main orchestration function to create and save training and testing datasets.
    
    - train_normal.csv: Contains ONLY normal operational traffic (uncontaminated).
    - test_mixed.csv: Contains normal traffic + simulated anomalies with ground truth labels.
    - baseline_stats.json: Pre-computed statistical benchmarks of normal traffic.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING SYNTHETIC TELECOM NETWORK DATASETS")
    print("Disclaimer: Simulated data for research/academic evaluation only.")
    print("=" * 70)

    # 1. Generate Training Set: Strictly Normal Traffic (Unsupervised Training)
    print(f"[*] Generating {n_train_normal} normal training observations...")
    df_train_normal = generate_normal_traffic(n_train_normal, seed=seed)
    # Note: training set does not have anomaly labels because it's unsupervised
    train_path = os.path.join(output_dir, "train_normal.csv")
    df_train_normal.to_csv(train_path, index=False)
    print(f"[+] Saved training dataset: {train_path} ({len(df_train_normal)} rows)")

    # 2. Compute Normal Baseline Statistics (Used for diagnostics & explainability)
    baseline_stats = {}
    for col in FEATURE_COLUMNS:
        baseline_stats[col] = {
            "mean": float(df_train_normal[col].mean()),
            "std": float(df_train_normal[col].std()),
            "min": float(df_train_normal[col].min()),
            "max": float(df_train_normal[col].max()),
            "p5": float(df_train_normal[col].quantile(0.05)),
            "p25": float(df_train_normal[col].quantile(0.25)),
            "p50": float(df_train_normal[col].quantile(0.50)),
            "p75": float(df_train_normal[col].quantile(0.75)),
            "p95": float(df_train_normal[col].quantile(0.95)),
            "p99": float(df_train_normal[col].quantile(0.99))
        }
    stats_path = os.path.join(output_dir, "baseline_stats.json")
    with open(stats_path, "w") as f:
        json.dump(baseline_stats, f, indent=4)
    print(f"[+] Saved baseline normal statistics: {stats_path}")

    # 3. Generate Test Set: Normal Traffic + Known Simulated Anomalies
    print(f"[*] Generating {n_test_normal} normal test observations...")
    df_test_normal = generate_normal_traffic(n_test_normal, seed=seed + 999)
    df_test_normal["is_anomaly"] = 0
    df_test_normal["anomaly_type"] = "Normal"

    print(f"[*] Generating {n_test_anomalies} simulated anomalies across 5 fault categories...")
    df_test_anomalies = generate_anomalous_traffic(n_test_anomalies, seed=seed + 777)

    # Combine and shuffle test set
    df_test_mixed = pd.concat([df_test_normal, df_test_anomalies], ignore_index=True)
    df_test_mixed = df_test_mixed.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    test_path = os.path.join(output_dir, "test_mixed.csv")
    df_test_mixed.to_csv(test_path, index=False)
    print(f"[+] Saved mixed test dataset: {test_path} ({len(df_test_mixed)} rows)")
    
    print("\nDataset Summary:")
    print(f"- Training set (Normal only): {len(df_train_normal)} samples")
    print(f"- Test set (Mixed): {len(df_test_mixed)} samples")
    print(f"  * Normal samples: {(df_test_mixed['is_anomaly'] == 0).sum()} ({(df_test_mixed['is_anomaly'] == 0).mean() * 100:.1f}%)")
    print(f"  * Anomalous samples: {(df_test_mixed['is_anomaly'] == 1).sum()} ({(df_test_mixed['is_anomaly'] == 1).mean() * 100:.1f}%)")
    print("  * Anomalies breakdown:")
    for atype, cnt in df_test_mixed[df_test_mixed['is_anomaly'] == 1]['anomaly_type'].value_counts().items():
        print(f"    - {atype}: {cnt}")
    print("=" * 70)

if __name__ == "__main__":
    generate_and_save_datasets()
