"""
=============================================================================
TELECOM NETWORK ANOMALY DETECTION SYSTEM
Module: tests/test_pipeline.py
-----------------------------------------------------------------------------
Unit tests verifying data generation, model training, inference,
diagnostic indicators, and serialization integrity.
=============================================================================
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from generate_data import (
    generate_normal_traffic,
    generate_anomalous_traffic,
    FEATURE_COLUMNS
)
from model import TelecomAnomalyDetector

def test_generate_normal_traffic():
    """Verifies normal traffic generation shape, columns, and data validity."""
    n = 200
    df = generate_normal_traffic(n, seed=42)
    assert len(df) == n, "Dataframe row count must match requested n_samples"
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Feature column {col} must exist in generated data"
    assert df.isnull().sum().sum() == 0, "Generated data must not contain any NaN values"
    
    # Check reasonable operational boundaries for normal network traffic
    assert (df["latency_ms"] >= 0).all(), "Latency must be non-negative"
    assert (df["packet_loss_pct"] >= 0).all() and (df["packet_loss_pct"] <= 5).all(), \
        "Normal packet loss should be low"
    assert (df["cpu_utilization_pct"] >= 0).all() and (df["cpu_utilization_pct"] <= 100).all(), \
        "CPU utilization must be bounded [0, 100]"

def test_generate_anomalous_traffic():
    """Verifies anomalous traffic simulation covers all 5 fault scenarios."""
    n = 100
    df_anom = generate_anomalous_traffic(n, seed=99)
    assert len(df_anom) == n
    assert "is_anomaly" in df_anom.columns
    assert (df_anom["is_anomaly"] == 1).all(), "All simulated anomaly rows must have label 1"
    assert "anomaly_type" in df_anom.columns
    
    expected_scenarios = {
        "High Latency Congestion",
        "Sudden Packet Loss",
        "Traffic Spike / DDoS",
        "Abnormal Connection Surge",
        "High Retransmission Storm"
    }
    present_scenarios = set(df_anom["anomaly_type"].unique())
    assert expected_scenarios == present_scenarios, "All 5 realistic anomaly scenarios must be present"

def test_model_fit_predict_and_scores():
    """Verifies that the detector fits on normal data, predicts 0/1, and outputs valid scores."""
    df_train = generate_normal_traffic(500, seed=12)
    detector = TelecomAnomalyDetector(n_estimators=50, random_state=42)
    detector.fit(df_train)
    assert detector.is_fitted is True

    # Test predictions
    df_test_norm = generate_normal_traffic(50, seed=13)
    preds = detector.predict(df_test_norm)
    assert set(np.unique(preds)).issubset({0, 1}), "Predictions must only be 0 (Normal) or 1 (Anomaly)"

    # Test anomaly scores
    scores = detector.score_samples(df_test_norm)
    assert len(scores) == len(df_test_norm)
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), "Anomaly scores must be bounded in [0.0, 1.0]"

def test_model_serialization():
    """Verifies that model can be saved and loaded with identical predictions."""
    df_train = generate_normal_traffic(300, seed=24)
    detector = TelecomAnomalyDetector(n_estimators=30, random_state=42)
    detector.fit(df_train)

    df_test = generate_normal_traffic(20, seed=25)
    orig_preds = detector.predict(df_test)
    orig_scores = detector.score_samples(df_test)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model.joblib")
        detector.save(model_path)
        assert os.path.exists(model_path)

        loaded_detector = TelecomAnomalyDetector.load(model_path)
        assert loaded_detector.is_fitted is True
        loaded_preds = loaded_detector.predict(df_test)
        loaded_scores = loaded_detector.score_samples(df_test)

        np.testing.assert_array_equal(orig_preds, loaded_preds)
        np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)

def test_diagnostic_indicators():
    """Verifies that rule-based diagnostic indicators correctly identify deviating features."""
    df_train = generate_normal_traffic(400, seed=36)
    detector = TelecomAnomalyDetector(n_estimators=30, random_state=42)
    detector.fit(df_train)

    # Construct an extreme DDoS / High Latency sample
    severe_sample = {
        "latency_ms": 650.0,             # Normal is ~28ms -> Critical
        "packet_loss_pct": 0.05,          # Normal
        "packets_per_sec": 45000.0,       # Normal is ~2800 -> Critical
        "data_rate_mbps": 550.0,          # Normal is ~40 -> Critical
        "active_connections": 500,        # Normal
        "retransmission_rate": 0.2,       # Normal
        "avg_packet_size": 750.0,         # Normal
        "cpu_utilization_pct": 98.0       # Normal is ~38% -> Critical
    }

    diagnostics = detector.diagnose_sample(severe_sample)
    assert isinstance(diagnostics, list)
    assert len(diagnostics) == 8

    # The top deviations should be the critical injected features
    top_dev_features = [d["feature"] for d in diagnostics[:4]]
    for expected in ["latency_ms", "packets_per_sec", "data_rate_mbps", "cpu_utilization_pct"]:
        assert expected in top_dev_features

    # Verify attributes in each diagnostic record
    for record in diagnostics:
        assert "feature" in record
        assert "name" in record
        assert "z_score" in record
        assert "status" in record
        assert "detail" in record
        assert record["status"] in ["Critical Outlier", "Moderate Deviation", "Nominal"]

def test_single_sample_inference():
    """Verifies inference works correctly on a single-row DataFrame."""
    df_train = generate_normal_traffic(200, seed=48)
    detector = TelecomAnomalyDetector(n_estimators=20, random_state=42)
    detector.fit(df_train)

    single_df = generate_normal_traffic(1, seed=99)
    pred = detector.predict(single_df)
    score = detector.score_samples(single_df)
    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert 0.0 <= score[0] <= 1.0
