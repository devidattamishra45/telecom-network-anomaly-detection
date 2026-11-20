"""
=============================================================================
TELECOM NETWORK ANOMALY DETECTION SYSTEM
Module: model.py
-----------------------------------------------------------------------------
This module defines the TelecomAnomalyDetector class wrapping Scikit-Learn's
Isolation Forest.

WHY ISOLATION FOREST?
--------------------
Isolation Forest is an unsupervised tree-based algorithm based on the concept
that anomalies are "few and different". It isolates anomalies by randomly
selecting a feature and then randomly selecting a split value between the
maximum and minimum values of the selected feature.
Because recursive partitioning can be represented by a tree structure,
the number of splittings required to isolate a sample is equivalent to the
path length from the root node to the terminating leaf node.
Anomalous instances have significantly shorter path lengths than normal instances!

In telecom networks, where:
- We have massive volumes of continuous operational telemetry
- Anomaly labels are scarce or non-existent in real-time
- Multiple interacting metrics (latency, loss, connection counts) shift simultaneously
Isolation Forest trained on normal traffic is the premier unsupervised ML choice.
=============================================================================
"""

import os
import json
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

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

FEATURE_UNITS = {
    "latency_ms": "ms",
    "packet_loss_pct": "%",
    "packets_per_sec": "pps",
    "data_rate_mbps": "Mbps",
    "active_connections": "conns",
    "retransmission_rate": "%",
    "avg_packet_size": "bytes",
    "cpu_utilization_pct": "%"
}

FEATURE_PRETTY_NAMES = {
    "latency_ms": "Network Latency",
    "packet_loss_pct": "Packet Loss",
    "packets_per_sec": "Packets Per Second",
    "data_rate_mbps": "Data Transfer Rate",
    "active_connections": "Active Connections",
    "retransmission_rate": "Retransmission Rate",
    "avg_packet_size": "Average Packet Size",
    "cpu_utilization_pct": "CPU Utilization"
}

class TelecomAnomalyDetector:
    """
    Production-ready wrapper around Scikit-Learn's Isolation Forest
    for Telecom Network Anomaly Detection.
    
    Trained strictly on clean normal network traffic to model the normal operational manifold.
    """
    def __init__(
        self,
        n_estimators: int = 150,
        max_samples: Any = "auto",
        contamination: float = 0.01,
        random_state: int = 42
    ):
        """
        Parameters:
        -----------
        n_estimators : int
            The number of base isolation trees in the ensemble (default: 150).
        max_samples : int or 'auto'
            The number of samples to draw to train each isolation tree.
        contamination : float or 'auto'
            Expected proportion of outliers. In our training setup on pure normal data,
            we set a tiny margin (e.g. 0.01 or 'auto') as decision boundary offset.
        random_state : int
            Seed for reproducible tree splits.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.is_fitted = False
        self.feature_names = FEATURE_COLUMNS
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame) -> "TelecomAnomalyDetector":
        """
        Fit the preprocessor and Isolation Forest model on normal training traffic.
        
        Important:
        - X must only contain features (no labels).
        - Computes baseline statistical profiles (mean, std, percentiles) for explainability.
        """
        # Validate columns
        for col in self.feature_names:
            if col not in X.columns:
                raise ValueError(f"Missing required feature column: {col}")

        X_subset = X[self.feature_names].copy()

        # 1. Store baseline distribution stats for diagnostics
        self.baseline_stats = {}
        for col in self.feature_names:
            values = X_subset[col].values
            self.baseline_stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)) if float(np.std(values)) > 1e-6 else 1.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p5": float(np.percentile(values, 5)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99))
            }

        # 2. Fit scaler and Isolation Forest
        X_scaled = self.scaler.fit_transform(X_subset)
        self.model.fit(X_scaled)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict whether each sample is normal or an anomaly.
        
        Returns:
        --------
        np.ndarray:
            0 for Normal Traffic
            1 for Anomaly Detected
        (Scikit-Learn IsolationForest returns +1 for inlier and -1 for outlier;
         we convert this to standard 0/1 binary labels).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() or load() first.")

        X_subset = X[self.feature_names].copy()
        X_scaled = self.scaler.transform(X_subset)
        
        # raw_preds: 1 for normal, -1 for anomaly
        raw_preds = self.model.predict(X_scaled)
        
        # Convert: 1 -> 0 (Normal), -1 -> 1 (Anomaly)
        binary_preds = np.where(raw_preds == -1, 1, 0)
        return binary_preds

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Direct access to raw Isolation Forest decision function.
        In scikit-learn, negative values indicate anomalies, positive indicate inliers.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        X_subset = X[self.feature_names].copy()
        X_scaled = self.scaler.transform(X_subset)
        return self.model.decision_function(X_scaled)

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns normalized Anomaly Scores between [0.0 and 1.0].
        
        0.0 = completely normal / deeply embedded in normal cluster
        1.0 = extreme anomaly / isolated early in trees
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")

        # decision_function gives signed distance to separating hyperplane:
        # positive = normal, negative = anomaly.
        raw_scores = self.decision_function(X)

        # We invert and map to a 0-1 probability-like scale using a calibrated logistic function:
        # Normal samples have positive raw_scores -> score < 0.5
        # Anomalies have negative raw_scores -> score > 0.5
        # Higher score = more anomalous.
        normalized_scores = 1.0 / (1.0 + np.exp(raw_scores * 10.0))
        return np.round(normalized_scores, 4)

    def diagnose_sample(self, sample_series_or_dict: Any) -> List[Dict[str, Any]]:
        """
        RULE-BASED DIAGNOSTIC INDICATOR ENGINE
        ---------------------------------------
        Provides metric-level attribution to explain WHY a network observation
        deviates from the normal telecom baseline.
        
        Note: This is explicitly a diagnostic triage layer that compares
        observed values against normal operational distributions (Z-scores & percentiles),
        which complements the non-linear multi-dimensional Isolation Forest decision.
        
        Returns a list of diagnostic records for each feature, sorted by deviation severity.
        """
        if not self.baseline_stats:
            raise RuntimeError("Baseline statistics not found. Ensure model is fitted or loaded.")

        diagnostics = []

        for feature in self.feature_names:
            val = float(sample_series_or_dict[feature])
            stats = self.baseline_stats[feature]
            mean = stats["mean"]
            std = stats["std"]
            p5 = stats["p5"]
            p95 = stats["p95"]
            p99 = stats["p99"]

            # Z-score: how many standard deviations away from normal mean
            z_score = (val - mean) / std
            abs_z = abs(z_score)

            # Determine indicator status and clinical description
            if abs_z >= 4.0 or val > p99 * 1.5:
                status = "Critical Outlier"
                badge_color = "red"
                is_contributor = True
            elif abs_z >= 2.5 or val > p95 or val < p5:
                status = "Moderate Deviation"
                badge_color = "orange"
                is_contributor = True
            else:
                status = "Nominal"
                badge_color = "green"
                is_contributor = False

            # Descriptive explanation
            unit = FEATURE_UNITS.get(feature, "")
            name = FEATURE_PRETTY_NAMES.get(feature, feature)
            if z_score > 2.0:
                detail = f"Elevated by +{abs_z:.1f}σ above normal baseline (Normal mean: {mean:.1f} {unit})"
            elif z_score < -2.0:
                detail = f"Depressed by -{abs_z:.1f}σ below normal baseline (Normal mean: {mean:.1f} {unit})"
            else:
                detail = f"Within expected baseline operating range ({p5:.1f} - {p95:.1f} {unit})"

            diagnostics.append({
                "feature": feature,
                "name": name,
                "observed_value": val,
                "unit": unit,
                "baseline_mean": mean,
                "baseline_std": std,
                "z_score": round(z_score, 2),
                "abs_z": round(abs_z, 2),
                "status": status,
                "badge_color": badge_color,
                "is_contributor": is_contributor,
                "detail": detail
            })

        # Sort with highest deviations first
        diagnostics.sort(key=lambda d: d["abs_z"], reverse=True)
        return diagnostics

    def save(self, filepath: str):
        """Serializes the complete detector pipeline and baseline statistics via joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "baseline_stats": self.baseline_stats,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "contamination": self.contamination,
                "random_state": self.random_state
            },
            "is_fitted": self.is_fitted
        }
        joblib.dump(payload, filepath)
        print(f"[+] Model pipeline successfully saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "TelecomAnomalyDetector":
        """Loads a saved detector pipeline from a joblib file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        payload = joblib.load(filepath)
        detector = cls(
            n_estimators=payload["hyperparameters"]["n_estimators"],
            max_samples=payload["hyperparameters"]["max_samples"],
            contamination=payload["hyperparameters"]["contamination"],
            random_state=payload["hyperparameters"]["random_state"]
        )
        detector.model = payload["model"]
        detector.scaler = payload["scaler"]
        detector.feature_names = payload["feature_names"]
        detector.baseline_stats = payload["baseline_stats"]
        detector.is_fitted = payload["is_fitted"]
        return detector
