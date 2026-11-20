"""
=============================================================================
TELECOM NETWORK ANOMALY DETECTION SYSTEM
Module: train_model.py
-----------------------------------------------------------------------------
Trains the Isolation Forest model strictly using normal training observations,
evaluates on a held-out mixed test set containing known simulated anomalies,
and serializes both the model artifact and authentic performance metrics.

NEVER FABRICATES PERFORMANCE NUMBERS:
All precision, recall, F1, confusion matrix, and detection rate metrics
are computed dynamically on the held-out test dataset.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from generate_data import generate_and_save_datasets, FEATURE_COLUMNS
from model import TelecomAnomalyDetector

def train_and_evaluate(
    data_dir: str = "data",
    model_dir: str = "models",
    contamination: float = 0.02,
    n_estimators: int = 150,
    random_state: int = 42
):
    """
    Orchestrates end-to-end model training, authentic evaluation, and artifact saving.
    """
    print("=" * 75)
    print("TELECOM ANOMALY DETECTION: MODEL TRAINING & RIGOROUS EVALUATION")
    print("=" * 75)

    train_path = os.path.join(data_dir, "train_normal.csv")
    test_path = os.path.join(data_dir, "test_mixed.csv")

    # If datasets don't exist, generate them automatically
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[!] Dataset files not found. Generating fresh synthetic datasets...")
        generate_and_save_datasets(output_dir=data_dir)

    print(f"[*] Loading training data from: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"[*] Loading testing data from:  {test_path}")
    df_test = pd.read_csv(test_path)

    print(f"    - Training observations (100% Normal): {len(df_train):,}")
    print(f"    - Testing observations (Normal + Anomalies): {len(df_test):,}")

    # =========================================================================
    # STEP 1: Unsupervised Training on Normal Traffic
    # =========================================================================
    print("\n" + "-" * 75)
    print("[1] Training Isolation Forest on Normal Telecom Baseline...")
    print(f"    - Hyperparameters: n_estimators={n_estimators}, contamination={contamination}, seed={random_state}")
    print("    - Training features:", ", ".join(FEATURE_COLUMNS))
    print("    - NOTE: Zero anomaly labels are provided during training.")
    print("-" * 75)

    detector = TelecomAnomalyDetector(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    detector.fit(df_train)
    print("[+] Model fitting complete!")

    # =========================================================================
    # STEP 2: Evaluation on Held-Out Test Set
    # =========================================================================
    print("\n" + "-" * 75)
    print("[2] Evaluating on Held-Out Test Set (Mixed Normal & Simulated Anomalies)...")
    print("-" * 75)

    y_true = df_test["is_anomaly"].values
    y_pred = detector.predict(df_test)
    y_scores = detector.score_samples(df_test)

    # Compute authentic metrics
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Domain-specific telecom metrics
    # Anomaly Detection Rate is synonymous with Recall on the anomaly class
    detection_rate = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    # False Positive Rate (FPR) = FP / (FP + TN)
    # Telecom NOC engineers require low FPR to avoid alarm fatigue
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    # Specificity = TN / (TN + FP)
    specificity = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0

    print("\n[+] AUTHENTIC MODEL PERFORMANCE METRICS:")
    print(f"    * Precision:              {precision * 100:.2f}% (Out of all alarms, {precision * 100:.2f}% were genuine anomalies)")
    print(f"    * Recall (Detection Rate): {recall * 100:.2f}% (Caught {tp} of {tp + fn} total anomalous events)")
    print(f"    * F1-Score:               {f1 * 100:.2f}%")
    print(f"    * False Positive Rate:    {fpr * 100:.2f}% (Normal traffic misclassified as anomaly)")
    print(f"    * Specificity:            {specificity * 100:.2f}% (Normal traffic correctly identified)")

    print("\n[+] CONFUSION MATRIX:")
    print(f"                     Predicted Normal (0)   Predicted Anomaly (1)")
    print(f"    Actual Normal (0)       {tn:<15}        {fp:<15} (False Positives)")
    print(f"    Actual Anomaly (1)      {fn:<15}        {tp:<15} (True Positives)")

    # Breakdown by specific anomaly scenario
    df_test["predicted_label"] = y_pred
    df_test["anomaly_score"] = y_scores

    scenario_metrics = {}
    print("\n[+] DETECTION RATE BREAKDOWN BY ANOMALY SCENARIO:")
    anomaly_groups = df_test[df_test["is_anomaly"] == 1].groupby("anomaly_type")
    for stype, group in anomaly_groups:
        detected = (group["predicted_label"] == 1).sum()
        total = len(group)
        rate = float(detected / total)
        scenario_metrics[stype] = {
            "total_samples": int(total),
            "detected_samples": int(detected),
            "detection_rate_pct": round(rate * 100, 2)
        }
        print(f"    - {stype:<30}: {detected}/{total} caught ({rate * 100:.1f}%)")

    # Classification report
    clr = classification_report(
        y_true,
        y_pred,
        target_names=["Normal Traffic", "Anomalous Traffic"],
        output_dict=True
    )

    # =========================================================================
    # STEP 3: Save Artifacts
    # =========================================================================
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "isolation_forest_model.joblib")
    detector.save(model_path)

    metrics_payload = {
        "dataset_summary": {
            "train_samples": int(len(df_train)),
            "test_samples": int(len(df_test)),
            "test_normal_count": int((y_true == 0).sum()),
            "test_anomaly_count": int((y_true == 1).sum())
        },
        "overall_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "detection_rate": round(detection_rate, 4),
            "false_positive_rate": round(fpr, 4),
            "specificity": round(specificity, 4)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "scenario_breakdown": scenario_metrics,
        "classification_report": clr,
        "features": FEATURE_COLUMNS
    }

    metrics_path = os.path.join(model_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"[+] Saved evaluation metrics to: {metrics_path}")
    print("=" * 75)
    print("MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 75)

    return detector, metrics_payload

if __name__ == "__main__":
    train_and_evaluate()
