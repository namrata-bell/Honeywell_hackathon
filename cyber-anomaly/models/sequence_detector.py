"""
Detection Model
----------------
Two-stage design chosen for reliability under a tight build window:

Stage A - Anomaly scoring (unsupervised, robust to extreme class imbalance):
  IsolationForest over the behavioural feature vector. Chosen over an
  LSTM/autoencoder as the primary model because it trains in seconds, needs no
  GPU, and is far less likely to overfit on ~18k rows with <2% positives -
  a safer choice than a deep sequence model under hackathon time constraints.
  (A GRU/LSTM sequence-encoder over `command_sequence` is documented as an
  optional stretch-goal upgrade path in report.md; it is not implemented in
  this file — flagged here so the gap is explicit rather than implied.)

Stage B - Anomaly-type classification (supervised, only run on flagged items):
  RandomForest trained on the labelled synthetic anomalies to say *which*
  attack category a flagged session resembles, satisfying deliverable #4
  ("not just anomalous, but which attack category it resembles").

Concept drift handling: profiles in baseline_profile.py are built from a
rolling history per entity (not a fixed training set), so "normal" naturally
shifts as an entity's real behaviour shifts. A periodic re-fit of the
IsolationForest (e.g. nightly) is recommended in production - documented in
report.md - to fully track drift.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import joblib

from baseline_profile import FEATURE_COLUMNS


def train_anomaly_scorer(feat_df: pd.DataFrame, contamination=0.02, random_state=42):
    X = feat_df[FEATURE_COLUMNS].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(Xs)

    # score_samples: higher = more normal. Convert to a 0-100 risk score, higher = riskier.
    raw_scores = iso.score_samples(Xs)
    risk_score = 100 * (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)

    return iso, scaler, risk_score


def train_anomaly_classifier(feat_df: pd.DataFrame, random_state=42):
    """Supervised classifier over labelled anomalies only, to say which attack
    category a flagged event resembles. Uses ground-truth labels for
    training/eval, never as an inference-time input to the scorer."""
    labelled = feat_df[feat_df["label"].isin(["anomaly", "edge_case"])].copy()
    if labelled["anomaly_type"].nunique() < 2:
        return None, None

    X = labelled[FEATURE_COLUMNS].fillna(0).values
    y = labelled["anomaly_type"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=random_state
    )
    clf.fit(X_train, y_train)
    report = classification_report(y_test, clf.predict(X_test), zero_division=0)
    return clf, report


def evaluate_scorer(feat_df: pd.DataFrame, risk_score, alert_budget_pct=1.0):
    """Evaluate detection quality at a realistic analyst alert budget
    (top X% of events by risk score), per the evaluation criteria."""
    y_true = (feat_df["label"] != "normal").astype(int).values
    n_alerts = max(1, int(len(risk_score) * alert_budget_pct / 100))
    top_idx = np.argsort(-risk_score)[:n_alerts]
    alerted = np.zeros(len(risk_score), dtype=int)
    alerted[top_idx] = 1

    tp = int(((alerted == 1) & (y_true == 1)).sum())
    fp = int(((alerted == 1) & (y_true == 0)).sum())
    fn = int(((alerted == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    try:
        auc = roc_auc_score(y_true, risk_score)
    except ValueError:
        auc = float("nan")

    return {
        "alert_budget_pct": alert_budget_pct,
        "n_alerts": n_alerts,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision_at_budget": round(precision, 4),
        "recall_at_budget": round(recall, 4),
        "roc_auc": round(float(auc), 4),
    }


if __name__ == "__main__":
    feat_df = pd.read_csv("/home/claude/cyber-anomaly/data/features.csv")

    iso, scaler, risk_score = train_anomaly_scorer(feat_df)
    feat_df["risk_score"] = risk_score

    clf, report = train_anomaly_classifier(feat_df)
    print("=== Anomaly-type classifier report (held-out labelled anomalies) ===")
    print(report)

    print("\n=== Detection performance at realistic alert budgets ===")
    for budget in [0.5, 1.0, 2.0, 5.0]:
        metrics = evaluate_scorer(feat_df, risk_score, alert_budget_pct=budget)
        print(metrics)

    # Persist artifacts for the dashboard
    joblib.dump(iso, "/home/claude/cyber-anomaly/models/isolation_forest.joblib")
    joblib.dump(scaler, "/home/claude/cyber-anomaly/models/scaler.joblib")
    joblib.dump(clf, "/home/claude/cyber-anomaly/models/anomaly_classifier.joblib")
    feat_df.to_csv("/home/claude/cyber-anomaly/data/scored_features.csv", index=False)
    print("\nSaved scored_features.csv and model artifacts.")
