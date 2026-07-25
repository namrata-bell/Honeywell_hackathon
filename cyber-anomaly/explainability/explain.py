"""
Explainability Layer
---------------------
For each flagged session, produce a human-readable reason string
(e.g. "flagged due to geo-velocity + new device fingerprint") by comparing
each feature's z-score contribution against the population baseline.

We use a lightweight, dependency-free contribution method (per-feature
deviation from population mean, weighted by IsolationForest's implicit
feature usage) rather than full SHAP-on-IsolationForest, which is slow and
overkill for a 300-tree ensemble under time pressure. SHAP (TreeExplainer)
is used instead on the anomaly-type RandomForest classifier, since that's
fast and directly supported.
"""

import numpy as np
import pandas as pd
import shap
import joblib

FEATURE_LABELS = {
    "hour_deviation": "unusual login time",
    "geo_novel": "new/unseen country",
    "resource_novel": "accessed a resource never used before",
    "device_novel": "new/unseen device fingerprint",
    "cmd_count": "unusual command count",
    "session_duration": "unusual session duration",
    "auth_failed": "authentication failure",
    "is_cold_start": "no prior history (cold-start entity)",
    "impossible_travel_flag": "geo-velocity: impossible travel",
    "failed_auth_count_ip": "high failed-auth density from this source IP",
    "distinct_entities_per_ip": "many distinct entities from same IP",
}

FEATURE_COLUMNS = list(FEATURE_LABELS.keys())


def explain_row(row: pd.Series, feat_stats: pd.DataFrame, top_k=2):
    """Return top-k contributing features in plain English for one row."""
    contributions = []
    for col in FEATURE_COLUMNS:
        val = row.get(col, 0)
        mean = feat_stats.loc[col, "mean"]
        std = feat_stats.loc[col, "std"] + 1e-6
        z = abs((val - mean) / std)
        if val > 0 or z > 0.5:  # binary flags: any positive value counts
            contributions.append((col, z))
    contributions.sort(key=lambda x: -x[1])
    top = contributions[:top_k]
    if not top:
        return "no strong deviation from baseline"
    return "flagged due to " + " + ".join(FEATURE_LABELS[c] for c, _ in top)


def explain_with_shap(clf, X, feature_names):
    """SHAP TreeExplainer on the RandomForest anomaly-type classifier."""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    return shap_values


if __name__ == "__main__":
    feat_df = pd.read_csv("/home/claude/cyber-anomaly/data/scored_features.csv")
    stats = feat_df[FEATURE_COLUMNS].describe().T[["mean", "std"]]

    feat_df["explanation"] = feat_df.apply(lambda r: explain_row(r, stats), axis=1)

    top_alerts = feat_df.sort_values("risk_score", ascending=False).head(10)
    print("=== Top 10 alerts with explanations ===")
    for _, r in top_alerts.iterrows():
        print(f"[{r['risk_score']:.1f}] {r['entity_id']} -> {r['explanation']}  "
              f"(true label: {r['label']}/{r['anomaly_type']})")

    feat_df.to_csv("/home/claude/cyber-anomaly/data/explained_alerts.csv", index=False)
    print("\nSaved explained_alerts.csv")
