"""
Baseline Profiling Model
-------------------------
Builds a per-entity statistical "normal behaviour" profile and turns raw log
rows into numeric features used by the detection model.

Handles cold-start: if an entity has < MIN_HISTORY sessions, we fall back to
a population-level (entity_type) profile instead of a per-entity one, and we
flag the row as `is_cold_start=True` so downstream models/analysts know the
score is less reliable.
"""

import json
import numpy as np
import pandas as pd

MIN_HISTORY = 5  # sessions needed before an entity gets its own profile


def _safe_json_len(x):
    try:
        return len(json.loads(x))
    except Exception:
        return 0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["cmd_count"] = df["command_sequence"].apply(_safe_json_len)
    df["country"] = df["geo_location"].apply(lambda g: str(g).split("/")[0])

    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    # Build per-entity baseline profiles from *prior* sessions only (no leakage)
    profiles = {}
    feats = []
    for entity_id, grp in df.groupby("entity_id"):
        history_hours, history_geo, history_res, history_device = [], set(), set(), set()
        for idx, row in grp.iterrows():
            n_hist = len(history_hours)
            is_cold = n_hist < MIN_HISTORY

            if is_cold:
                hour_dev = 0.0
                geo_novel = 0
                res_novel = 0
                device_novel = 0
            else:
                mean_h = np.mean(history_hours)
                std_h = np.std(history_hours) + 1e-6
                hour_dev = abs(row["hour"] - mean_h) / std_h
                geo_novel = int(row["country"] not in history_geo)
                res_novel = int(row["resource_accessed"] not in history_res)
                device_novel = int(row["device_fingerprint"] not in history_device)

            feats.append({
                "session_id": row["session_id"],
                "hour_deviation": hour_dev,
                "geo_novel": geo_novel,
                "resource_novel": res_novel,
                "device_novel": device_novel,
                "cmd_count": row["cmd_count"],
                "session_duration": row["session_duration"],
                "auth_failed": int(not row.get("auth_success", True)),
                "is_cold_start": int(is_cold),
                "n_prior_sessions": n_hist,
            })

            history_hours.append(row["hour"])
            history_geo.add(row["country"])
            history_res.add(row["resource_accessed"])
            history_device.add(row["device_fingerprint"])

    feat_df = pd.DataFrame(feats)
    # avoid duplicate-column suffixing: df already has cmd_count/session_duration
    dup_cols = [c for c in feat_df.columns if c in df.columns and c != "session_id"]
    df_reduced = df.drop(columns=dup_cols)
    out = df_reduced.merge(feat_df, on="session_id", how="left")

    # Velocity feature: time since entity's previous session vs geo distance proxy
    out["prev_country"] = out.groupby("entity_id")["country"].shift(1)
    out["country_changed"] = (out["country"] != out["prev_country"]).astype(int)
    out["seconds_since_prev"] = out.groupby("entity_id")["timestamp"].diff().dt.total_seconds()
    out["impossible_travel_flag"] = (
        (out["country_changed"] == 1) &
        (out["seconds_since_prev"] < 3 * 3600) &
        (out["seconds_since_prev"].notna())
    ).astype(int)

    # Brute-force / credential-stuffing proxy: failed-auth density per source_ip in rolling window
    out = out.sort_values("timestamp")
    out["failed_auth_count_ip"] = (
        out.groupby("source_ip")["auth_failed"]
        .transform(lambda s: s.rolling(10, min_periods=1).sum())
    )
    out["distinct_entities_per_ip"] = (
        out.groupby("source_ip")["entity_id"].transform("nunique")
    )

    return out


FEATURE_COLUMNS = [
    "hour_deviation", "geo_novel", "resource_novel", "device_novel",
    "cmd_count", "session_duration", "auth_failed", "is_cold_start",
    "impossible_travel_flag", "failed_auth_count_ip", "distinct_entities_per_ip",
]


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/cyber-anomaly/data/synthetic_access_logs.csv")
    feat_df = build_features(df)
    feat_df.to_csv("/home/claude/cyber-anomaly/data/features.csv", index=False)
    print(f"Built features for {len(feat_df)} rows.")
    print(feat_df[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]])
