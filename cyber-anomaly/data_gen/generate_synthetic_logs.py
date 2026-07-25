"""
Synthetic Access-Log Generator for AI-Powered Behavioral Anomaly Detection

Generates per-entity behavioral logs matching the suggested schema:
entity_id, entity_type, timestamp, source_ip, geo_location, resource_accessed,
auth_method, session_duration, command_sequence, device_fingerprint, label, anomaly_type

Design assumptions (documented per deliverable #1):
- Each entity (user/service_account/edge_device) has a stable "home" behavioral
  profile: typical login-hour window, home geo/IP range, typical resource set,
  typical auth method, typical device fingerprint.
- Normal sessions are sampled from that profile with Gaussian/categorical noise.
- Anomalies are injected at a controlled rate (default 1.5% of sessions) by
  drawing from one of 8 documented attack/edge-case generators.
- Ground truth (`label`, `anomaly_type`) is retained in a separate column and
  must be dropped before feeding data to any model (only used for training
  labels / evaluation, never as a model input feature).
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid
import json

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

ENTITY_TYPES = ["user", "service_account", "edge_device"]
AUTH_METHODS = ["password", "token", "certificate", "biometric"]
RESOURCES = [
    "billing_db", "hr_portal", "file_server_1", "file_server_2", "crm_api",
    "payment_gateway", "customer_records", "network_switch_cfg", "plc_endpoint",
    "vpn_gateway", "email_server", "backup_service", "admin_console", "dns_service",
]
GEOS = [
    ("US", "New York"), ("US", "San Francisco"), ("US", "Chicago"),
    ("IN", "Mumbai"), ("IN", "Bengaluru"), ("GB", "London"),
    ("DE", "Berlin"), ("SG", "Singapore"), ("BR", "Sao Paulo"),
    ("RU", "Moscow"), ("CN", "Shanghai"), ("NG", "Lagos"),
]
COMMANDS_POOL = [
    "login", "read_file", "write_file", "list_dir", "download", "upload",
    "change_permission", "create_user", "delete_user", "escalate_privilege",
    "export_data", "query_db", "restart_service", "read_config", "write_config",
]


def _geo_ip(geo):
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


class Entity:
    """Represents one user/service_account/edge_device with a stable behavioral profile."""

    def __init__(self, entity_id, entity_type):
        self.entity_id = entity_id
        self.entity_type = entity_type
        # Typical login-hour window (e.g. 8am-7pm with some spread)
        self.home_hour_start = random.randint(6, 10)
        self.home_hour_end = self.home_hour_start + random.randint(6, 10)
        self.home_geo = random.choice(GEOS)
        self.home_auth = random.choice(AUTH_METHODS)
        self.home_resources = random.sample(RESOURCES, k=random.randint(2, 5))
        self.home_device = {
            "os": random.choice(["Windows11", "Ubuntu22.04", "macOS14", "IoT-Linux-5.10"]),
            "mac": fake.mac_address(),
        }
        self.typical_session_duration = random.uniform(120, 1800)  # seconds


def make_entities(n_users=120, n_service=25, n_devices=40):
    entities = []
    for i in range(n_users):
        entities.append(Entity(f"user_{i:04d}", "user"))
    for i in range(n_service):
        entities.append(Entity(f"svc_{i:04d}", "service_account"))
    for i in range(n_devices):
        entities.append(Entity(f"dev_{i:04d}", "edge_device"))
    return entities


def _ts(base_day, hour, jitter_min=30):
    h = int(hour) % 24
    m = random.randint(0, 59)
    dt = base_day.replace(hour=h, minute=m, second=random.randint(0, 59))
    dt += timedelta(minutes=random.randint(-jitter_min, jitter_min))
    return dt


def gen_normal(entity, base_day):
    hour = random.uniform(entity.home_hour_start, entity.home_hour_end)
    ts = _ts(base_day, hour)
    geo = entity.home_geo
    return {
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "timestamp": ts,
        "source_ip": _geo_ip(geo),
        "geo_location": f"{geo[0]}/{geo[1]}",
        "resource_accessed": random.choice(entity.home_resources),
        "auth_method": entity.home_auth,
        "session_duration": max(5, np.random.normal(entity.typical_session_duration, 200)),
        "command_sequence": json.dumps(random.sample(COMMANDS_POOL[:6], k=random.randint(1, 3))),
        "device_fingerprint": json.dumps(entity.home_device),
        "auth_success": True,
        "label": "normal",
        "anomaly_type": "none",
    }


def gen_brute_force(entity, base_day):
    """Rapid repeated failed-auth attempts from one source in a short window."""
    rows = []
    ts0 = _ts(base_day, random.uniform(0, 23), jitter_min=0)
    ip = _geo_ip(random.choice(GEOS))
    n = random.randint(8, 30)
    for i in range(n):
        rows.append({
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
            "timestamp": ts0 + timedelta(seconds=i * random.uniform(1, 5)),
            "source_ip": ip,
            "geo_location": f"{entity.home_geo[0]}/{entity.home_geo[1]}",
            "resource_accessed": "vpn_gateway",
            "auth_method": entity.home_auth,
            "session_duration": 0,
            "command_sequence": json.dumps(["login"]),
            "device_fingerprint": json.dumps(entity.home_device),
            "auth_success": False if i < n - 1 else random.choice([True, False]),
            "label": "anomaly",
            "anomaly_type": "brute_force",
        })
    return rows


def gen_impossible_travel(entity, base_day):
    """Same entity logging in from geographically distant locations in an implausible time gap."""
    geo2 = random.choice([g for g in GEOS if g != entity.home_geo])
    ts0 = _ts(base_day, random.uniform(0, 23))
    ts1 = ts0 + timedelta(minutes=random.randint(5, 40))  # too short for real travel
    r1 = gen_normal(entity, base_day)
    r1["timestamp"] = ts0
    r2 = dict(r1)
    r2["timestamp"] = ts1
    r2["source_ip"] = _geo_ip(geo2)
    r2["geo_location"] = f"{geo2[0]}/{geo2[1]}"
    r2["label"] = "anomaly"
    r2["anomaly_type"] = "impossible_travel"
    return [r1, r2]


def gen_credential_stuffing(entity, base_day):
    """Many entity_ids, few source_ips, high failure rate (attacker perspective, injected per entity)."""
    ip = _geo_ip(random.choice(GEOS))
    ts0 = _ts(base_day, random.uniform(0, 23))
    return [{
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "timestamp": ts0 + timedelta(seconds=random.uniform(0, 20)),
        "source_ip": ip,
        "geo_location": "UNKNOWN/UNKNOWN",
        "resource_accessed": "vpn_gateway",
        "auth_method": entity.home_auth,
        "session_duration": 0,
        "command_sequence": json.dumps(["login"]),
        "device_fingerprint": json.dumps({"os": "unknown", "mac": fake.mac_address()}),
        "auth_success": random.random() < 0.1,
        "label": "anomaly",
        "anomaly_type": "credential_stuffing",
        "_shared_ip_group": ip,  # helper for feature engineering
    }]


def gen_lateral_movement(entity, base_day):
    """Compromised entity accessing an unusual sequence/breadth of resources it never touched before."""
    ts0 = _ts(base_day, random.uniform(0, 23))
    unusual_resources = [r for r in RESOURCES if r not in entity.home_resources]
    rows = []
    for i, res in enumerate(random.sample(unusual_resources, k=min(4, len(unusual_resources)))):
        rows.append({
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type,
            "timestamp": ts0 + timedelta(minutes=i * 2),
            "source_ip": _geo_ip(entity.home_geo),
            "geo_location": f"{entity.home_geo[0]}/{entity.home_geo[1]}",
            "resource_accessed": res,
            "auth_method": entity.home_auth,
            "session_duration": random.uniform(30, 300),
            "command_sequence": json.dumps(random.sample(COMMANDS_POOL, k=3)),
            "device_fingerprint": json.dumps(entity.home_device),
            "auth_success": True,
            "label": "anomaly",
            "anomaly_type": "lateral_movement",
        })
    return rows


def gen_device_spoofing(entity, base_day):
    """A device_id reappearing with a mismatched fingerprint (different OS/MAC than history)."""
    r = gen_normal(entity, base_day)
    r["device_fingerprint"] = json.dumps({
        "os": random.choice(["WindowsXP", "Android7", "unknown_fw"]),
        "mac": fake.mac_address(),
    })
    r["label"] = "anomaly"
    r["anomaly_type"] = "device_spoofing"
    return [r]


def gen_low_and_slow_exfiltration(entity, base_day, n_days=10):
    """Gradual, small, off-hours resource access building up over days/weeks."""
    rows = []
    off_hour = (entity.home_hour_end + random.randint(2, 5)) % 24
    for d in range(n_days):
        day = base_day + timedelta(days=d)
        r = gen_normal(entity, day)
        r["timestamp"] = _ts(day, off_hour, jitter_min=20)
        r["resource_accessed"] = random.choice(entity.home_resources)
        r["session_duration"] = random.uniform(60, 180)
        r["label"] = "anomaly"
        r["anomaly_type"] = "low_and_slow_exfiltration"
        rows.append(r)
    return rows


def gen_insider_drift(entity, base_day, n_days=14):
    """Legitimate entity slowly expanding privilege/resource footprint - ambiguous edge case."""
    rows = []
    extra_resources = [r for r in RESOURCES if r not in entity.home_resources][:3]
    for d in range(n_days):
        day = base_day + timedelta(days=d)
        r = gen_normal(entity, day)
        if d > n_days // 2 and extra_resources:
            r["resource_accessed"] = extra_resources[min(d - n_days // 2 - 1, len(extra_resources) - 1)]
        r["label"] = "edge_case"
        r["anomaly_type"] = "insider_drift"
        rows.append(r)
    return rows


ANOMALY_GENERATORS = {
    "brute_force": gen_brute_force,
    "impossible_travel": gen_impossible_travel,
    "credential_stuffing": gen_credential_stuffing,
    "lateral_movement": gen_lateral_movement,
    "device_spoofing": gen_device_spoofing,
    "low_and_slow_exfiltration": gen_low_and_slow_exfiltration,
    "insider_drift": gen_insider_drift,
}


def generate_dataset(
    n_users=120, n_service=25, n_devices=40,
    n_days=30, sessions_per_entity_per_day=3,
    anomaly_rate=0.015, insider_drift_entities=6,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    entities = make_entities(n_users, n_service, n_devices)
    start_day = datetime(2026, 6, 1)

    rows = []
    # 1. Bulk of normal traffic
    for d in range(n_days):
        day = start_day + timedelta(days=d)
        for entity in entities:
            for _ in range(sessions_per_entity_per_day):
                rows.append(gen_normal(entity, day))

    # 2. Inject point anomalies (brute force, impossible travel, credential stuffing,
    #    lateral movement, device spoofing) at controlled rate
    point_types = ["brute_force", "impossible_travel", "credential_stuffing",
                   "lateral_movement", "device_spoofing"]
    n_normal_sessions = len(rows)
    n_injections = int(n_normal_sessions * anomaly_rate)
    for _ in range(n_injections):
        entity = random.choice(entities)
        day = start_day + timedelta(days=random.randint(0, n_days - 1))
        atype = random.choice(point_types)
        rows.extend(ANOMALY_GENERATORS[atype](entity, day))

    # 3. Inject a handful of extended-pattern anomalies (low-and-slow exfiltration)
    for _ in range(max(2, n_injections // 20)):
        entity = random.choice(entities)
        day = start_day + timedelta(days=random.randint(0, n_days - 10))
        rows.extend(gen_low_and_slow_exfiltration(entity, day, n_days=random.randint(5, 10)))

    # 4. Inject insider-drift edge cases (ambiguous, for false-positive tuning)
    drift_entities = random.sample(entities, k=min(insider_drift_entities, len(entities)))
    for entity in drift_entities:
        day = start_day + timedelta(days=random.randint(0, n_days - 14))
        rows.extend(gen_insider_drift(entity, day, n_days=14))

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["session_id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
    # ensure column order matches suggested schema
    cols = ["session_id", "entity_id", "entity_type", "timestamp", "source_ip",
            "geo_location", "resource_accessed", "auth_method", "session_duration",
            "command_sequence", "device_fingerprint", "auth_success",
            "label", "anomaly_type"]
    df = df[[c for c in cols if c in df.columns]]
    return df, entities


if __name__ == "__main__":
    df, entities = generate_dataset()
    out_path = "/home/claude/cyber-anomaly/data/synthetic_access_logs.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df["label"].value_counts())
    print(df["anomaly_type"].value_counts())
