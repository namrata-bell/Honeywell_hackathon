# AI-Powered Behavioral Anomaly Detection — System Report

## 1. Architecture Overview

```
generate_synthetic_logs.py  →  synthetic_access_logs.csv
            ↓
baseline_profile.py (feature engineering, per-entity rolling profile)
            ↓
features.csv
            ↓
sequence_detector.py
   ├─ IsolationForest  → risk_score (0–100, unsupervised)
   └─ RandomForest      → anomaly_type (supervised, labelled anomalies only)
            ↓
scored_features.csv
            ↓
explain.py → explained_alerts.csv (per-alert plain-English reason)
            ↓
dashboard/app.py (Streamlit) — ranked alert queue, drill-down, drift/cold-start indicators
```

## 2. Design Decisions & Rationale

**Why IsolationForest over an LSTM/GRU as the primary detector:**
Given the hackathon time budget, IsolationForest was chosen as the primary
scorer because it trains in seconds, requires no GPU, and is markedly less
prone to overfitting on a dataset with <2% positive labels. A sequence-aware
deep model remains the stronger long-term choice (documented as a stretch
upgrade) but introduces real overfitting/tuning risk on a small labelled set
under time pressure — reliability was prioritized over sophistication.

**Handling extreme class imbalance:** the detector is unsupervised
(IsolationForest doesn't need labelled positives), so the ~1.5–3% synthetic
injection rate doesn't need oversampling/SMOTE. The supervised classifier
(anomaly *type*) is only ever run on already-flagged sessions, where the
label distribution is far less skewed.

**Handling concept drift:** per-entity baseline profiles (hour distribution,
geo set, resource set, device set) are built from *rolling prior history*,
not a static training snapshot — so an entity's "normal" naturally shifts as
their real behaviour shifts. In production, the IsolationForest itself
should be periodically re-fit (e.g. nightly) on a trailing window; this is
not yet automated in the PoC and is a known limitation.

**Handling cold-start:** entities with fewer than 5 prior sessions get
`is_cold_start=1` and their deviation features are neutralized (rather than
falsely flagged as "everything is novel"). This is surfaced explicitly in
the dashboard so analysts know to treat these scores with more caution.

**Explainability:** each alert carries a plain-English reason string derived
from per-feature deviation (z-score) versus the entity's baseline, e.g.
*"flagged due to geo-velocity: impossible travel + new/unseen device
fingerprint."* SHAP (TreeExplainer) is used on the RandomForest
classifier for anomaly-type attribution, since IsolationForest doesn't
support fast, exact SHAP attribution at this ensemble size.

## 3. System Design & Scalability (Real-Time Streaming Feasibility)

**Why this matters:** the PoC pipeline in this repo runs over a static CSV
end-to-end. The PS asks for near-real-time detection, so this section
describes how the same components map onto a streaming deployment — this is
a design specification, not yet built/benchmarked in this submission.

**Proposed production data flow:**

```
access event  →  stream ingest (Kafka/Kinesis-style topic, partitioned by entity_id)
              →  stateful stream processor: updates the per-entity rolling
                 profile (hour set, geo set, resource set, device set,
                 last-seen location/time) in a low-latency key-value store
                 (e.g. Redis), keyed by entity_id
              →  feature vector assembled for this single event
                 (same FEATURE_COLUMNS as baseline_profile.py)
              →  stateless scorer service: loaded IsolationForest +
                 RandomForest (.predict()/.decision_function() only,
                 no retraining in the hot path)
              →  risk_score + explanation → alert topic → dashboard/SOC queue
```

**Why this is feasible with the current model choice:**
- Both IsolationForest and RandomForest inference are O(number of trees ×
  tree depth) per event — sub-millisecond to low-single-digit-millisecond
  scoring on commodity CPU, with no GPU dependency. This was also the reason
  IsolationForest was picked over a deep sequence model (see §2): it is
  cheap enough to run inline, per-event, at ingestion time.
- The scorer is stateless (all "memory" of an entity's history lives in the
  external per-entity profile store, not in the model), so it scales
  horizontally — add more scorer replicas behind the stream, no
  cross-replica coordination needed.
- The only stateful component is the per-entity profile store, which is a
  simple keyed read-modify-write per event and scales the same way any
  session-store/cache does (sharded by entity_id).
- Target latency budget: profile lookup + feature assembly + inference in
  well under 100ms per event end-to-end — comfortably inside a "near
  real-time" SOC alerting requirement (seconds-scale), with large headroom.

**Concept drift in this architecture, made concrete:** the rolling per-entity
profile already adapts continuously (§2), but the IsolationForest/RandomForest
themselves are static once trained. The production plan is a scheduled batch
retrain — e.g. nightly, on a trailing 30-day window of scored sessions — with
a champion/challenger check (new model must match or beat the current
model's precision at the target alert budget on a held-out slice) before it
replaces the live model version. This is called out as a known limitation in
§6 because it is not automated in this PoC; this section specifies *how* it
would be, which is what the evaluation criterion asks for.

**Where this deviates from the suggested approach, stated directly:** the PS
lists a sequence-aware detector (LSTM/GRU, Transformer, or graph-based) as
the suggested detection model. This submission's primary detector is
IsolationForest + RandomForest instead (rationale in §2: <2% labelled
positives makes a deep sequence model prone to overfitting under a hackathon
time budget, and tree ensembles are cheap enough to run inline in the
streaming design above). A GRU/LSTM autoencoder over `command_sequence` is
documented as the natural next upgrade once enough labelled/normal sequence
data exists to train it safely, but it is not implemented in this
submission — noted here explicitly rather than left implicit.

## 4. Results (on generated dataset: 18,280 sessions, 8 patterns)

| Alert Budget | Precision | Recall | ROC-AUC |
|---|---|---|---|
| 0.5% | 97.8% | 5.6% | 0.945 |
| 1.0% | 92.9% | 10.7% | 0.945 |
| 2.0% | 86.9% | 20.0% | 0.945 |
| 5.0% | 74.0% | 42.7% | 0.945 |

Anomaly-type classifier (held-out labelled anomalies): **99% overall
accuracy**, with the weakest class being `low_and_slow_exfiltration`
(85% precision) since individual sessions in that pattern look close to
normal by design — it's meant to be caught by its *cumulative* pattern
over days, not a single-session score. This is a known limitation (see §6).

## 5. Assumptions

- Synthetic data injection rate: 1.5% point-anomalies + separate small
  batches of extended-pattern anomalies (low-and-slow, insider drift),
  per the suggested 0.5–3% guidance.
- `label`/`anomaly_type` are used only for training/evaluation and are
  never passed as model input features at inference time.
- Population-level fallback profile is implicitly handled by neutralizing
  deviation features for cold-start entities rather than a separate model.

## 6. Known Limitations

- Low-and-slow exfiltration and insider drift are inherently harder to catch
  per-session; a proper fix requires an entity-level rolling aggregate score
  (e.g. cumulative anomalous-resource count over a trailing 14-day window)
  rather than a purely per-session score — flagged as future work.
- IsolationForest is not re-fit on a rolling window in this PoC; production
  deployment would need scheduled retraining to fully track concept drift.
- `distinct_entities_per_ip` did not vary as intended in this synthetic run
  (credential-stuffing sessions were generated per-entity rather than
  batched by shared IP) — the feature is correctly computed but under-
  informative in the current dataset; the generator's credential-stuffing
  function is a good next target to strengthen (batch multiple entity_ids
  per session under one shared IP).

## 7. Files

- `data_gen/generate_synthetic_logs.py` — synthetic data generator
- `models/baseline_profile.py` — feature engineering + cold-start handling
- `models/sequence_detector.py` — IsolationForest scorer + RandomForest classifier
- `explainability/explain.py` — per-alert explanation generation
- `dashboard/app.py` — Streamlit analyst dashboard
- `data/*.csv` — generated data at each pipeline stage
