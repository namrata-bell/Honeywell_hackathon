"""
Analyst-Facing Dashboard (Streamlit)

Run with:
    streamlit run dashboard/app.py

Shows:
  - Ranked alert queue (by risk_score)
  - Risk score + contributing factors (explanation string) per alert
  - Predicted attack-type classification
  - Entity history view (drill-down)
  - Cold-start / concept-drift indicators
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "explainability"))

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="SOC Anomaly Dashboard", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "explained_alerts.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


df = load_data()

st.title("🛡️ Behavioral Anomaly Detection — SOC Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total sessions", f"{len(df):,}")
col2.metric("Flagged (top 1% risk)", f"{int(len(df)*0.01):,}")
col3.metric("True anomalies (ground truth)", f"{(df['label']!='normal').sum():,}")
col4.metric("Cold-start sessions", f"{df['is_cold_start'].sum():,}")

st.divider()

alert_budget = st.slider("Analyst alert budget (top % of events by risk score)", 0.1, 10.0, 1.0, 0.1)
n_alerts = max(1, int(len(df) * alert_budget / 100))
alerts = df.sort_values("risk_score", ascending=False).head(n_alerts)

tab1, tab2, tab3 = st.tabs(["🚨 Alert Queue", "📊 Risk Distribution", "🔎 Entity History"])

with tab1:
    st.subheader(f"Top {n_alerts} alerts (ranked by risk score)")
    display_cols = ["timestamp", "entity_id", "entity_type", "risk_score",
                     "anomaly_type", "explanation", "resource_accessed",
                     "geo_location", "is_cold_start"]
    st.dataframe(
        alerts[display_cols].style.background_gradient(subset=["risk_score"], cmap="Reds"),
        use_container_width=True,
        height=500,
    )
    precision = (alerts["label"] != "normal").mean()
    st.caption(f"Precision at this alert budget: **{precision:.1%}** "
               f"({(alerts['label']!='normal').sum()} of {n_alerts} alerts are true anomalies/edge-cases)")

with tab2:
    fig = px.histogram(df, x="risk_score", color="label", nbins=50,
                        title="Risk score distribution by ground-truth label",
                        barmode="overlay", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    type_counts = df[df["label"] != "normal"]["anomaly_type"].value_counts().reset_index()
    type_counts.columns = ["anomaly_type", "count"]
    fig2 = px.bar(type_counts, x="anomaly_type", y="count",
                  title="Injected anomaly types (ground truth)")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    entity_id = st.selectbox("Select entity", sorted(df["entity_id"].unique()))
    ent_df = df[df["entity_id"] == entity_id].sort_values("timestamp")
    st.write(f"**{len(ent_df)} sessions** for `{entity_id}` "
             f"({ent_df['entity_type'].iloc[0]})")
    st.line_chart(ent_df.set_index("timestamp")["risk_score"])
    st.dataframe(
        ent_df[["timestamp", "resource_accessed", "geo_location", "risk_score",
                "explanation", "label", "anomaly_type"]],
        use_container_width=True,
    )

st.divider()
st.caption(
    "Explainability: each alert's reason is derived from per-feature deviation "
    "from the entity's (or population's, for cold-start entities) baseline profile. "
    "Ground-truth labels are shown here for demo/evaluation purposes only — "
    "in production, `label`/`anomaly_type` are hidden at inference time."
)
