import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="SentinelIQ",
    page_icon="🛡️",
    layout="wide"
)

engine = create_engine('postgresql://vatsalj05@localhost:5432/sentineliq')
API_URL = "http://127.0.0.1:8000"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ SentinelIQ")
st.markdown("**Financial Risk Intelligence Platform** | Fraud Detection · Confidence Scoring · Customer Value Protection")
st.divider()

# ── Load Data ─────────────────────────────────────────────────────────────────
with engine.connect() as conn:
    smart_df = pd.read_sql("SELECT * FROM smart_risk_scores", conn)
    exposure_df = pd.read_sql("SELECT * FROM exposure_profiles", conn)
    clv_df = pd.read_sql("SELECT * FROM clv_scores", conn)

# ── Row 1: Portfolio Metrics ──────────────────────────────────────────────────
st.subheader("📊 Portfolio Overview")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Clients", f"{len(smart_df):,}")
col2.metric("🔴 Auto Freeze", 
            f"{(smart_df['smart_action']=='AUTO FREEZE').sum():,}")
col3.metric("🟠 Soft Freeze",
            f"{(smart_df['smart_action']=='SOFT FREEZE').sum():,}")
col4.metric("🟡 Step-Up Auth",
            f"{(smart_df['smart_action']=='STEP-UP AUTH').sum():,}")
col5.metric("⭐ Priority Review",
            f"{(smart_df['smart_action']=='FREEZE + PRIORITY REVIEW').sum():,}")

st.divider()

# ── Row 2: Charts ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Smart Action Distribution")
    action_counts = smart_df['smart_action'].value_counts()
    colors = {
        'FREEZE + PRIORITY REVIEW': '#7c3aed',
        'AUTO FREEZE': '#ef4444',
        'SOFT FREEZE': '#f97316',
        'STEP-UP AUTH': '#f59e0b',
        'ENHANCED MONITORING': '#3b82f6',
        'PASSIVE MONITORING': '#6b7280',
        'CLEAR': '#22c55e'
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(
        action_counts.index,
        action_counts.values,
        color=[colors.get(a, '#6b7280') for a in action_counts.index]
    )
    ax.set_xlabel('Client Count')
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    for bar, val in zip(bars, action_counts.values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', color='white', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

with col_right:
    st.subheader("Confidence Width Distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    high_conf = (smart_df['ci_width'] < 0.10).sum()
    med_conf = ((smart_df['ci_width'] >= 0.10) & (smart_df['ci_width'] < 0.20)).sum()
    low_conf = (smart_df['ci_width'] >= 0.20).sum()
    ax2.bar(['HIGH\n(CI < 0.10)', 'MEDIUM\n(0.10-0.20)', 'LOW\n(CI > 0.20)'],
            [high_conf, med_conf, low_conf],
            color=['#22c55e', '#f59e0b', '#ef4444'])
    ax2.set_ylabel('Client Count')
    ax2.set_title('How Confident Are Our Predictions?', color='white')
    ax2.set_facecolor('#0e1117')
    fig2.patch.set_facecolor('#0e1117')
    ax2.tick_params(colors='white')
    ax2.yaxis.label.set_color('white')
    ax2.title.set_color('white')
    plt.tight_layout()
    st.pyplot(fig2)

st.divider()

# ── Row 3: CLV vs Risk ────────────────────────────────────────────────────────
st.subheader("💎 Customer Value vs Fraud Risk")
st.markdown("Clients in the top-right are most dangerous to freeze wrongly — high value, uncertain fraud signal")

col_a, col_b = st.columns([2, 1])

with col_a:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    tier_color_map = {'PREMIUM': '#f59e0b', 'STANDARD': '#3b82f6', 'BASIC': '#6b7280'}
    for tier in ['BASIC', 'STANDARD', 'PREMIUM']:
        subset = smart_df[smart_df['clv_tier'] == tier]
        ax3.scatter(
            subset['fraud_probability'],
            subset['ci_width'],
            c=tier_color_map[tier],
            label=tier, alpha=0.4, s=8
        )
    ax3.axvline(x=0.70, color='#ef4444', linestyle='--', alpha=0.7, label='FREEZE threshold')
    ax3.set_xlabel('Fraud Probability', color='white')
    ax3.set_ylabel('Confidence Interval Width (uncertainty)', color='white')
    ax3.set_title('Fraud Risk vs Prediction Uncertainty by CLV Tier', color='white')
    ax3.legend(facecolor='#1e293b', labelcolor='white')
    ax3.set_facecolor('#0e1117')
    fig3.patch.set_facecolor('#0e1117')
    ax3.tick_params(colors='white')
    plt.tight_layout()
    st.pyplot(fig3)

with col_b:
    st.markdown("**CLV Tier Breakdown**")
    clv_counts = smart_df['clv_tier'].value_counts()
    for tier, count in clv_counts.items():
        icon = "⭐" if tier == "PREMIUM" else "🔵" if tier == "STANDARD" else "⚪"
        st.metric(f"{icon} {tier}", f"{count:,} clients")
    
    st.markdown("---")
    st.markdown("**False Positive Risk**")
    premium_freeze = smart_df[
        (smart_df['clv_tier'] == 'PREMIUM') & 
        (smart_df['risk_tier'] == 'FREEZE')
    ]
    st.metric("PREMIUM clients in FREEZE", 
              f"{len(premium_freeze):,}",
              "Need priority human review")

st.divider()

# ── Row 4: Breach Exposure Impact ─────────────────────────────────────────────
st.subheader("🔓 Breach Exposure Impact")
exposed = smart_df[smart_df['exposure_risk'] > 0]
not_exposed = smart_df[smart_df['exposure_risk'] == 0]

col_e1, col_e2, col_e3, col_e4 = st.columns(4)
col_e1.metric("Breach-Exposed Clients", f"{len(exposed):,}")
col_e2.metric("Avg Fraud Prob — Exposed", 
              f"{exposed['fraud_probability'].mean():.3f}")
col_e3.metric("Avg Fraud Prob — Not Exposed",
              f"{not_exposed['fraud_probability'].mean():.3f}")
col_e4.metric("Uplift from Breach",
              f"+{(exposed['fraud_probability'].mean() - not_exposed['fraud_probability'].mean()):.3f}")

st.divider()

# ── Row 5: Live Assessment ────────────────────────────────────────────────────
st.subheader("🔍 Live Client Risk Assessment")

client_input = st.text_input("Enter Client ID", placeholder="e.g. 3353286")

if st.button("Assess Risk", type="primary"):
    if client_input:
        with st.spinner("Querying SentinelIQ API..."):
            try:
                response = requests.post(
                    f"{API_URL}/assess",
                    json={"client_id": client_input}
                )
                if response.status_code == 200:
                    data = response.json()

                    tier_icons = {
                        'FREEZE': '🔴', 'ALERT': '🟡',
                        'MONITOR': '🔵', 'CLEAR': '🟢'
                    }

                    st.success(f"Assessment complete for client {client_input}")

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Fraud Probability",
                              f"{data['fraud_probability']:.4f}")
                    r2.metric("Exposure Risk",
                              f"{data['exposure_risk']:.4f}")
                    r3.metric("Composite Score",
                              f"{data['composite_score']:.4f}")
                    r4.metric("Risk Tier",
                              f"{tier_icons.get(data['risk_tier'], '')} {data['risk_tier']}")

                    st.warning(f"**Recommendation:** {data['recommendation']}")

                    st.markdown("**Top Risk Drivers (SHAP Explainability)**")
                    for driver in data['top_risk_drivers']:
                        icon = "🔴" if driver['impact'] > 0 else "🟢"
                        st.markdown(
                            f"{icon} **{driver['feature']}**: "
                            f"{driver['explanation']}"
                        )
                else:
                    st.error(f"Client not found: {response.status_code}")
            except Exception as e:
                st.error(f"API connection error: {e}")

st.divider()

# ── Row 6: High Risk Table ────────────────────────────────────────────────────
st.subheader("🚨 Clients Requiring Immediate Action")

action_filter = st.selectbox(
    "Filter by action",
    ["ALL HIGH RISK", "FREEZE + PRIORITY REVIEW", "AUTO FREEZE", 
     "SOFT FREEZE", "STEP-UP AUTH"]
)

if action_filter == "ALL HIGH RISK":
    display_df = smart_df[smart_df['risk_tier'] == 'FREEZE']
else:
    display_df = smart_df[smart_df['smart_action'] == action_filter]

display_df = display_df.sort_values('composite_score', ascending=False).head(25)

st.dataframe(
    display_df[[
        'client_id', 'fraud_probability', 'exposure_risk',
        'composite_score', 'ci_lower', 'ci_upper', 
        'clv_tier', 'smart_action'
    ]].rename(columns={
        'fraud_probability': 'Fraud Prob',
        'exposure_risk': 'Exposure',
        'composite_score': 'Composite',
        'ci_lower': 'CI Lower',
        'ci_upper': 'CI Upper',
        'clv_tier': 'CLV Tier',
        'smart_action': 'Action'
    }),
    use_container_width=True
)

st.divider()

# ── Model Drift Monitor ───────────────────────────────────────────────────────
st.subheader("📈 Model Drift Monitor")
st.markdown("Tracks model behaviour across time windows to detect when retraining is needed")

with engine.connect() as conn:
    drift_df = pd.read_sql('SELECT * FROM drift_monitor ORDER BY "window"', conn)

# Drift metrics row
baseline = drift_df['avg_fraud_prob'][:3].mean()
current = drift_df['avg_fraud_prob'].iloc[-1]
drift_pct = ((current - baseline) / baseline) * 100

d1, d2, d3, d4 = st.columns(4)
d1.metric("Baseline Fraud Prob (M1-3)", f"{baseline:.4f}")
d2.metric("Current Fraud Prob (M8)", f"{current:.4f}",
          f"{drift_pct:+.1f}% from baseline")
d3.metric("Avg Transaction Amt — M1", f"${drift_df['avg_transaction_amt'].iloc[0]:.0f}")
d4.metric("Avg Transaction Amt — M8", f"${drift_df['avg_transaction_amt'].iloc[-1]:.0f}",
          f"+{((drift_df['avg_transaction_amt'].iloc[-1]/drift_df['avg_transaction_amt'].iloc[0])-1)*100:.0f}% drift")

# Drift charts
col_d1, col_d2 = st.columns(2)

with col_d1:
    fig_d1, ax_d1 = plt.subplots(figsize=(6, 3))
    drift_threshold = baseline + 2 * drift_df['avg_fraud_prob'][:3].std()
    ax_d1.plot(drift_df['month'], drift_df['avg_fraud_prob'],
               'o-', color='#3b82f6', linewidth=2, label='Avg Fraud Prob')
    ax_d1.axhline(y=baseline, color='#22c55e', linestyle='--',
                  label=f'Baseline ({baseline:.4f})')
    ax_d1.axhline(y=drift_threshold, color='#ef4444', linestyle='--',
                  label=f'Alert ({drift_threshold:.4f})')
    ax_d1.set_title('Prediction Drift Over Time', color='white')
    ax_d1.set_facecolor('#0e1117')
    fig_d1.patch.set_facecolor('#0e1117')
    ax_d1.tick_params(colors='white', rotation=45)
    ax_d1.legend(facecolor='#1e293b', labelcolor='white', fontsize=7)
    plt.tight_layout()
    st.pyplot(fig_d1)

with col_d2:
    fig_d2, ax_d2 = plt.subplots(figsize=(6, 3))
    colors_bar = ['#ef4444' if i >= 4 else '#3b82f6' 
                  for i in range(len(drift_df))]
    ax_d2.bar(drift_df['month'], drift_df['avg_transaction_amt'],
              color=colors_bar)
    ax_d2.set_title('Data Drift — Avg Transaction Amount', color='white')
    ax_d2.set_facecolor('#0e1117')
    fig_d2.patch.set_facecolor('#0e1117')
    ax_d2.tick_params(colors='white', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_d2)

# Drift status
max_prob = drift_df['avg_fraud_prob'].max()
if max_prob > drift_threshold:
    st.error("⚠️ Drift detected — Model retraining recommended")
else:
    st.success(f"✅ Model stable — No significant drift detected across {len(drift_df)} time windows")