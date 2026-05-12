import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="SentinelIQ — Financial Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.stApp { background: #020408; }
.block-container { 
    padding: 0 2rem 2rem 2rem !important; 
    max-width: 1400px !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* ── HERO ── */
.hero-wrapper {
    background: linear-gradient(135deg, #020408 0%, #0d1117 50%, #020408 100%);
    border-bottom: 1px solid #1e2d3d;
    padding: 2.5rem 0 2rem 0;
    margin: -1rem -2rem 2rem -2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -10%;
    width: 40%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(99,102,241,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrapper::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 40%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(236,72,153,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #818cf8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.8rem;
    font-weight: 900;
    line-height: 1.05;
    background: linear-gradient(135deg, #e2e8f0 0%, #818cf8 40%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.8rem 0;
    letter-spacing: -0.02em;
}
.hero-desc {
    font-size: 1rem;
    color: #64748b;
    max-width: 600px;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}
.hero-stats {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
}
.hero-stat {
    display: flex;
    flex-direction: column;
}
.hero-stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
}
.hero-stat-label {
    font-size: 0.72rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── KPI CARDS ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    border-radius: 16px;
    padding: 1.3rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-total {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155;
}
.kpi-total::before { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.kpi-freeze {
    background: linear-gradient(135deg, #1a0505, #2d0f0f);
    border: 1px solid #7f1d1d;
}
.kpi-freeze::before { background: linear-gradient(90deg, #ef4444, #dc2626); }
.kpi-priority {
    background: linear-gradient(135deg, #1a0a1a, #2d0f2d);
    border: 1px solid #6b21a8;
}
.kpi-priority::before { background: linear-gradient(90deg, #a855f7, #7c3aed); }
.kpi-alert {
    background: linear-gradient(135deg, #1a1205, #2d200a);
    border: 1px solid #78350f;
}
.kpi-alert::before { background: linear-gradient(90deg, #f59e0b, #d97706); }
.kpi-safe {
    background: linear-gradient(135deg, #051a0a, #0a2d14);
    border: 1px solid #14532d;
}
.kpi-safe::before { background: linear-gradient(90deg, #22c55e, #16a34a); }
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.kpi-value {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
}
.kpi-sub {
    font-size: 0.72rem;
    margin-top: 0.4rem;
    opacity: 0.6;
}
.kpi-total .kpi-label { color: #818cf8; }
.kpi-total .kpi-value { color: #e2e8f0; }
.kpi-freeze .kpi-label { color: #f87171; }
.kpi-freeze .kpi-value { color: #fca5a5; }
.kpi-priority .kpi-label { color: #c084fc; }
.kpi-priority .kpi-value { color: #d8b4fe; }
.kpi-alert .kpi-label { color: #fbbf24; }
.kpi-alert .kpi-value { color: #fde68a; }
.kpi-safe .kpi-label { color: #4ade80; }
.kpi-safe .kpi-value { color: #86efac; }

/* ── SECTION ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1e293b;
}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #cbd5e1;
    letter-spacing: -0.01em;
}
.section-badge {
    font-size: 0.65rem;
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    padding: 0.15rem 0.5rem;
    border-radius: 100px;
    font-weight: 600;
}

/* ── RISK CARD (assessment result) ── */
.risk-result {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.risk-result-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
}
.risk-tier-badge {
    padding: 0.4rem 1rem;
    border-radius: 8px;
    font-weight: 800;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
}
.shap-bar {
    margin: 0.3rem 0;
    padding: 0.6rem 0.8rem;
    background: #0f172a;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    border-left: 3px solid;
}

/* ── TABLE ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── DIVIDER ── */
hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }

/* ── INPUTS ── */
.stTextInput input {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}
.stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stButton button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}

/* ── SELECTBOX ── */
.stSelectbox select, div[data-baseweb="select"] {
    background: #0f172a !important;
    border-color: #334155 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ── DB CONNECTION ─────────────────────────────────────────────────────────────
try:
    DATABASE_URL = st.secrets["DATABASE_URL"]
except:
    from dotenv import load_dotenv
    load_dotenv('../.env')
    DATABASE_URL = os.getenv('DATABASE_URL')

engine = create_engine(DATABASE_URL)
API_URL = "http://127.0.0.1:8000"

@st.cache_data(ttl=300)
def load_data():
    with engine.connect() as conn:
        smart_df = pd.read_sql("SELECT * FROM smart_risk_scores", conn)
        drift_df = pd.read_sql('SELECT * FROM drift_monitor ORDER BY "window"', conn)
    return smart_df, drift_df

smart_df, drift_df = load_data()

# ── COMPUTED METRICS ──────────────────────────────────────────────────────────
total = len(smart_df)
auto_freeze = (smart_df['smart_action'] == 'AUTO FREEZE').sum()
priority = (smart_df['smart_action'] == 'FREEZE + PRIORITY REVIEW').sum()
step_up = (smart_df['smart_action'] == 'STEP-UP AUTH').sum()
enhanced = (smart_df['smart_action'] == 'ENHANCED MONITORING').sum()
clear = (smart_df['smart_action'] == 'CLEAR').sum()
avg_fraud = smart_df['fraud_probability'].mean()
high_conf = (smart_df['ci_width'] < 0.10).sum()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrapper">
    <div class="hero-badge">🛡️ &nbsp; Financial Risk Intelligence</div>
    <h1 class="hero-title">SentinelIQ</h1>
    <p class="hero-desc">
        A production-grade fraud detection system combining XGBoost machine learning, 
        Bayesian confidence scoring, and customer lifetime value weighting — 
        making smarter risk decisions that protect both your clients and your business.
    </p>
    <div class="hero-stats">
        <div class="hero-stat">
            <span class="hero-stat-value">0.9528</span>
            <span class="hero-stat-label">AUC-ROC Score</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-value">0.7017</span>
            <span class="hero-stat-label">AUC-PR Score</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-value">54</span>
            <span class="hero-stat-label">Features</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-value">1,500</span>
            <span class="hero-stat-label">Trees Trained</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-value">{total:,}</span>
            <span class="hero-stat-label">Clients Scored</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card kpi-total">
        <div class="kpi-label">Total Scored</div>
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-sub">Avg risk: {avg_fraud:.4f}</div>
    </div>
    <div class="kpi-card kpi-freeze">
        <div class="kpi-label">🔴 Auto Freeze</div>
        <div class="kpi-value">{auto_freeze}</div>
        <div class="kpi-sub">{auto_freeze/total*100:.2f}% of portfolio</div>
    </div>
    <div class="kpi-card kpi-priority">
        <div class="kpi-label">⭐ Priority Review</div>
        <div class="kpi-value">{priority}</div>
        <div class="kpi-sub">Senior analyst required</div>
    </div>
    <div class="kpi-card kpi-alert">
        <div class="kpi-label">🟡 Step-Up Auth</div>
        <div class="kpi-value">{step_up:,}</div>
        <div class="kpi-sub">{step_up/total*100:.1f}% need verification</div>
    </div>
    <div class="kpi-card kpi-safe">
        <div class="kpi-label">🟢 Clear</div>
        <div class="kpi-value">{clear:,}</div>
        <div class="kpi-sub">{clear/total*100:.1f}% low risk</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── ROW: ACTION + CONFIDENCE ──────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown('<div class="section-header"><span class="section-title">Smart Action Distribution</span><span class="section-badge">7 tiers</span></div>', unsafe_allow_html=True)
    
    action_counts = smart_df['smart_action'].value_counts().reset_index()
    action_counts.columns = ['Action', 'Count']
    color_map = {
        'FREEZE + PRIORITY REVIEW': '#a855f7',
        'AUTO FREEZE': '#ef4444',
        'SOFT FREEZE': '#f97316',
        'STEP-UP AUTH': '#f59e0b',
        'ENHANCED MONITORING': '#6366f1',
        'PASSIVE MONITORING': '#475569',
        'CLEAR': '#22c55e'
    }
    fig1 = go.Figure(go.Bar(
        x=action_counts['Count'],
        y=action_counts['Action'],
        orientation='h',
        marker_color=[color_map.get(a, '#475569') for a in action_counts['Action']],
        marker_line_width=0,
        text=action_counts['Count'].apply(lambda x: f'{x:,}'),
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11, family='Inter')
    ))
    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Inter'),
        margin=dict(l=0, r=60, t=10, b=10), height=280,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=11))
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_r:
    st.markdown('<div class="section-header"><span class="section-title">Model Confidence</span><span class="section-badge">Bootstrap CI</span></div>', unsafe_allow_html=True)
    
    high_c = (smart_df['ci_width'] < 0.10).sum()
    med_c = ((smart_df['ci_width'] >= 0.10) & (smart_df['ci_width'] < 0.20)).sum()
    low_c = (smart_df['ci_width'] >= 0.20).sum()

    fig2 = go.Figure()
    for label, val, color in [('HIGH', high_c, '#22c55e'), ('MEDIUM', med_c, '#f59e0b'), ('LOW', low_c, '#ef4444')]:
        fig2.add_trace(go.Bar(
            name=label, x=[label], y=[val],
            marker_color=color, marker_line_width=0,
            text=[f'{val:,}'], textposition='outside',
            textfont=dict(color='#94a3b8', size=12)
        ))
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Inter'),
        margin=dict(l=10, r=10, t=10, b=10), height=280,
        showlegend=False, barmode='group',
        xaxis=dict(showgrid=False, tickfont=dict(size=13, color='#cbd5e1')),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── FRAUD PROBABILITY HISTOGRAM ───────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="section-title">Fraud Probability Distribution</span><span class="section-badge">590k transactions</span></div>', unsafe_allow_html=True)

fig3 = px.histogram(smart_df, x='fraud_probability', nbins=80,
                    color_discrete_sequence=['#6366f1'])
fig3.add_vline(x=0.70, line_dash="dash", line_color="#ef4444", line_width=1.5,
               annotation_text="FREEZE (0.70)", annotation_font_color="#ef4444",
               annotation_font_size=11)
fig3.add_vline(x=0.45, line_dash="dash", line_color="#f59e0b", line_width=1.5,
               annotation_text="ALERT (0.45)", annotation_font_color="#f59e0b",
               annotation_font_size=11)
fig3.add_vrect(x0=0.70, x1=1.0, fillcolor="rgba(239,68,68,0.05)", line_width=0)
fig3.add_vrect(x0=0.45, x1=0.70, fillcolor="rgba(245,158,11,0.05)", line_width=0)
fig3.update_traces(marker_line_width=0)
fig3.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='Inter'),
    height=220, margin=dict(l=10, r=10, t=20, b=10),
    showlegend=False,
    xaxis=dict(showgrid=False, title='Fraud Probability', color='#64748b'),
    yaxis=dict(showgrid=True, gridcolor='#0f172a', title='Count', color='#64748b')
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── CLV vs RISK ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="section-title">Customer Value vs Fraud Risk</span><span class="section-badge">False Positive Protection</span></div>', unsafe_allow_html=True)
st.caption("PREMIUM clients near the FREEZE threshold receive human review before any action — protecting high-value relationships from false positives")

col_s, col_info = st.columns([3, 1])

with col_s:
    sample = smart_df.sample(min(8000, len(smart_df)), random_state=42)
    fig4 = px.scatter(
        sample, x='fraud_probability', y='ci_width',
        color='clv_tier',
        color_discrete_map={'PREMIUM': '#f59e0b', 'STANDARD': '#6366f1', 'BASIC': '#334155'},
        opacity=0.7,
        labels={'fraud_probability': 'Fraud Probability', 
                'ci_width': 'Prediction Uncertainty (CI Width)',
                'clv_tier': 'Customer Value'}
    )
    fig4.add_vline(x=0.70, line_dash="dash", line_color="#ef4444", 
                   line_width=1.5, opacity=0.6)
    fig4.add_vrect(x0=0.70, x1=1.0, fillcolor="rgba(239,68,68,0.04)", line_width=0)
    fig4.update_traces(marker=dict(size=5, line=dict(width=0)))
    fig4.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Inter'),
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor='rgba(15,23,42,0.8)', bordercolor='#334155', 
                   borderwidth=1, title_text='CLV Tier'),
        xaxis=dict(showgrid=True, gridcolor='#0f172a', color='#64748b'),
        yaxis=dict(showgrid=True, gridcolor='#0f172a', color='#64748b')
    )
    st.plotly_chart(fig4, use_container_width=True)

with col_info:
    st.markdown("<br>", unsafe_allow_html=True)
    for tier, icon, color, bg in [
        ('PREMIUM', '⭐', '#f59e0b', 'rgba(245,158,11,0.1)'),
        ('STANDARD', '🔵', '#6366f1', 'rgba(99,102,241,0.1)'),
        ('BASIC', '⚪', '#475569', 'rgba(71,85,105,0.1)')
    ]:
        count = (smart_df['clv_tier'] == tier).sum()
        st.markdown(f"""
        <div style='background:{bg};border-radius:10px;padding:0.8rem;margin-bottom:0.6rem;border:1px solid {color}33'>
            <div style='font-size:0.7rem;color:{color};font-weight:600;text-transform:uppercase'>{icon} {tier}</div>
            <div style='font-size:1.5rem;font-weight:800;color:#e2e8f0'>{count:,}</div>
            <div style='font-size:0.7rem;color:#64748b'>clients</div>
        </div>
        """, unsafe_allow_html=True)

    premium_freeze = len(smart_df[(smart_df['clv_tier']=='PREMIUM') & (smart_df['risk_tier']=='FREEZE')])
    st.markdown(f"""
    <div style='background:rgba(239,68,68,0.08);border:1px solid #7f1d1d;border-radius:10px;padding:0.8rem;margin-top:0.5rem'>
        <div style='font-size:0.7rem;color:#f87171;font-weight:600'>⚠️ FALSE POSITIVE RISK</div>
        <div style='font-size:1.8rem;font-weight:800;color:#fca5a5'>{premium_freeze}</div>
        <div style='font-size:0.7rem;color:#64748b'>PREMIUM in FREEZE</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── BREACH EXPOSURE ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="section-title">Breach Exposure Impact</span><span class="section-badge">Post-breach analysis</span></div>', unsafe_allow_html=True)

exposed = smart_df[smart_df['exposure_risk'] > 0]
not_exposed = smart_df[smart_df['exposure_risk'] == 0]
uplift = exposed['fraud_probability'].mean() - not_exposed['fraud_probability'].mean()

b1, b2, b3, b4 = st.columns(4)
b1.metric("Breach-Exposed Clients", f"{len(exposed):,}", 
          f"{len(exposed)/total*100:.1f}% of portfolio")
b2.metric("Avg Fraud Prob — Exposed", f"{exposed['fraud_probability'].mean():.4f}")
b3.metric("Avg Fraud Prob — Safe", f"{not_exposed['fraud_probability'].mean():.4f}")
b4.metric("Risk Uplift from Breach", f"{uplift:+.4f}",
          delta_color="inverse")

st.divider()

# ── LIVE ASSESSMENT ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="section-title">Live Client Risk Assessment</span><span class="section-badge">Real-time API</span></div>', unsafe_allow_html=True)

col_in, col_hint = st.columns([2, 3])
with col_in:
    client_input = st.text_input("client_id_input", placeholder="Enter Client ID — try: 3353286", 
                                  label_visibility="collapsed")
    btn = st.button("⚡  Run Risk Assessment", type="primary", use_container_width=True)

with col_hint:
    st.markdown("""
    <div style='padding:0.8rem;background:#0f172a;border-radius:10px;border:1px solid #1e293b;margin-top:0.2rem'>
        <div style='font-size:0.7rem;color:#475569;margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.05em'>Sample high-risk IDs</div>
        <div style='display:flex;gap:0.5rem;flex-wrap:wrap'>
            <span style='background:#1e293b;color:#818cf8;padding:0.2rem 0.6rem;border-radius:6px;font-size:0.8rem;font-family:monospace'>3353286</span>
            <span style='background:#1e293b;color:#818cf8;padding:0.2rem 0.6rem;border-radius:6px;font-size:0.8rem;font-family:monospace'>3314821</span>
            <span style='background:#1e293b;color:#818cf8;padding:0.2rem 0.6rem;border-radius:6px;font-size:0.8rem;font-family:monospace'>3367366</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if btn and client_input:
    with st.spinner(""):
        try:
            resp = requests.post(f"{API_URL}/assess", json={"client_id": client_input}, timeout=5)
            if resp.status_code == 200:
                d = resp.json()
                tier_cfg = {
                    'FREEZE': ('#ef4444', '#7f1d1d', 'rgba(239,68,68,0.1)'),
                    'ALERT': ('#f59e0b', '#78350f', 'rgba(245,158,11,0.1)'),
                    'MONITOR': ('#6366f1', '#312e81', 'rgba(99,102,241,0.1)'),
                    'CLEAR': ('#22c55e', '#14532d', 'rgba(34,197,94,0.1)')
                }
                tc, tb, tbg = tier_cfg.get(d['risk_tier'], ('#94a3b8', '#334155', 'rgba(148,163,184,0.1)'))

                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#0f172a,#1e1b4b);border:1px solid #312e81;border-radius:16px;padding:1.5rem;margin:1rem 0'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:1.2rem'>
                        <div>
                            <div style='color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em'>Client Assessment</div>
                            <div style='color:#e2e8f0;font-size:1.8rem;font-weight:800;font-family:monospace'>#{client_input}</div>
                        </div>
                        <div style='background:{tbg};border:2px solid {tc};border-radius:10px;padding:0.5rem 1.2rem;text-align:center'>
                            <div style='color:{tc};font-size:1.1rem;font-weight:900;letter-spacing:0.05em'>{d["risk_tier"]}</div>
                        </div>
                    </div>
                    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1rem'>
                        <div style='background:#0f172a;border-radius:10px;padding:0.8rem;text-align:center'>
                            <div style='color:#475569;font-size:0.7rem;text-transform:uppercase'>Fraud Prob</div>
                            <div style='color:#e2e8f0;font-size:1.4rem;font-weight:700'>{d["fraud_probability"]:.4f}</div>
                        </div>
                        <div style='background:#0f172a;border-radius:10px;padding:0.8rem;text-align:center'>
                            <div style='color:#475569;font-size:0.7rem;text-transform:uppercase'>Exposure Risk</div>
                            <div style='color:#e2e8f0;font-size:1.4rem;font-weight:700'>{d["exposure_risk"]:.4f}</div>
                        </div>
                        <div style='background:#0f172a;border-radius:10px;padding:0.8rem;text-align:center'>
                            <div style='color:#475569;font-size:0.7rem;text-transform:uppercase'>Composite</div>
                            <div style='color:{tc};font-size:1.4rem;font-weight:700'>{d["composite_score"]:.4f}</div>
                        </div>
                    </div>
                    <div style='background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:10px;padding:0.8rem;margin-bottom:1rem'>
                        <div style='color:#818cf8;font-size:0.7rem;font-weight:600;text-transform:uppercase;margin-bottom:0.3rem'>Recommendation</div>
                        <div style='color:#cbd5e1;font-size:0.9rem'>{d["recommendation"]}</div>
                    </div>
                    <div style='color:#475569;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem'>SHAP Risk Drivers</div>
                    {"".join([f'''
                    <div style='margin:0.3rem 0;padding:0.6rem 0.8rem;background:#0f172a;border-radius:8px;display:flex;align-items:center;gap:0.8rem;border-left:3px solid {"#ef4444" if x["impact"]>0 else "#22c55e"}'>
                        <div style='width:60px;color:{"#f87171" if x["impact"]>0 else "#4ade80"};font-weight:700;font-size:0.85rem;font-family:monospace'>{x["feature"]}</div>
                        <div style='flex:1;background:#020408;border-radius:3px;height:6px'>
                            <div style='width:{min(abs(x["impact"])*150,100)}%;background:{"#ef4444" if x["impact"]>0 else "#22c55e"};height:6px;border-radius:3px'></div>
                        </div>
                        <div style='color:{"#f87171" if x["impact"]>0 else "#4ade80"};font-size:0.8rem;font-weight:600'>{x["impact"]:+.3f}</div>
                        <div style='color:#475569;font-size:0.75rem'>{"increases" if x["impact"]>0 else "reduces"} risk</div>
                    </div>
                    ''' for x in d["top_risk_drivers"]])}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Client not found in database")
        except Exception as e:
            st.warning("⚡ API offline — start FastAPI locally to enable live scoring")

st.divider()

# ── HIGH RISK TABLE ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="section-title">Clients Requiring Immediate Action</span><span class="section-badge">Sortable</span></div>', unsafe_allow_html=True)

col_f, _ = st.columns([2, 3])
with col_f:
    action_filter = st.selectbox("filter", 
        ["ALL HIGH RISK", "FREEZE + PRIORITY REVIEW", "AUTO FREEZE", 
         "STEP-UP AUTH", "ENHANCED MONITORING"],
        label_visibility="collapsed")

display_df = smart_df[smart_df['risk_tier']=='FREEZE'] if action_filter == "ALL HIGH RISK" \
             else smart_df[smart_df['smart_action']==action_filter]
display_df = display_df.sort_values('composite_score', ascending=False).head(25)

st.dataframe(
    display_df[['client_id','fraud_probability','exposure_risk','composite_score',
                'ci_lower','ci_upper','clv_tier','smart_action']]
    .rename(columns={
        'fraud_probability':'Fraud Prob','exposure_risk':'Exposure',
        'composite_score':'Composite','ci_lower':'CI Lower',
        'ci_upper':'CI Upper','clv_tier':'CLV Tier','smart_action':'Action'
    }),
    use_container_width=True, hide_index=True
)

st.divider()

# ── DRIFT MONITOR ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header"><span class="section-title">Model Drift Monitor</span><span class="section-badge">8 time windows</span></div>', unsafe_allow_html=True)

baseline = drift_df['avg_fraud_prob'][:3].mean()
current = drift_df['avg_fraud_prob'].iloc[-1]
drift_threshold = baseline + 2 * drift_df['avg_fraud_prob'][:3].std()
drift_pct = ((current - baseline) / baseline) * 100

d1, d2, d3, d4 = st.columns(4)
d1.metric("Baseline Fraud Prob", f"{baseline:.4f}")
d2.metric("Current Fraud Prob", f"{current:.4f}", f"{drift_pct:+.1f}% from baseline")
d3.metric("Transaction Amt — M1", f"${drift_df['avg_transaction_amt'].iloc[0]:.0f}")
d4.metric("Transaction Amt — M8", f"${drift_df['avg_transaction_amt'].iloc[-1]:.0f}",
          f"+{((drift_df['avg_transaction_amt'].iloc[-1]/drift_df['avg_transaction_amt'].iloc[0])-1)*100:.0f}% drift")

col_d1, col_d2 = st.columns(2)

with col_d1:
    fig_d1 = go.Figure()
    fig_d1.add_trace(go.Scatter(
        x=drift_df['month'], y=drift_df['avg_fraud_prob'],
        mode='lines+markers', name='Fraud Prob',
        line=dict(color='#6366f1', width=2.5),
        marker=dict(size=8, color='#6366f1', line=dict(color='#e2e8f0', width=2)),
        fill='tozeroy', fillcolor='rgba(99,102,241,0.05)'
    ))
    fig_d1.add_hline(y=baseline, line_dash="dot", line_color="#22c55e", line_width=1.5,
                     annotation_text=f"Baseline", annotation_font_color="#22c55e", 
                     annotation_font_size=10)
    fig_d1.add_hline(y=drift_threshold, line_dash="dot", line_color="#ef4444", line_width=1.5,
                     annotation_text=f"Alert threshold", annotation_font_color="#ef4444",
                     annotation_font_size=10)
    fig_d1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Inter'), height=260,
        margin=dict(l=10, r=80, t=20, b=10), showlegend=False,
        title=dict(text='Prediction Drift Over Time', font=dict(color='#cbd5e1', size=12)),
        xaxis=dict(showgrid=False, color='#64748b'),
        yaxis=dict(showgrid=True, gridcolor='#0f172a', color='#64748b')
    )
    st.plotly_chart(fig_d1, use_container_width=True)

with col_d2:
    fig_d2 = go.Figure(go.Bar(
        x=drift_df['month'], y=drift_df['avg_transaction_amt'],
        marker_color=['#ef4444' if i >= 4 else '#6366f1' for i in range(len(drift_df))],
        marker_line_width=0,
        text=drift_df['avg_transaction_amt'].apply(lambda x: f'${x:.0f}'),
        textposition='outside', textfont=dict(color='#94a3b8', size=10)
    ))
    fig_d2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Inter'), height=260,
        margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
        title=dict(text='Data Drift — Avg Transaction Amount', font=dict(color='#cbd5e1', size=12)),
        xaxis=dict(showgrid=False, color='#64748b'),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    st.plotly_chart(fig_d2, use_container_width=True)

max_prob = drift_df['avg_fraud_prob'].max()
if max_prob > drift_threshold:
    st.error("⚠️ Drift detected — Model retraining recommended")
else:
    st.success(f"✅ Model stable — No significant drift detected across {len(drift_df)} time windows")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1.5rem 0;'>
    <div style='color:#1e293b;font-size:0.75rem;margin-bottom:0.5rem'>
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    </div>
    <div style='color:#334155;font-size:0.8rem'>
        <span style='color:#6366f1;font-weight:700'>SentinelIQ</span> &nbsp;·&nbsp; 
        Built by <span style='color:#818cf8'>Vatsal Jain</span> &nbsp;·&nbsp;
        <a href='https://github.com/Vatsal-cs/sentineliq' style='color:#6366f1;text-decoration:none'>GitHub ↗</a>
        &nbsp;·&nbsp; XGBoost · SHAP · FastAPI · PostgreSQL
    </div>
    <div style='color:#1e293b;font-size:0.7rem;margin-top:0.5rem'>
        AUC-ROC 0.9528 · AUC-PR 0.7017 · 54 features · 1,500 trees
    </div>
</div>
""", unsafe_allow_html=True)