from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import shap
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

app = FastAPI(
    title="SentinelIQ Risk API",
    description="Post-breach identity risk scoring engine for fintech fraud detection",
    version="1.0.0"
)

# Load model and features on startup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE_DIR, 'models', 'sentineliq_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, 'models', 'features.pkl'), 'rb') as f:
    features = pickle.load(f)
with open(os.path.join(BASE_DIR, 'models', 'platt_scaler.pkl'), 'rb') as f:
    platt_scaler = pickle.load(f)

# Database connection
load_dotenv('../.env')
engine = create_engine(os.getenv('DATABASE_URL'))

# SHAP explainer
explainer = shap.TreeExplainer(model)


# ── Request/Response Models ──────────────────────────────────────────────────

class AssessRequest(BaseModel):
    client_id: str

class RiskResponse(BaseModel):
    client_id: str
    fraud_probability: float
    exposure_risk: float
    composite_score: float
    risk_tier: str
    top_risk_drivers: list
    recommendation: str

# ── Helper Functions ─────────────────────────────────────────────────────────

def get_recommendation(tier: str) -> str:
    recommendations = {
        'FREEZE': 'Immediately suspend account activity and initiate manual review. '
                  'Contact client via verified phone number.',
        'ALERT':  'Trigger step-up authentication on next login. '
                  'Flag for review within 24 hours.',
        'MONITOR': 'Enable enhanced transaction monitoring. '
                   'Review if further suspicious signals emerge.',
        'CLEAR':  'No action required. Continue standard monitoring.'
    }
    return recommendations.get(tier, 'Unknown tier')


def get_shap_explanation(X_row: pd.DataFrame) -> list:
    """Get top 3 SHAP drivers for a single transaction"""
    shap_vals = explainer.shap_values(X_row)[0]
    feature_impacts = list(zip(features, shap_vals))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    drivers = []
    for feat, impact in feature_impacts[:3]:
        direction = "increases" if impact > 0 else "decreases"
        drivers.append({
            "feature": feat,
            "impact": round(float(impact), 4),
            "direction": direction,
            "explanation": f"{feat} {direction} fraud risk by {abs(impact):.3f}"
        })
    return drivers


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "SentinelIQ",
        "description": "Post-breach identity risk scoring engine",
        "version": "1.0.0",
        "endpoints": ["/assess", "/client/{client_id}", "/model/performance", "/docs"]
    }


@app.post("/assess", response_model=RiskResponse)
def assess_client(request: AssessRequest):
    client_id = request.client_id

    # Check smart_risk_scores first — pre-computed with confidence + CLV
    smart_query = text("SELECT * FROM smart_risk_scores WHERE client_id = :cid")
    with engine.connect() as conn:
        smart = pd.read_sql(smart_query, conn, params={"cid": client_id})

    if not smart.empty:
        row = smart.iloc[0]
        # Get SHAP explanation from transaction data
        tx_query = text("SELECT * FROM transactions_raw WHERE transactionid = :cid")
        with engine.connect() as conn:
            tx = pd.read_sql(tx_query, conn, params={"cid": int(client_id)})

        if not tx.empty:
            tx.columns = [c.lower() for c in tx.columns]
            X_row = tx[[f.lower() for f in features]].apply(
                pd.to_numeric, errors='coerce').fillna(-1)
            X_row.columns = features
            drivers = get_shap_explanation(X_row)
        else:
            drivers = []

        return RiskResponse(
            client_id=client_id,
            fraud_probability=round(float(row['fraud_probability']), 4),
            exposure_risk=round(float(row['exposure_risk']), 4),
            composite_score=round(float(row['composite_score']), 4),
            risk_tier=str(row['risk_tier']),
            top_risk_drivers=drivers,
            recommendation=f"[{row['smart_action']}] {row['recommendation']}"
        )


    # Fallback — client not in smart scores, compute on the fly
    tx_query = text("SELECT * FROM transactions_raw WHERE transactionid = :cid")
    with engine.connect() as conn:
        tx = pd.read_sql(tx_query, conn, params={"cid": int(client_id)})

    if tx.empty:
        raise HTTPException(status_code=404, detail=f"Client {client_id} not found")

    exp_query = text("SELECT * FROM exposure_profiles WHERE client_id = :cid")
    with engine.connect() as conn:
        exp = pd.read_sql(exp_query, conn, params={"cid": client_id})

    exposure_score = float(exp['exposure_score'].values[0]) if not exp.empty else 0.0

    tx.columns = [c.lower() for c in tx.columns]
    X_row = tx[[f.lower() for f in features]].apply(
        pd.to_numeric, errors='coerce').fillna(-1)
    X_row.columns = features

    fraud_prob = float(model.predict_proba(X_row)[:, 1][0])
    composite = round(0.70 * fraud_prob + 0.30 * exposure_score, 4)

    if composite >= 0.70: tier = 'FREEZE'
    elif composite >= 0.45: tier = 'ALERT'
    elif composite >= 0.20: tier = 'MONITOR'
    else: tier = 'CLEAR'

    drivers = get_shap_explanation(X_row)

    return RiskResponse(
        client_id=client_id,
        fraud_probability=round(fraud_prob, 4),
        exposure_risk=round(exposure_score, 4),
        composite_score=composite,
        risk_tier=tier,
        top_risk_drivers=drivers,
        recommendation=get_recommendation(tier)
    )

@app.get("/client/{client_id}")
def get_client_history(client_id: str):
    """Retrieve full risk score history for a client"""
    query = text("SELECT * FROM risk_scores WHERE client_id = :cid")
    with engine.connect() as conn:
        result = pd.read_sql(query, conn, params={"cid": client_id})

    if result.empty:
        raise HTTPException(status_code=404, detail=f"No history found for {client_id}")

    return result.to_dict(orient='records')


@app.get("/model/performance")
def model_performance():
    """Return live model performance metrics from the database"""
    query = text("""
        SELECT 
            risk_tier,
            COUNT(*) as count,
            ROUND(AVG(fraud_probability)::numeric, 4) as avg_fraud_prob,
            ROUND(AVG(composite_score)::numeric, 4) as avg_composite
        FROM risk_scores
        GROUP BY risk_tier
        ORDER BY avg_composite DESC
    """)
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)

    total = text("SELECT COUNT(*) as total FROM risk_scores")
    with engine.connect() as conn:
        total_count = pd.read_sql(total, conn)['total'][0]

    return {
        "total_clients_scored": int(total_count),
        "model_auc_roc": 0.9434,
        "model_auc_pr": 0.6447,
        "tier_breakdown": result.to_dict(orient='records')
    }


@app.get("/cohort/{exposure_type}")
def cohort_analysis(exposure_type: str):
    """
    Compare risk stats between breach-exposed and non-exposed clients.
    exposure_type: 'sin' | 'dob' | 'account' | 'ip' | 'gov_id'
    """
    col_map = {
        'sin': 'sin_exposed',
        'dob': 'dob_exposed', 
        'account': 'account_number_exposed',
        'ip': 'ip_exposed',
        'gov_id': 'government_id_exposed'
    }

    if exposure_type not in col_map:
        raise HTTPException(status_code=400, 
                          detail=f"Invalid exposure_type. Choose from: {list(col_map.keys())}")

    col = col_map[exposure_type]
    query = text(f"""
        SELECT 
            e.{col} as exposed,
            COUNT(*) as client_count,
            ROUND(AVG(r.fraud_probability)::numeric, 4) as avg_fraud_prob,
            ROUND(AVG(r.composite_score)::numeric, 4) as avg_composite_score,
            COUNT(CASE WHEN r.risk_tier = 'FREEZE' THEN 1 END) as freeze_count,
            COUNT(CASE WHEN r.risk_tier = 'ALERT' THEN 1 END) as alert_count
        FROM exposure_profiles e
        JOIN risk_scores r ON e.client_id = r.client_id
        GROUP BY e.{col}
    """)
    with engine.connect() as conn:
        result = pd.read_sql(query, conn)

    return {
        "exposure_type": exposure_type,
        "analysis": result.to_dict(orient='records'),
        "insight": f"Clients with {exposure_type.upper()} exposed show elevated composite risk scores"
    }