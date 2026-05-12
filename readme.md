# SentinelIQ

**Most fraud systems ask one question: is this fraud?**

SentinelIQ asks three: *Is this fraud? How confident are we? And what does it cost to be wrong?*

That third question is what most models miss — and it's the one that matters most in production. A high-value customer with four years of clean history getting their account frozen because of a borderline model score isn't a win. It's a churn event, a support ticket, and a reputational hit. SentinelIQ is built around the idea that fraud decisions should be proportional to both risk *and* certainty.

**Live demo:** [sentineliq-godbless.streamlit.app](https://sentineliq-godbless.streamlit.app)  
**API docs:** Run the FastAPI locally at `http://127.0.0.1:8000/docs`

---

## What it does

SentinelIQ scores every client across three independent signals and combines them into a single risk decision:

**1. ML Fraud Score**  
An XGBoost model trained on 590,540 real financial transactions from the IEEE-CIS Fraud Detection dataset. 54 features including transaction timing, card network signals, count-based behavioural features (how many cards are linked to this address?), and time-delta features (how long since this account was last active?). AUC-ROC of 0.9528, AUC-PR of 0.7017 — which crosses the threshold most fraud papers treat as the target for heavily imbalanced datasets.

**2. Breach Exposure Score**  
Motivated by the August 2025 Wealthsimple breach that exposed SINs, government IDs, and account numbers for tens of thousands of clients. For each client, a weighted exposure score is computed based on which data fields were compromised — SIN exposure carries the most weight (0.35) because combined with a DOB it enables synthetic identity fraud; IP exposure alone carries the least (0.08). This layer catches risk that transaction behaviour alone can't see.

**3. Calibration + Confidence Intervals**  
The raw XGBoost probabilities are poorly calibrated due to class imbalance handling — when the model says 0.85, the actual fraud rate at that score is closer to 0.36. Platt scaling fixes this. Then bootstrap confidence intervals are computed per prediction: a wide interval (HIGH uncertainty) triggers a softer response than a narrow interval (HIGH confidence), even at the same raw score. This is what prevents borderline cases from being treated with the same urgency as clear-cut fraud.

The three signals combine into **7 smart actions** instead of a blunt freeze/clear binary:

| Action | Trigger |
|--------|---------|
| FREEZE + PRIORITY REVIEW | High confidence fraud on a PREMIUM client — senior analyst within 1 hour |
| AUTO FREEZE | High confidence fraud on a standard account — automatic |
| SOFT FREEZE | Elevated risk but uncertain signal — restrict new transactions, review within 24hrs |
| STEP-UP AUTH | Suspicious — require SMS/email verification on next transaction |
| ENHANCED MONITORING | Alert-level risk on a PREMIUM client — don't disrupt, watch closely |
| PASSIVE MONITORING | Low-moderate risk — log and observe |
| CLEAR | No action needed |

---

## Why the model got better over time

The initial model scored AUC-PR 0.6447 on 43 features. Four iterations later it reached 0.7017 on 54 features and 1,500 trees. Here's what actually moved the needle:

- **Adding V columns**: The IEEE-CIS dataset has 339 anonymized V columns. Most are missing 50%+ of the time or have near-zero correlation with fraud. Picking the 11 with the highest correlation and lowest missingness (V123, V302, V304, V303, V283, V125, V282, V111, V113, V69, V70) pushed AUC-PR from 0.6447 to 0.6807.
- **More trees at lower learning rate**: Going from 500 trees at lr=0.05 to 1,500 trees at lr=0.03 pushed AUC-PR from 0.6807 to 0.7017. The model was still improving at tree 1,499.
- **Calibration**: Platt scaling didn't improve AUC but made the probability outputs trustworthy — which is what enables the confidence interval layer to work correctly.

---

## What the data actually shows

A few findings from the EDA that are worth noting:

**Fraud peaks at 7am, not 3am.** Contrary to the intuitive assumption, fraud in this dataset is lowest at midnight (3.1% rate) and peaks at 7am (10.6%). Fraudsters blend into the morning rush when transactions look normal and victims are less likely to notice.

**Product type C has 3x the fraud rate.** ProductCD=C (likely crypto/cash-adjacent transactions) has an 11.7% fraud rate versus 2.0% for ProductCD=W (standard e-commerce). A single categorical feature carries enormous signal.

**Behavioural counts matter more than transaction-level signals.** The top SHAP features are C8, C4, and C14 — count-based features measuring how many cards are linked to an address, how many addresses to a card. These network-level signals dominate over transaction amount or time of day. The fraud in this dataset is primarily organized ring fraud, not individual account compromises.

**The model is more useful than accuracy implies.** On a 3.5% fraud rate dataset, a model predicting "not fraud" for every transaction would be 96.5% accurate. SentinelIQ's AUC-PR of 0.7017 means it's genuinely finding fraud cases while keeping false positives manageable — the metric that actually matters for an operations team.

---

## Architecture

```
590,540 transactions (IEEE-CIS dataset)
            │
            ▼
    Feature Engineering
    54 features: transaction signals,
    card network, count features (C1-C14),
    time deltas (D1-D15), V columns
            │
            ▼
    XGBoost Classifier
    1,500 trees · lr=0.03 · scale_pos_weight=27.6
    AUC-ROC: 0.9528 · AUC-PR: 0.7017
            │
            ▼
    Platt Scaling Calibration
    Logistic regression on raw scores →
    trustworthy probabilities
            │
            ├──────────────────────┐
            ▼                      ▼
    Breach Exposure Layer    Confidence Intervals
    Weighted by field type   Bootstrap (n=200)
    SIN=0.35, GovID=0.25     90% CI per prediction
    Account=0.20, DOB=0.12
    IP=0.08
            │                      │
            └──────────┬───────────┘
                       ▼
            CLV Scoring + Smart Decision Engine
            7 actions based on risk × confidence × value
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
        FastAPI                PostgreSQL
        4 endpoints            (Neon cloud)
        SHAP explainability    6 tables
            │
            ▼
        Streamlit Dashboard
        Live · Interactive · Plotly charts
```

---

## Running it locally

```bash
git clone https://github.com/Vatsal-cs/sentineliq.git
cd sentineliq

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your database credentials
cp .env.example .env
# Edit .env with a Neon PostgreSQL connection string

# Terminal 1 — API
cd api && uvicorn main:app --reload

# Terminal 2 — Dashboard
cd dashboard && python -m streamlit run app.py
```

The notebooks in `/notebooks` document the full analysis in order:
- `01_exploration.ipynb` — EDA, feature engineering, model training
- `02_confidence_clv.ipynb` — calibration, confidence intervals, CLV scoring, smart decisions
- `03_drift_monitor.ipynb` — model drift detection across time windows

---

## Project structure

```
sentineliq/
├── api/
│   └── main.py                   # FastAPI — /assess, /client, /model/performance, /cohort
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── notebooks/
│   ├── 01_exploration.ipynb      # EDA + model training
│   ├── 02_confidence_clv.ipynb   # Calibration + smart decisions
│   └── 03_drift_monitor.ipynb    # Drift detection
├── models/
│   ├── sentineliq_model.pkl      # Trained XGBoost (v4)
│   ├── features.pkl              # 54 feature names
│   └── platt_scaler.pkl          # Calibration model
├── data/
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── calibration_comparison.png
│   └── drift_monitor.png
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech stack

| Layer | Tool |
|-------|------|
| Database | PostgreSQL via Neon |
| Data processing | Python, pandas, NumPy |
| ML model | XGBoost |
| Calibration | Platt scaling (scikit-learn) |
| Explainability | SHAP |
| Confidence intervals | Bootstrap sampling, SciPy |
| API | FastAPI + Swagger UI |
| Dashboard | Streamlit + Plotly |
| Deployment | Streamlit Community Cloud |
| Version control | Git / GitHub |

---

## Model performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9528 |
| AUC-PR | 0.7017 |
| Training transactions | 472,432 |
| Test transactions | 118,108 |
| Features | 54 |
| Trees | 1,500 |
| Fraud rate in dataset | 3.50% |
| Scale pos weight | 27.6 |
| Clients scored | 50,000 |
| AUTO FREEZE tier | 75 (0.15%) |
| PRIORITY REVIEW tier | 14 |
| CLEAR tier | 43,056 (86.1%) |

---

*Built by Vatsal Jain — https://www.linkedin.com/in/vatsal-jain-873920228/— vatsalj05@gmail.com*
