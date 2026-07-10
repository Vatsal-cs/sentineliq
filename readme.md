# SentinelIQ

Fraud detection model + API + dashboard for scoring client risk. Most fraud systems just spit out "fraud / not fraud." This one also tracks how confident it actually is, and factors in what it costs to be wrong — freezing a long-time customer's account over a borderline score is its own kind of expensive (support tickets, churn, angry emails).

**Live demo:** [sentineliq-godbless.streamlit.app](https://sentineliq-godbless.streamlit.app)  
**API docs:** [sentineliq.onrender.com/docs](https://sentineliq.onrender.com/docs)

---

## What it does

Three signals get combined into one risk decision per client:

**1. ML fraud score**  
XGBoost trained on the IEEE-CIS Fraud Detection dataset (590,540 transactions). 54 features — transaction timing, card network stuff, count-based features like "how many cards are tied to this address," time-deltas like "how long since this account last did anything." Ended up at AUC-ROC 0.9528 / AUC-PR 0.7017, which is roughly in line with what other people get on this dataset.

**2. Breach exposure score**  
Added this after the Wealthsimple breach in Aug 2025 (SINs, gov IDs, account numbers leaked for a bunch of clients). Each client gets a weighted exposure score based on what fields of theirs were exposed — SIN is weighted highest (0.35) since it's the main ingredient for synthetic identity fraud, IP address is weighted lowest (0.08). Point of this layer is to catch risk that transaction history alone won't show you.

**3. Calibration + confidence intervals**  
Turns out the raw XGBoost probabilities aren't well calibrated because of how the class imbalance is handled — a 0.85 score actually corresponds to more like a 36% real fraud rate. Fixed that with Platt scaling. On top of that, bootstrap confidence intervals per prediction, so a wide interval (not sure) gets a softer response than a narrow interval (sure) even if the raw score is identical.

Those three signals feed into **7 possible actions** instead of just freeze/clear:

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

## Model iteration notes

Started at AUC-PR 0.6447 with 43 features, ended up at 0.7017 with 54 features and 1,500 trees. What actually helped:

- **V columns**: IEEE-CIS has 339 anonymized V columns, most either mostly-missing or basically uncorrelated with fraud. Went through and picked the 11 with best correlation/missingness tradeoff (V123, V302, V304, V303, V283, V125, V282, V111, V113, V69, V70). That alone took AUC-PR from 0.6447 to 0.6807.
- **More trees, lower learning rate**: 500 trees @ lr=0.05 → 1,500 trees @ lr=0.03. Got 0.6807 to 0.7017. Was still climbing slowly at tree 1,499, could probably squeeze out a bit more.
- **Calibration**: Platt scaling doesn't change AUC at all but makes the probabilities actually mean something, which the confidence interval step depends on.

---

## Notes from the EDA

**Fraud peaks around 7am, not overnight.** I expected 3am to be the worst. It's actually the lowest (3.1%), and 7am is the highest (10.6%) — guessing fraudsters ride the morning rush when everything looks like normal traffic.

**ProductCD=C is way riskier.** 11.7% fraud rate vs 2.0% for ProductCD=W. One categorical column doing a lot of work.

**Count features beat transaction-level ones.** Top SHAP features are C8, C4, C14 — stuff like "how many cards linked to this address." Outranks amount and time of day by a good margin. Looks like most of the fraud here is ring-based rather than one-off account takeovers.

**Accuracy is a bad metric here.** Fraud rate is 3.5%, so guessing "not fraud" every time gets you 96.5% accuracy for free. AUC-PR 0.7017 is the number that actually matters for whether this catches real cases without burying an ops team in false positives.

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
