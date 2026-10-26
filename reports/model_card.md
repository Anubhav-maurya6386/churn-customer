# Model Card - Customer Churn Predictor v2.0.0

*Auto-generated at training time (2026-10-26T09:05:25+00:00).*

## Purpose
Rank telecom customers by probability of churning in the next period and decide **who is economically worth
contacting** with a retention offer. Intended users: retention / CRM team. Not for pricing, credit or any decision that
denies a customer service.

## Model
- Family selected by cross-validation: **Logistic Regression**
- CV AUC by family: Logistic Regression 0.8666, Random Forest 0.857, Gradient Boosting 0.8601
- Wrapped in probability calibration (isotonic) so a score of 0.30 means ~30% of such customers churn.
- Feature engineering is part of the saved pipeline (no train/serve skew).

## Test-set performance (data never used for tuning)
| Metric | Calibrated (final) | Uncalibrated |
|---|---|---|
| ROC-AUC | 0.8768 | 0.8766 |
| PR-AUC | 0.7363 | 0.7395 |
| Brier score (lower = better) | 0.1195 | 0.1196 |
| Lift in top decile | 3.11x | 3.11x |
| Churners found in riskiest 20% | 55% | 55% |

Base churn rate: 26.5%.

## Business impact (held-out customers, using real outcomes)
Assumptions: margin 35%, 24-month horizon, offer cost 40.0,
offer saves 30% of would-be churners.

| Strategy | Profit |
|---|---|
| Contact nobody | 0 |
| Contact everybody | 12,697 |
| **Model rule (contact if expected profit > 0)** | **41,354** (37% of customers) |
| Perfect knowledge (upper bound) | 59,737 |

## Top drivers (permutation importance, drop in AUC)
- `contract`: 0.146
- `internet_service`: 0.044
- `total_charges`: 0.021
- `tenure_months`: 0.014
- `monthly_charges`: 0.011
- `support_calls`: 0.009

## Fairness check
- **gender**: contact rate Female=39%, Male=35%; AUC Female=0.89, Male=0.87
- **senior_citizen**: contact rate 0=37%, 1=39%; AUC 0=0.87, 1=0.90
Contact rates and AUC are compared across groups. Differences can reflect genuine behaviour differences;
review them before deployment. `gender` is not predictive by design (synthetic data); check this on real data.

## Limitations
- Trained on **synthetic** data in this demo; re-run on your own customer data before trusting any number.
- Correlation, not causation: drivers show who is at risk, not what will change their mind. Validate offers with an A/B test.
- Retrain at least monthly or when drift monitoring shows ALERT (`python -m churn.monitor`).
