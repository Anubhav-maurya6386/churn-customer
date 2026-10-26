# Customer Churn Prediction System (v2, production-style)

Predicts which customers will leave, **explains why**, decides **who is worth contacting based on money**,
recommends a retention action, serves results through an **API and dashboard**, and **monitors itself for drift**.

## Why this is more than a notebook
| Real-world concern | How this project handles it |
|---|---|
| Garbage data | Schema validation (duplicates, bad target, negative values) fails loudly - `data.py` |
| Train/serve mismatch | Feature engineering is *inside* the saved sklearn pipeline - `features.py` |
| Overfitting / luck | Stratified split, cross-validated random search over 3 model families, test set touched once |
| Probabilities used for money | Probability **calibration** (isotonic) + Brier score + calibration plot |
| "High risk" is not "worth acting on" | **Profit rule**: contact only if `P(churn) x P(save) x value > offer cost` - `business.py` |
| Proving business value | Test-set profit vs. contact-nobody / contact-everybody / perfect-knowledge |
| Black box | Permutation importance (global) + per-customer **risk drivers** and driver-based actions |
| Bias | Fairness audit (AUC and contact rate by gender / senior citizen) |
| Model decay | **PSI drift monitor** for inputs and predicted scores - `monitor.py` |
| Deployment | FastAPI service, Streamlit dashboard, Dockerfile, Makefile, CI workflow |
| Regressions | 16 automated tests (data, features, business math, model sanity, API, drift) |
| Governance | Auto-generated `reports/model_card.md` with every training run |

## Results on the included demo data (held-out 1,600 customers)
- ROC-AUC **0.877**, PR-AUC 0.736, top-decile lift **3.1x**; riskiest 20% of customers contain 55% of churners.
- Profit: **41,354** using the model's rule vs **12,697** contacting everyone (nobody = 0; theoretical max 59,737).
- The rule contacts only ~37% of customers - the profit curve (`reports/profit_curve.png`) peaks right there.
- Logistic Regression won because the synthetic data is mostly additive. On messy real data, gradient boosting often wins; the pipeline selects automatically.
- These numbers come from **synthetic** data. Re-run on your own data before trusting any figure.

## Quick start
```bash
pip install -r requirements.txt
export PYTHONPATH=src          # Windows PowerShell:  $env:PYTHONPATH="src"
python -m churn.data generate  # demo data (skip if you have real data)
python -m churn.train          # tune, calibrate, evaluate, write artifacts/ + reports/
python -m churn.predict        # contact list -> reports/predictions.csv
python -m pytest -q            # run tests
uvicorn churn.api:app --port 8000      # API docs at http://localhost:8000/docs
streamlit run app/dashboard.py         # dashboard
```
`make all` does the first four steps. `docker build -t churn . && docker run -p 8000:8000 churn` runs the API.

## Use your real data
1. **Kaggle Telco Churn CSV:** `python -m churn.data adapt-kaggle WA_Fn-UseC_-Telco-Customer-Churn.csv`
2. **Your own CRM export:** save as `data/raw/customers.csv` with a `customer_id` column, a 0/1 `churn` column, plus
   `tenure_months`, `monthly_charges` (required by the business rule) and any other columns. Numeric/categorical types are auto-detected.
3. Edit `config.yaml -> business` with **your** margin, offer cost and expected save rate (this changes who gets contacted).
4. `python -m churn.train`, read `reports/model_card.md`.

## Project layout
```
config.yaml               all settings (business economics, tiers, drift thresholds)
src/churn/
  data.py                 synthetic generator, Kaggle adapter, validation
  features.py             feature engineering (price-shock, new-customer, services count, ...)
  train.py                tuning, calibration, evaluation, packaging, model card
  evaluate.py             metrics, lift, fairness, plots
  business.py             value and expected-profit logic
  predict.py              scoring, risk drivers, recommended actions
  monitor.py              PSI drift report
  api.py                  FastAPI service
app/dashboard.py          Streamlit command center (contact list, what-if, model quality, drift)
tests/                    pytest suite
artifacts/                trained model (churn_model.joblib)
reports/                  charts, metrics.json, fairness.csv, model_card.md, sample predictions
```

## API example
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"customers":[{
 "gender":"Male","senior_citizen":0,"partner":"No","dependents":"No","tenure_months":3,"phone_service":"Yes",
 "internet_service":"Fiber optic","online_security":"No","tech_support":"No","contract":"Month-to-month",
 "paperless_billing":"Yes","payment_method":"Electronic check","monthly_charges":95,"total_charges":285,
 "support_calls":4,"late_payments":2}]}'
```
Returns probability, risk tier, value at risk, expected profit, contact yes/no, top risk drivers and recommended action.
`GET /model-info` lists required fields and test metrics. Bad input returns HTTP 422 with a clear message.

## Operating it in production
1. **Weekly:** score all active customers, push the `contact=True` list to CRM / call centre.
2. **Weekly:** `python -m churn.monitor --input <this week's customers>`; any ALERT (PSI > 0.25) means investigate or retrain.
3. **Monthly:** retrain on new data with confirmed outcomes; compare the new model card with the old one before promoting.
4. **Prove it:** run an A/B test (offer to a random half of the contact list, nothing to the other half) to measure the real save rate, then update `offer_success_rate`.

## Known limitations (be honest about these in interviews / reports)
- Churn "label" here is a single yes/no; real systems often use time-windowed labels (churn in next 30 days) and time-based train/test splits.
- Drivers show correlation, not causation; only an A/B test shows what actually retains customers.
- The save rate, margin and offer cost are assumptions; profit results are only as good as these inputs.
- Fairness check covers two attributes and is a screening step, not a legal compliance review.




deployment link: https://churn-customer-r6xzhmctcbud7jubym8pv7.streamlit.app/
