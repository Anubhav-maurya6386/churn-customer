import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from churn.monitor import build_reference, drift_report
from churn.predict import ChurnModel, InferenceError


def test_model_beats_chance_and_is_calibrated(trained):
    X, y, p = trained["test"]
    assert trained["meta"]["metrics"]["roc_auc"] > 0.75
    assert abs(p.mean() - y.mean()) < 0.05          # calibration sanity check


def test_score_output_schema(trained, df):
    m = ChurnModel.load(trained["path"])
    res = m.score(df.head(20))
    assert {"churn_probability", "risk_tier", "contact", "top_risk_drivers", "recommended_action"} <= set(res.columns)
    assert res["churn_probability"].between(0, 1).all() and set(res.risk_tier) <= {"LOW", "MEDIUM", "HIGH"}


def test_missing_field_rejected(trained, df):
    with pytest.raises(InferenceError, match="missing"):
        ChurnModel.load(trained["path"]).score(df.head(3).drop(columns=["contract"]))


def test_unseen_category_does_not_crash(trained, df):
    row = df.head(2).copy(); row["payment_method"] = "Crypto"
    assert len(ChurnModel.load(trained["path"]).score(row)) == 2


def test_risky_customer_scores_higher_than_loyal(trained):
    m = ChurnModel.load(trained["path"])
    base = dict(gender="Male", senior_citizen=0, partner="Yes", dependents="Yes", phone_service="Yes",
                online_security="Yes", tech_support="Yes", paperless_billing="No", total_charges=1500.0, late_payments=0)
    risky = dict(base, tenure_months=2, internet_service="Fiber optic", contract="Month-to-month", payment_method="Electronic check",
                 monthly_charges=95.0, support_calls=5, partner="No", dependents="No", tech_support="No", total_charges=190.0)
    loyal = dict(base, tenure_months=60, internet_service="DSL", contract="Two year", payment_method="Credit card",
                 monthly_charges=50.0, support_calls=0, total_charges=3000.0)
    p = m.predict_proba(pd.DataFrame([risky, loyal]))
    assert p[0] > 0.5 > p[1]


def test_api_end_to_end(trained, df, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_PATH", str(trained["path"]))
    from churn.api import app
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert "required_fields" in c.get("/model-info").json()
        recs = df.drop(columns=["churn"]).head(5).replace({np.nan: None}).to_dict("records")
        r = c.post("/predict", json={"customers": recs})
        assert r.status_code == 200 and r.json()["n"] == 5
        bad = c.post("/predict", json={"customers": [{"tenure_months": 3}]})
        assert bad.status_code == 422
        assert c.post("/predict", json={"customers": []}).status_code == 422


def test_drift_detection(df):
    X = df.drop(columns=["customer_id", "churn"])
    ref = build_reference(X)
    same = drift_report(ref, X.sample(frac=0.5, random_state=0))
    assert (same.status == "ok").all()
    shifted = X.copy(); shifted["monthly_charges"] += 40
    rep = drift_report(ref, shifted)
    assert rep.set_index("feature").loc["monthly_charges", "status"] == "ALERT"
