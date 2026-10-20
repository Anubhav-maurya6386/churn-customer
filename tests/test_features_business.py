import numpy as np
from churn.business import customer_value, expected_profit, profit_analysis
from churn.features import add_features

BIZ = {"horizon_months": 24, "gross_margin": 0.35, "offer_cost": 40, "offer_success_rate": 0.30}


def test_features_added_and_input_untouched(df):
    before = df.copy()
    out = add_features(df.drop(columns=["customer_id", "churn"]))
    for c in ["avg_monthly_spend", "is_new_customer", "tenure_bucket", "n_services", "flexible_and_manual_pay"]:
        assert c in out.columns
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()
    assert df.equals(before)


def test_expected_profit_math():
    assert customer_value(100, BIZ) == 100 * 0.35 * 24
    assert np.isclose(expected_profit(0.5, 100, BIZ), 0.5 * 0.3 * 840 - 40)


def test_cheap_low_risk_customers_not_worth_contacting():
    assert expected_profit(0.1, 20, BIZ) < 0 < expected_profit(0.8, 100, BIZ)


def test_profit_analysis_ordering():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000); p = np.clip(y * 0.5 + rng.uniform(0, 0.5, 2000), 0, 1)
    r = profit_analysis(y, p, np.full(2000, 80.0), BIZ)
    assert r["model_rule"] > r["contact_everybody"] and r["perfect_knowledge"] >= r["model_rule"]
