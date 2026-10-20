import pytest
from churn.data import DataValidationError, validate, adapt_kaggle_telco
import pandas as pd


def test_valid_data_passes(df, cfg):
    assert len(validate(df, cfg)) == len(df)


def test_duplicate_ids_rejected(df, cfg):
    bad = pd.concat([df, df.head(5)])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate(bad, cfg)


def test_bad_target_rejected(df, cfg):
    bad = df.copy(); bad.loc[0, "churn"] = 7
    with pytest.raises(DataValidationError, match="target"):
        validate(bad, cfg)


def test_negative_tenure_rejected(df, cfg):
    bad = df.copy(); bad.loc[0, "tenure_months"] = -3
    with pytest.raises(DataValidationError, match="tenure"):
        validate(bad, cfg)


def test_kaggle_adapter():
    raw = pd.DataFrame({"customerID": ["a", "b"], "tenure": [1, 2], "MonthlyCharges": [20.0, 30.0],
                        "TotalCharges": ["20.0", " "], "Churn": ["Yes", "No"], "PaymentMethod": ["x", "y"]})
    out = adapt_kaggle_telco(raw)
    assert {"customer_id", "tenure_months", "monthly_charges", "total_charges", "churn", "payment_method"} <= set(out.columns)
    assert out["churn"].tolist() == [1, 0] and out["total_charges"].isna().sum() == 1
