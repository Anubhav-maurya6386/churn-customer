"""Feature engineering. Lives INSIDE the model pipeline so training and serving can never diverge."""
import numpy as np
import pandas as pd

SERVICE_COLS = ["phone_service", "online_security", "online_backup", "device_protection",
                "tech_support", "streaming_tv", "streaming_movies", "multiple_lines"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    cols = set(X.columns)
    if {"total_charges", "tenure_months", "monthly_charges"} <= cols:
        tc = pd.to_numeric(X["total_charges"], errors="coerce")
        X["total_charges"] = tc
        X["avg_monthly_spend"] = tc / X["tenure_months"].clip(lower=1)
        # positive => customer now pays more than they historically did (price-shock signal)
        X["charge_vs_history"] = X["monthly_charges"] - X["avg_monthly_spend"]
    if "tenure_months" in cols:
        X["is_new_customer"] = (X["tenure_months"] <= 6).astype(int)
        X["tenure_bucket"] = pd.cut(X["tenure_months"], [-1, 6, 12, 24, 48, 1000],
                                    labels=["0-6", "7-12", "13-24", "25-48", "49+"]).astype(str)
    present = [c for c in SERVICE_COLS if c in cols]
    if present:
        X["n_services"] = sum((X[c] == "Yes").astype(int) for c in present)
    if {"contract", "payment_method"} <= cols:
        X["flexible_and_manual_pay"] = ((X["contract"] == "Month-to-month") &
                                        (X["payment_method"] == "Electronic check")).astype(int)
    if {"support_calls", "tenure_months"} <= cols:
        X["calls_per_year"] = X["support_calls"] / (X["tenure_months"].clip(lower=1) / 12).clip(lower=0.5)
    return X.replace([np.inf, -np.inf], np.nan)
