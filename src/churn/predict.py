"""Scoring, explanations and retention recommendations.

CLI:  python -m churn.predict --input data/raw/customers.csv --output reports/predictions.csv
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from .business import customer_value, expected_profit
from .config import resolve, get_logger

log = get_logger("churn.predict")
DEFAULT_MODEL = "artifacts/churn_model.joblib"


class InferenceError(ValueError):
    pass


def action_for(col, val):
    """Map a churn driver to a concrete retention play (None if not actionable)."""
    v = str(val)
    if col == "contract" and v == "Month-to-month": return "Offer a discounted 12-24 month contract"
    if col == "tech_support" and v == "No": return "Free 3-month tech-support trial"
    if col == "online_security" and v == "No": return "Bundle online security free for 3 months"
    if col == "payment_method" and v == "Electronic check": return "Incentivise auto-pay (bank / card)"
    if col in ("support_calls", "calls_per_year"): return "Proactive call from a senior support agent"
    if col == "late_payments": return "Offer flexible billing date / payment plan"
    if col in ("monthly_charges", "charge_vs_history"): return "Review plan fit; offer loyalty discount"
    if col in ("tenure_months", "is_new_customer", "tenure_bucket"): return "Run a 90-day onboarding / welcome journey"
    if col == "internet_service" and v == "Fiber optic": return "Service-quality check + speed guarantee"
    if col == "senior_citizen" and v == "1": return "Dedicated assisted-support line"
    return None


class ChurnModel:
    def __init__(self, model, meta):
        self.model, self.meta = model, meta

    @classmethod
    def load(cls, path=None):
        path = resolve(path or os.environ.get("CHURN_MODEL_PATH", DEFAULT_MODEL))
        if not path.exists():
            raise FileNotFoundError(f"No model at {path}. Run `python -m churn.train` first.")
        art = joblib.load(path)
        return cls(art["model"], art["meta"])

    # ---- validation -------------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        need = self.meta["raw_features"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise InferenceError(f"missing required fields: {missing}")
        X = df[need].copy()
        for c in self.meta["numeric_features"]:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        if X["monthly_charges"].isna().any():
            raise InferenceError("monthly_charges must be numeric for every customer")
        unseen = {c: sorted(set(X[c].astype(str)) - set(lv)) for c, lv in self.meta["category_levels"].items()}
        unseen = {c: v for c, v in unseen.items() if v}
        if unseen:
            log.warning("unseen categories (will be treated as unknown): %s", unseen)
        return X

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.prepare(df))[:, 1]

    # ---- explanations -----------------------------------------------------
    def drivers(self, X: pd.DataFrame, p: np.ndarray, k: int = 3):
        """Local explanation by occlusion: how much does risk drop if this field takes the typical value?
        Model-agnostic and fast; positive = pushes this customer toward churning."""
        deltas = {}
        for c in self.meta["raw_features"]:
            Xb = X.copy()
            Xb[c] = self.meta["baseline"][c]
            deltas[c] = p - self.model.predict_proba(Xb)[:, 1]
        D = pd.DataFrame(deltas, index=X.index)
        out_txt, out_act = [], []
        for i in range(len(X)):
            top = D.iloc[i].sort_values(ascending=False)
            top = top[top > 0.01].head(k)
            out_txt.append("; ".join(f"{c}={X.iloc[i][c]} (+{d * 100:.0f} pts)" for c, d in top.items()) or "no strong risk drivers")
            acts = []
            for c in top.index:
                a = action_for(c, X.iloc[i][c])
                if a and a not in acts: acts.append(a)
            out_act.append("; ".join(acts[:2]) or "Standard engagement")
        return out_txt, out_act

    # ---- main entry -------------------------------------------------------
    def score(self, df: pd.DataFrame, explain: bool = True) -> pd.DataFrame:
        X = self.prepare(df)
        p = self.model.predict_proba(X)[:, 1]
        biz, tiers = self.meta["business"], self.meta["risk_tiers"]
        ep = expected_profit(p, X["monthly_charges"], biz)
        out = pd.DataFrame(index=df.index)
        if "customer_id" in df: out["customer_id"] = df["customer_id"]
        out["churn_probability"] = p.round(4)
        out["risk_tier"] = np.where(p >= tiers["high"], "HIGH", np.where(p >= tiers["medium"], "MEDIUM", "LOW"))
        out["value_at_risk"] = (p * customer_value(X["monthly_charges"], biz)).round(0)
        out["expected_profit_if_contacted"] = ep.round(0)
        out["contact"] = ep > 0
        if explain:
            txt, act = self.drivers(X, p)
            out["top_risk_drivers"], out["recommended_action"] = txt, np.where(out["contact"], act, "No action (not cost-effective)")
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/customers.csv"); ap.add_argument("--output", default="reports/predictions.csv")
    a = ap.parse_args()
    res = ChurnModel.load().score(pd.read_csv(resolve(a.input)))
    res = res.sort_values("expected_profit_if_contacted", ascending=False)
    resolve(a.output).parent.mkdir(parents=True, exist_ok=True); res.to_csv(resolve(a.output), index=False)
    print(res["risk_tier"].value_counts().to_string())
    print(f"\nContact list: {int(res.contact.sum())} of {len(res)} customers | value at risk (contact list): {res.loc[res.contact, 'value_at_risk'].sum():,.0f}")
    print(f"Saved {a.output}")
