"""Data generation, loading, schema adaptation and validation.

CLI:
  python -m churn.data generate                    # training data
  python -m churn.data generate --drift --n 2000 --out data/raw/new_batch.csv
  python -m churn.data adapt-kaggle WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
import argparse
import re
import numpy as np
import pandas as pd
from .config import load_config, resolve, get_logger

log = get_logger("churn.data")


class DataValidationError(ValueError):
    pass


def generate_synthetic(n: int = 8000, seed: int = 42, drift: bool = False) -> pd.DataFrame:
    """Synthetic telecom customers with non-linear churn behaviour.
    drift=True simulates a changed customer base (price hike, fewer long contracts)."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n,
                          p=[0.72, 0.15, 0.13] if drift else [0.55, 0.21, 0.24])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.52, 0.30, 0.18] if drift else [0.44, 0.34, 0.22])
    tenure = np.where(contract == "Month-to-month", rng.integers(1, 40, n),
             np.where(contract == "One year", rng.integers(6, 60, n), rng.integers(12, 73, n)))
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n, p=[0.34, 0.23, 0.22, 0.21])
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n)
    dependents = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    ni = internet == "No"
    tech = np.where(ni, "No internet", rng.choice(["Yes", "No"], n, p=[0.3, 0.7]))
    sec = np.where(ni, "No internet", rng.choice(["Yes", "No"], n, p=[0.3, 0.7]))
    paperless = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
    base = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 50, 20))
    monthly = (base + rng.normal(0, 8, n) + (phone == "Yes") * 10 + (15 if drift else 0)).clip(18, 125).round(2)
    total = (monthly * tenure * rng.uniform(0.9, 1.05, n)).round(2)
    calls = rng.poisson(np.where(tech == "No", 1.8, 0.8))
    late = rng.poisson(np.where(payment == "Electronic check", 1.2, 0.4))
    mtm = contract == "Month-to-month"

    z = (-2.9
         + 1.2 * mtm - 1.0 * (contract == "Two year")
         + 0.7 * (internet == "Fiber optic") + 0.9 * ((internet == "Fiber optic") & mtm)   # interaction
         - 0.03 * tenure + 0.9 * (tenure <= 6)                                             # new-customer cliff
         + 0.015 * (monthly - 65) + 0.5 * (payment == "Electronic check")
         + 0.35 * (tech == "No") + 0.3 * (sec == "No")
         + 0.25 * senior + 0.2 * (paperless == "Yes")
         + 0.28 * calls + 0.22 * late + 0.25 * (calls >= 4)                                # frustration threshold
         - 0.3 * (partner == "Yes") - 0.25 * (dependents == "Yes")
         + rng.normal(0, 0.6, n))
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n), "senior_citizen": senior,
        "partner": partner, "dependents": dependents, "tenure_months": tenure,
        "phone_service": phone, "internet_service": internet, "online_security": sec,
        "tech_support": tech, "contract": contract, "paperless_billing": paperless,
        "payment_method": payment, "monthly_charges": monthly, "total_charges": total,
        "support_calls": calls, "late_payments": late, "churn": churn})
    df.loc[rng.choice(n, max(1, n // 200), replace=False), "total_charges"] = np.nan  # realistic missingness
    return df


def _snake(s: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s.strip())
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def adapt_kaggle_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the public Kaggle 'Telco Customer Churn' CSV to this project's schema."""
    out = df.rename(columns={c: _snake(c) for c in df.columns})
    out = out.rename(columns={"customerid": "customer_id", "customer_i_d": "customer_id", "tenure": "tenure_months"})
    out["total_charges"] = pd.to_numeric(out["total_charges"], errors="coerce")
    out["churn"] = out["churn"].map({"Yes": 1, "No": 0}).astype(int)
    return out


def validate(df: pd.DataFrame, cfg: dict, need_target: bool = True) -> pd.DataFrame:
    """Fail loudly on bad data (garbage in -> garbage out is the #1 production ML failure)."""
    d, problems = cfg["data"], []
    if len(df) < 200:
        problems.append(f"too few rows ({len(df)})")
    need = [d["id_col"]] + ([d["target"]] if need_target else [])
    problems += [f"missing column '{c}'" for c in need if c not in df.columns]
    if d["id_col"] in df and df[d["id_col"]].duplicated().any():
        problems.append("duplicate customer ids")
    if need_target and d["target"] in df and not set(df[d["target"]].unique()) <= {0, 1}:
        problems.append("target must be 0/1")
    if "tenure_months" in df and (df["tenure_months"] < 0).any():
        problems.append("negative tenure")
    if "monthly_charges" in df and (pd.to_numeric(df["monthly_charges"], errors="coerce") <= 0).any():
        problems.append("non-positive monthly_charges")
    if problems:
        raise DataValidationError("; ".join(problems))
    miss = df.isna().mean()
    for c in miss[miss > 0.3].index:
        log.warning("column %s is %.0f%% missing", c, miss[c] * 100)
    return df


def load_data(cfg: dict | None = None, path=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve(path or cfg["data"]["raw_path"])
    if not path.exists():
        log.info("No data at %s - generating synthetic data", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic(cfg["data"]["n_synthetic"], cfg["project"]["seed"]).to_csv(path, index=False)
    return validate(pd.read_csv(path), cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "adapt-kaggle"])
    ap.add_argument("src", nargs="?")
    ap.add_argument("--n", type=int)
    ap.add_argument("--out")
    ap.add_argument("--drift", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "generate":
        df = generate_synthetic(a.n or cfg["data"]["n_synthetic"],
                                cfg["project"]["seed"] + (99 if a.drift else 0), a.drift)
        out = resolve(a.out or cfg["data"]["raw_path"])
    else:
        df = adapt_kaggle_telco(pd.read_csv(a.src))
        out = resolve(a.out or cfg["data"]["raw_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("saved %d rows to %s (churn rate %s)", len(df), out,
             f"{df['churn'].mean():.1%}" if "churn" in df else "n/a")
