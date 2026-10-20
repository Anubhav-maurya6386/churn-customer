"""Data & prediction drift monitoring using the Population Stability Index (PSI).
PSI < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 major shift -> investigate / retrain.

CLI:  python -m churn.monitor --input data/raw/new_batch.csv
"""
import argparse
import numpy as np
import pandas as pd
from .config import load_config, resolve, get_logger

EPS = 1e-4
log = get_logger("churn.monitor")


def build_reference(X: pd.DataFrame, scores=None, bins: int = 10) -> dict:
    ref = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            edges = np.unique(np.nanquantile(X[c].dropna(), np.linspace(0, 1, bins + 1)))
            if len(edges) < 3:
                continue
            edges[0], edges[-1] = -np.inf, np.inf
            ref[c] = {"type": "num", "edges": edges.tolist(), "props": _num_props(X[c], edges).tolist()}
        else:
            ref[c] = {"type": "cat", "props": X[c].astype(str).value_counts(normalize=True).to_dict()}
    if scores is not None:
        edges = np.unique(np.quantile(scores, np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
        ref["__score__"] = {"type": "num", "edges": edges.tolist(), "props": _num_props(pd.Series(scores), edges).tolist()}
    return ref


def _num_props(s, edges):
    counts, _ = np.histogram(pd.Series(s).dropna(), bins=np.asarray(edges))
    return counts / max(counts.sum(), 1)


def psi_value(expected, actual) -> float:
    e, a = np.clip(np.asarray(expected, float), EPS, None), np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(ref: dict, new: pd.DataFrame, scores=None, warn=0.10, alert=0.25) -> pd.DataFrame:
    rows = []
    for c, r in ref.items():
        if c == "__score__":
            if scores is None: continue
            cur = _num_props(pd.Series(scores), r["edges"])
        elif c not in new.columns:
            continue
        elif r["type"] == "num":
            cur = _num_props(pd.to_numeric(new[c], errors="coerce"), r["edges"])
        else:
            cats = list(dict.fromkeys(list(r["props"]) + new[c].astype(str).unique().tolist()))
            vc = new[c].astype(str).value_counts(normalize=True)
            cur = np.array([vc.get(k, 0) for k in cats]); r = {"props": [r["props"].get(k, 0) for k in cats]}
        v = psi_value(r["props"], cur)
        rows.append({"feature": "predicted_score" if c == "__score__" else c, "psi": round(v, 4),
                     "status": "ALERT" if v >= alert else "WARN" if v >= warn else "ok"})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from .predict import ChurnModel
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out", default="reports/drift_report.csv")
    a = ap.parse_args(); cfg = load_config()
    m = ChurnModel.load(); new = pd.read_csv(resolve(a.input))
    rep = drift_report(m.meta["reference"], new, m.predict_proba(new), cfg["drift"]["psi_warn"], cfg["drift"]["psi_alert"])
    rep.to_csv(resolve(a.out), index=False); print(rep.to_string(index=False))
    n_alert = (rep.status == "ALERT").sum()
    print(f"\n{n_alert} feature(s) in ALERT -> " + ("consider retraining." if n_alert else "model inputs look stable."))
