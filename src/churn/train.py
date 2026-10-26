"""Train -> tune -> calibrate -> evaluate -> package.   Run:  python -m churn.train

Pipeline: validation -> stratified split -> random-search tuning of 3 model families (CV AUC)
-> probability calibration -> single touch of the held-out test set -> business-profit analysis
-> fairness audit -> drift reference -> versioned artifact + auto-generated model card.
"""
import argparse
import json
import platform
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import __version__
from .business import expected_profit, profit_analysis
from .config import get_logger, load_config, resolve
from .data import load_data
from .evaluate import fairness_table, lift_table, metrics, save_plots
from .features import add_features
from .monitor import build_reference

log = get_logger("churn.train")


def build_pipeline(clf) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude=np.number)),
    ])
    return Pipeline([("features", FunctionTransformer(add_features)), ("pre", pre), ("clf", clf)])


def candidates(seed):
    return {
        "Logistic Regression": (LogisticRegression(max_iter=3000),
                                {"clf__C": loguniform(1e-2, 1e2)}),
        "Random Forest": (RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1),
                          {"clf__max_depth": [6, 10, 16, None], "clf__min_samples_leaf": [3, 5, 10, 20],
                           "clf__max_features": ["sqrt", 0.5]}),
        "Gradient Boosting": (HistGradientBoostingClassifier(early_stopping=True, random_state=seed),
                              {"clf__learning_rate": loguniform(0.02, 0.2), "clf__max_leaf_nodes": randint(8, 32),
                               "clf__min_samples_leaf": randint(20, 80), "clf__l2_regularization": loguniform(1e-3, 10),
                               "clf__max_iter": randint(150, 400)}),
    }


def run(cfg=None, df=None, artifacts_dir="artifacts", reports_dir="reports", fast=False, make_reports=True, light=False):
    """light=True: tiny, low-memory training (Logistic Regression only) used to rebuild a compatible model on small servers."""
    cfg = cfg or load_config()
    seed, tgt, idc = cfg["project"]["seed"], cfg["data"]["target"], cfg["data"]["id_col"]
    df = load_data(cfg) if df is None else df
    X, y = df.drop(columns=[idc, tgt]), df[tgt]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=seed)
    log.info("train=%d test=%d churn rate=%.1f%%", len(X_tr), len(X_te), y.mean() * 100)

    folds = 2 if fast else 3 if light else cfg["model"]["cv_folds"]
    n_iter = 1 if fast else 3 if light else cfg["model"]["search_iter"]
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)

    # 1) tune every model family
    searches, test_raw = {}, {}
    cands = candidates(seed)
    if light:
        cands = {"Logistic Regression": cands["Logistic Regression"]}
    for name, (clf, space) in cands.items():
        s = RandomizedSearchCV(build_pipeline(clf), space, n_iter=n_iter, cv=cv, scoring="roc_auc",
                               random_state=seed, n_jobs=1 if (fast or light) else -1, refit=True)
        s.fit(X_tr, y_tr); searches[name] = s
        test_raw[name] = s.predict_proba(X_te)[:, 1]
        log.info("%-20s CV AUC=%.4f  test AUC=%.4f", name, s.best_score_, metrics(y_te, test_raw[name])["roc_auc"])
    best = max(searches, key=lambda k: searches[k].best_score_)

    # 2) calibrate the winner so probabilities mean what they say (needed for money decisions)
    final = CalibratedClassifierCV(clone(searches[best].best_estimator_), method=cfg["model"]["calibration_method"], cv=folds)
    final.fit(X_tr, y_tr)
    p = final.predict_proba(X_te)[:, 1]
    m = metrics(y_te, p)
    m_raw = metrics(y_te, test_raw[best])
    log.info("FINAL %s | AUC=%.3f PR-AUC=%.3f Brier %.4f (uncalibrated %.4f) lift@10%%=%.2f",
             best, m["roc_auc"], m["pr_auc"], m["brier"], m_raw["brier"], m["lift_top_decile"])

    # 3) business value + fairness
    biz = cfg["business"]
    profit = profit_analysis(y_te, p, X_te["monthly_charges"], biz)
    exp_profit = expected_profit(p, X_te["monthly_charges"], biz)
    contact = exp_profit > 0
    value = X_te["monthly_charges"].to_numpy() * biz["gross_margin"] * biz["horizon_months"]
    realized = y_te.to_numpy() * biz["offer_success_rate"] * value - biz["offer_cost"]
    curve = np.cumsum(realized[np.argsort(-exp_profit)])
    fair = fairness_table(X_te, y_te, p, contact, cfg["fairness"]["groups"])
    lift = lift_table(y_te, p)
    log.info("PROFIT on test set | model rule: %.0f | contact everyone: %.0f | contact nobody: 0 | perfect knowledge: %.0f",
             profit["model_rule"], profit["contact_everybody"], profit["perfect_knowledge"])

    # 4) global importance + drift reference + baseline for local explanations
    if light:
        importance = pd.Series(dtype=float)
    else:
        imp = permutation_importance(final, X_te, y_te, scoring="roc_auc", n_repeats=3 if fast else 5, random_state=seed, n_jobs=1 if fast else -1)
        importance = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
    num_cols = X.select_dtypes("number").columns.tolist()
    baseline = {c: (X_tr[c].median() if c in num_cols else X_tr[c].mode().iloc[0]) for c in X.columns}
    baseline = {k: (v.item() if hasattr(v, "item") else v) for k, v in baseline.items()}

    meta = {
        "version": __version__, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_family": best, "raw_features": X.columns.tolist(), "numeric_features": num_cols,
        "baseline": baseline, "reference": build_reference(X_tr, p),
        "business": biz, "risk_tiers": cfg["risk_tiers"], "metrics": m,
        "cv_auc": {k: round(float(v.best_score_), 4) for k, v in searches.items()},
        "calibration": cfg["model"]["calibration_method"],
        "best_params": {k: {a.replace("clf__", ""): (float(b) if isinstance(b, (np.floating, float)) else b if not hasattr(b, "item") else b.item())
                            for a, b in v.best_params_.items()} for k, v in searches.items()},
        "sklearn": sklearn.__version__, "python": platform.python_version(),
        "category_levels": {c: sorted(X_tr[c].astype(str).unique().tolist()) for c in X.columns if c not in num_cols},
    }
    art_dir = resolve(artifacts_dir); art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "meta": meta}, art_dir / "churn_model.joblib")

    if make_reports:
        rep = resolve(reports_dir); rep.mkdir(parents=True, exist_ok=True)
        save_plots(rep, y_te, p, test_raw[best], {k: v for k, v in test_raw.items()}, lift, curve, importance, profit)
        lift.to_csv(rep / "lift_table.csv"); fair.to_csv(rep / "fairness.csv", index=False)
        importance.round(4).to_csv(rep / "feature_importance.csv")
        pd.DataFrame({"actual": y_te.to_numpy(), "probability": p.round(4), "monthly_charges": X_te["monthly_charges"].to_numpy()}
                     ).to_csv(rep / "test_predictions.csv", index=False)
        summary = {"model": best, "cv_auc": meta["cv_auc"], "test_metrics": m, "uncalibrated_metrics": m_raw, "profit": profit}
        (rep / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
        (rep / "model_card.md").write_text(model_card(meta, m, m_raw, profit, fair, importance, lift, biz))
        log.info("reports written to %s", rep)
    return {"meta": meta, "model": final, "test": (X_te, y_te, p), "profit": profit}


def model_card(meta, m, m_raw, profit, fair, importance, lift, biz) -> str:
    top = "\n".join(f"- `{k}`: {v:.3f}" for k, v in importance.head(6).items())
    gap = ""
    if len(fair):
        for g, d in fair.groupby("group"):
            gap += f"- **{g}**: contact rate " + ", ".join(f"{r.value}={r.contact_rate:.0%}" for r in d.itertuples()) + \
                   "; AUC " + ", ".join(f"{r.value}={r.auc:.2f}" for r in d.itertuples()) + "\n"
    return f"""# Model Card - Customer Churn Predictor v{meta['version']}

*Auto-generated at training time ({meta['trained_at']}).*

## Purpose
Rank telecom customers by probability of churning in the next period and decide **who is economically worth
contacting** with a retention offer. Intended users: retention / CRM team. Not for pricing, credit or any decision that
denies a customer service.

## Model
- Family selected by cross-validation: **{meta['model_family']}**\n- CV AUC by family: {', '.join(f'{k} {v}' for k, v in meta['cv_auc'].items())}
- Wrapped in probability calibration ({meta['calibration']}) so a score of 0.30 means ~30% of such customers churn.
- Feature engineering is part of the saved pipeline (no train/serve skew).

## Test-set performance (data never used for tuning)
| Metric | Calibrated (final) | Uncalibrated |
|---|---|---|
| ROC-AUC | {m['roc_auc']} | {m_raw['roc_auc']} |
| PR-AUC | {m['pr_auc']} | {m_raw['pr_auc']} |
| Brier score (lower = better) | {m['brier']} | {m_raw['brier']} |
| Lift in top decile | {m['lift_top_decile']}x | {m_raw['lift_top_decile']}x |
| Churners found in riskiest 20% | {m['recall_in_top_20pct']:.0%} | {m_raw['recall_in_top_20pct']:.0%} |

Base churn rate: {m['base_rate']:.1%}.

## Business impact (held-out customers, using real outcomes)
Assumptions: margin {biz['gross_margin']:.0%}, {biz['horizon_months']}-month horizon, offer cost {biz['offer_cost']},
offer saves {biz['offer_success_rate']:.0%} of would-be churners.

| Strategy | Profit |
|---|---|
| Contact nobody | 0 |
| Contact everybody | {profit['contact_everybody']:,.0f} |
| **Model rule (contact if expected profit > 0)** | **{profit['model_rule']:,.0f}** ({profit['share_contacted']:.0%} of customers) |
| Perfect knowledge (upper bound) | {profit['perfect_knowledge']:,.0f} |

## Top drivers (permutation importance, drop in AUC)
{top}

## Fairness check
{gap}Contact rates and AUC are compared across groups. Differences can reflect genuine behaviour differences;
review them before deployment. `gender` is not predictive by design (synthetic data); check this on real data.

## Limitations
- Trained on **synthetic** data in this demo; re-run on your own customer data before trusting any number.
- Correlation, not causation: drivers show who is at risk, not what will change their mind. Validate offers with an A/B test.
- Retrain at least monthly or when drift monitoring shows ALERT (`python -m churn.monitor`).
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true", help="quick smoke-test run")
    a = ap.parse_args(); run(fast=a.fast)