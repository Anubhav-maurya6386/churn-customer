"""Evaluation: metrics, lift/gain, calibration, fairness, plots."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    top10 = order[: max(1, len(y) // 10)]
    top20 = order[: max(1, len(y) // 5)]
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        "lift_top_decile": round(float(y[top10].mean() / y.mean()), 2),
        "recall_in_top_20pct": round(float(y[top20].sum() / y.sum()), 3),
        "base_rate": round(float(y.mean()), 4),
    }


def lift_table(y, p, bins=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = (np.arange(len(d)) * bins // len(d)) + 1
    t = d.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"),
                                churn_rate=("y", "mean"), avg_predicted=("p", "mean"))
    t["lift"] = t["churn_rate"] / d["y"].mean()
    t["cum_capture"] = t["churners"].cumsum() / d["y"].sum()
    return t.round(3)


def fairness_table(df_groups: pd.DataFrame, y, p, contact, groups) -> pd.DataFrame:
    """Per-group performance + contact rates. Large gaps deserve a human review."""
    rows = []
    y, p, contact = np.asarray(y), np.asarray(p), np.asarray(contact)
    for g in groups:
        if g not in df_groups:
            continue
        for val in sorted(df_groups[g].unique(), key=str):
            m = (df_groups[g] == val).to_numpy()
            if m.sum() < 30 or len(set(y[m])) < 2:
                continue
            rows.append({"group": g, "value": val, "n": int(m.sum()),
                         "churn_rate": round(float(y[m].mean()), 3),
                         "auc": round(float(roc_auc_score(y[m], p[m])), 3),
                         "contact_rate": round(float(contact[m].mean()), 3),
                         "recall_of_contacted": round(float(contact[m & (y == 1)].mean()) if (m & (y == 1)).any() else np.nan, 3)})
    return pd.DataFrame(rows)


def save_plots(out, y, p_cal, p_raw, p_by_model, lift, realized_curve, importance, contact_stats):
    y = np.asarray(y)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in p_by_model.items():
        fpr, tpr, _ = roc_curve(y, p); ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set(title="ROC curves (test set)", xlabel="False positive rate", ylabel="True positive rate"); ax[0].legend(fontsize=8)
    pr, rc, _ = precision_recall_curve(y, p_cal)
    ax[1].plot(rc, pr, color="#c0392b"); ax[1].axhline(y.mean(), color="k", ls="--", lw=1)
    ax[1].set(title="Precision-Recall (final model)", xlabel="Recall", ylabel="Precision")
    fig.tight_layout(); fig.savefig(out / "roc_pr_curves.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for p, lab in [(p_raw, "Before calibration"), (p_cal, "After calibration")]:
        fp, mp = calibration_curve(y, p, n_bins=10, strategy="quantile"); ax.plot(mp, fp, "o-", label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(title="Calibration: predicted vs actual churn", xlabel="Predicted probability", ylabel="Observed churn rate"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].bar(lift.index, lift["churn_rate"], color="#3b6fb6"); ax[0].axhline(y.mean(), color="k", ls="--", lw=1)
    ax[0].set(title="Churn rate by risk decile (1 = riskiest)", xlabel="Decile", ylabel="Actual churn rate")
    ax[1].plot(np.r_[0, lift.index * 10], np.r_[0, lift["cum_capture"] * 100], "o-", color="#c0392b")
    ax[1].plot([0, 100], [0, 100], "k--", lw=1); ax[1].set(title="Cumulative gains", xlabel="% customers contacted (riskiest first)", ylabel="% churners captured")
    fig.tight_layout(); fig.savefig(out / "lift_gains.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(realized_curve)) / len(realized_curve) * 100
    ax.plot(x, realized_curve, color="#27ae60")
    ax.axvline(contact_stats["share_contacted"] * 100, color="k", ls="--", lw=1, label="Model's rule (contact if profitable)")
    ax.axhline(0, color="gray", lw=0.8); ax.legend(fontsize=8)
    ax.set(title="Profit vs how many customers you contact", xlabel="% customers contacted (best expected profit first)", ylabel="Realized profit (test set)")
    fig.tight_layout(); fig.savefig(out / "profit_curve.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    importance.sort_values().tail(12).plot.barh(ax=ax, color="#3b6fb6")
    ax.set(title="Top churn drivers (permutation importance: AUC drop)")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=130); plt.close(fig)
