"""Retention-team dashboard.   streamlit run app/dashboard.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import streamlit as st
from churn.config import load_config, resolve
from churn.monitor import drift_report
from churn.predict import ChurnModel

st.set_page_config(page_title="Churn Command Center", page_icon="📉", layout="wide")


@st.cache_resource(show_spinner="Preparing the model for this server (first run only, about 1-2 minutes)...")
def get_model():
    from churn.data import load_data
    try:
        m = ChurnModel.load()
        m.score(load_data().head(5), explain=False)   # smoke test: can this server really use the file?
        return m
    except Exception:
        from churn.train import run
        run()                                          # retrain with THIS server's library versions
        return ChurnModel.load()


try:
    model = get_model()
except FileNotFoundError as e:
    st.error(f"{e}"); st.stop()
cfg, meta = load_config(), model.meta
st.title("📉 Churn Command Center")
st.caption(f"Model: {meta['model_family']} · v{meta['version']} · trained {meta['trained_at'][:10]} · "
           f"test AUC {meta['metrics']['roc_auc']}")

t1, t2, t3, t4 = st.tabs(["📋 Portfolio & contact list", "🔍 Single customer / what-if", "📈 Model quality", "🚨 Drift monitor"])

with t1:
    up = st.file_uploader("Upload customers (CSV with the fields from /model-info)", type="csv", key="batch")
    path = resolve(cfg["data"]["raw_path"])
    src = pd.read_csv(up) if up else (pd.read_csv(path) if path.exists() else None)
    if src is None:
        st.info("Upload a CSV to score customers.")
    else:
        if not up: st.caption(f"Showing demo data: {path.name}")
        try:
            res = model.score(src)
        except Exception as e:
            st.error(f"Could not score file: {e}"); st.stop()
        c = st.columns(4)
        c[0].metric("Customers", f"{len(res):,}")
        c[1].metric("High risk", f"{(res.risk_tier == 'HIGH').sum():,}")
        c[2].metric("Worth contacting", f"{int(res.contact.sum()):,}")
        c[3].metric("Value at risk (contact list)", f"{res.loc[res.contact, 'value_at_risk'].sum():,.0f}")
        st.bar_chart(res["risk_tier"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]))
        only = st.checkbox("Show only customers worth contacting", True)
        view = res[res.contact] if only else res
        view = view.sort_values("expected_profit_if_contacted", ascending=False)
        st.dataframe(view.head(500), width="stretch", hide_index=True)
        st.download_button("⬇ Download contact list", view.to_csv(index=False), "contact_list.csv")

with t2:
    lv = meta["category_levels"]
    a, b, c = st.columns(3)
    with a:
        contract = st.selectbox("Contract", lv["contract"]); internet = st.selectbox("Internet", lv["internet_service"])
        tenure = st.slider("Tenure (months)", 0, 72, 4); pay = st.selectbox("Payment method", lv["payment_method"])
    with b:
        monthly = st.number_input("Monthly charges", 18.0, 130.0, 90.0); calls = st.slider("Support calls", 0, 10, 4)
        late = st.slider("Late payments", 0, 6, 1); tech = st.selectbox("Tech support", lv["tech_support"])
    with c:
        sec = st.selectbox("Online security", lv["online_security"]); senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Partner", lv["partner"]); dep = st.selectbox("Dependents", lv["dependents"])
        paperless = st.selectbox("Paperless billing", lv["paperless_billing"])
    row = pd.DataFrame([{"gender": "Female", "senior_citizen": senior, "partner": partner, "dependents": dep,
                         "tenure_months": tenure, "phone_service": "Yes", "internet_service": internet, "online_security": sec,
                         "tech_support": tech, "contract": contract, "paperless_billing": paperless, "payment_method": pay,
                         "monthly_charges": monthly, "total_charges": monthly * tenure, "support_calls": calls, "late_payments": late}])
    r = model.score(row).iloc[0]
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{r.churn_probability:.1%}"); m2.metric("Risk tier", r.risk_tier)
    m3.metric("Expected profit if contacted", f"{r.expected_profit_if_contacted:,.0f}")
    (st.success if not r.contact else st.warning)("Contact this customer" if r.contact else "Not cost-effective to contact")
    st.write("**Why:**", r.top_risk_drivers); st.write("**Recommended action:**", r.recommended_action)

with t3:
    mp = resolve("reports/metrics.json")
    if mp.exists():
        s = json.loads(mp.read_text()); tm, pr = s["test_metrics"], s["profit"]
        c = st.columns(5)
        for col, (k, lab) in zip(c, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("brier", "Brier"), ("lift_top_decile", "Top-decile lift"), ("recall_in_top_20pct", "Churners in top 20%")]):
            col.metric(lab, tm[k])
        st.subheader("Profit on held-out customers")
        st.bar_chart(pd.Series({"Contact nobody": 0, "Contact everybody": pr["contact_everybody"], "Model rule": pr["model_rule"], "Perfect knowledge": pr["perfect_knowledge"]}))
        for img in ["roc_pr_curves", "lift_gains", "calibration", "profit_curve", "feature_importance"]:
            p = resolve(f"reports/{img}.png")
            if p.exists(): st.image(str(p))
        fp = resolve("reports/fairness.csv")
        if fp.exists(): st.subheader("Fairness audit"); st.dataframe(pd.read_csv(fp), hide_index=True)
    else:
        st.info("Run `python -m churn.train` to generate reports.")

with t4:
    st.write("Upload recent customer data to check whether it still looks like the data the model was trained on.")
    up2 = st.file_uploader("Recent customers CSV", type="csv", key="drift")
    if up2:
        new = pd.read_csv(up2)
        rep = drift_report(meta["reference"], new, model.predict_proba(new), cfg["drift"]["psi_warn"], cfg["drift"]["psi_alert"])
        n = (rep.status == "ALERT").sum()
        (st.error if n else st.success)(f"{n} feature(s) in ALERT - consider retraining." if n else "No major drift detected.")
        st.dataframe(rep, hide_index=True)
