"""Business economics: turn probabilities into money and contact decisions.

A good churn system does NOT just rank by risk. A customer is worth contacting only if
    P(churn) * P(offer saves them) * value_if_saved  >  cost_of_offer
"""
import numpy as np


def customer_value(monthly_charges, biz: dict):
    """Margin earned from a customer over the planning horizon."""
    return np.asarray(monthly_charges, dtype=float) * biz["gross_margin"] * biz["horizon_months"]


def expected_profit(p, monthly_charges, biz: dict):
    """Expected net profit of contacting each customer."""
    return np.asarray(p) * biz["offer_success_rate"] * customer_value(monthly_charges, biz) - biz["offer_cost"]


def profit_analysis(y_true, p, monthly_charges, biz: dict) -> dict:
    """Compare retention strategies on held-out data using ACTUAL outcomes."""
    y = np.asarray(y_true)
    value = customer_value(monthly_charges, biz)
    s, c = biz["offer_success_rate"], biz["offer_cost"]
    realized = y * s * value - c                       # profit if this customer is contacted
    contact = expected_profit(p, monthly_charges, biz) > 0
    return {
        "contact_nobody": 0.0,
        "contact_everybody": float(realized.sum()),
        "model_rule": float(realized[contact].sum()),
        "perfect_knowledge": float(realized[y == 1].clip(min=0).sum()),
        "customers_contacted": int(contact.sum()),
        "share_contacted": float(contact.mean()),
    }
