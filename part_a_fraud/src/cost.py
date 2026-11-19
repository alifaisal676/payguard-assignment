"""Cost-based threshold selection, decision tiers and sensitivity analysis.

All money is in EUR (the dataset is European card transactions). Every number below is an
ASSUMPTION, stated explicitly so the business can replace it with real figures:

    Missed fraud (FN)   = transaction amount + CHARGEBACK_FEE + FN_OPS_COST
    Any alert           = REVIEW_COST            (analyst / step-up handling, paid for TP and FP)
    False alarm (FP)    = + FRICTION_COST        (blocked legit payment: support contact, churn risk)
    Caught fraud (TP)   = review cost only       (the payment is stopped, so no loss)

Two ways to turn scores into decisions are compared:
1. Global threshold: one cut-off, chosen by minimising total cost on VALIDATION.
2. Amount-aware rule: flag when the expected loss of letting it through exceeds the expected
   cost of an alert:  p * (amount + FN_FIXED) > REVIEW_COST + (1 - p) * FRICTION_COST.
   This needs p to behave like a probability (checked in the notebook: LightGBM yes, LogReg no).

Usage (from the repo root):
    python -m part_a_fraud.src.cost
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class Costs:
    chargeback_fee: float = 15.0   # scheme/acquirer fee per chargeback
    fn_ops_cost: float = 10.0      # handling the dispute, customer contact
    review_cost: float = 4.0       # ~3 min of analyst time at ~EUR 80/h (or OTP + handling)
    friction_cost: float = 10.0    # wrongly blocked legit payment: support call, lost sale, churn risk

    @property
    def fn_fixed(self) -> float:
        return self.chargeback_fee + self.fn_ops_cost


DEFAULT = Costs()


def total_cost(y, flagged, amount, c: Costs = DEFAULT) -> dict:
    y, flagged, amount = np.asarray(y), np.asarray(flagged, bool), np.asarray(amount, float)
    fraud = y == 1
    fn, tp, fp = fraud & ~flagged, fraud & flagged, ~fraud & flagged
    parts = {
        "missed fraud": amount[fn].sum() + fn.sum() * c.fn_fixed,
        "reviews": flagged.sum() * c.review_cost,
        "friction": fp.sum() * c.friction_cost,
    }
    return {**parts, "total": sum(parts.values()), "alerts": int(flagged.sum()), "tp": int(tp.sum()),
            "fp": int(fp.sum()), "fn": int(fn.sum()),
            "fraud amount caught": amount[tp].sum(), "fraud amount missed": amount[fn].sum()}


def candidate_thresholds(s) -> np.ndarray:
    """All distinct score values (plus +inf = flag nothing). Exact, and cheap enough here."""
    return np.r_[np.unique(s), np.inf]


def cost_curve(y, s, amount, c: Costs = DEFAULT) -> pd.DataFrame:
    """Total cost for every threshold (vectorised over sorted scores)."""
    y, s, amount = np.asarray(y), np.asarray(s), np.asarray(amount, float)
    order = np.argsort(-s, kind="stable")
    ys, ss, am = y[order], s[order], amount[order]
    # Flagging the top k rows: cumulative counts as k grows.
    tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
    caught_amt = np.cumsum(np.where(ys == 1, am, 0.0))
    total_fraud_amt, n_fraud = am[ys == 1].sum(), (ys == 1).sum()
    # Only cut between distinct scores (ties are flagged together).
    last_of_value = np.r_[ss[1:] != ss[:-1], True]
    k = np.arange(1, len(ss) + 1)[last_of_value]
    tp, fp, caught_amt, thr = tp[last_of_value], fp[last_of_value], caught_amt[last_of_value], ss[last_of_value]
    fn = n_fraud - tp
    total = (total_fraud_amt - caught_amt) + fn * c.fn_fixed + k * c.review_cost + fp * c.friction_cost
    curve = pd.DataFrame({"threshold": thr, "alerts": k, "tp": tp, "fp": fp, "fn": fn, "total": total})
    nothing = pd.DataFrame({"threshold": [np.inf], "alerts": [0], "tp": [0], "fp": [0], "fn": [n_fraud],
                            "total": [total_fraud_amt + n_fraud * c.fn_fixed]})
    return pd.concat([nothing, curve], ignore_index=True)


def best_threshold(y, s, amount, c: Costs = DEFAULT) -> float:
    curve = cost_curve(y, s, amount, c)
    return float(curve.loc[curve["total"].idxmin(), "threshold"])


def amount_aware_flags(p, amount, c: Costs = DEFAULT) -> np.ndarray:
    """Flag when expected loss of approving > expected cost of alerting (needs calibrated p)."""
    p, amount = np.asarray(p), np.asarray(amount, float)
    return p * (amount + c.fn_fixed) > c.review_cost + (1 - p) * c.friction_cost


def threshold_for_precision(y, s, target: float) -> float:
    precision, _, thresholds = precision_recall_curve(y, s)
    ok = np.where(precision[:-1] >= target)[0]
    return float(thresholds[ok[0]]) if len(ok) else float("inf")


def tiers(s, t_stepup: float, t_block: float) -> np.ndarray:
    """approve < t_stepup <= step-up (OTP / manual review) < t_block <= block."""
    s = np.asarray(s)
    return np.where(s >= t_block, "block", np.where(s >= t_stepup, "step-up", "approve"))


def tier_table(y, s, amount, t_stepup, t_block) -> pd.DataFrame:
    t = tiers(s, t_stepup, t_block)
    df = pd.DataFrame({"tier": t, "fraud": np.asarray(y), "amount": np.asarray(amount, float)})
    out = df.groupby("tier").agg(transactions=("fraud", "size"), frauds=("fraud", "sum"),
                                 fraud_amount=("amount", lambda a: a[df.loc[a.index, "fraud"] == 1].sum()))
    out["precision"] = out["frauds"] / out["transactions"]
    return out.reindex(["block", "step-up", "approve"]).fillna(0)


def sensitivity(y_val, s_val, a_val, y_te, s_te, a_te, friction_values=(1, 2, 5, 10, 25, 50, 100, 250)) -> pd.DataFrame:
    """How the val-chosen threshold, and its test outcome, move as the false-alarm cost changes."""
    rows = []
    for f in friction_values:
        c = replace(DEFAULT, friction_cost=float(f))
        t = best_threshold(y_val, s_val, a_val, c)
        r = total_cost(y_te, np.asarray(s_te) >= t, a_te, c)
        rows.append({"friction cost (EUR)": f, "FP:FN-fixed cost ratio": round((c.review_cost + f) / c.fn_fixed, 2),
                     "threshold (val)": t, "test alerts": r["alerts"], "test frauds caught": r["tp"],
                     "test false alarms": r["fp"], "test total cost": round(r["total"])})
    return pd.DataFrame(rows)


def compare_policies(y_te, s_te, a_te, t_cost: float, alert_budget_k: int, c: Costs = DEFAULT) -> pd.DataFrame:
    """Test-set cost of each policy. Thresholds were fixed on validation beforehand."""
    s_te = np.asarray(s_te)
    top_k = np.zeros(len(s_te), bool); top_k[np.argsort(-s_te, kind="stable")[:alert_budget_k]] = True
    policies = {
        "no model (approve all)": np.zeros(len(s_te), bool),
        f"review top {alert_budget_k} (alert budget)": top_k,
        "global cost-optimal threshold": s_te >= t_cost,
        "amount-aware rule": amount_aware_flags(s_te, a_te, c),
    }
    return pd.DataFrame({k: total_cost(y_te, f, a_te, c) for k, f in policies.items()}).T.round(0)


if __name__ == "__main__":
    from part_a_fraud.src.data import load, time_split, xy
    from part_a_fraud.src.train import load_models, score

    _, val, test = time_split(load())
    model = load_models()["lightgbm"]
    s_val, s_te = score(model, xy(val)[0]), score(model, xy(test)[0])
    y_val, y_te = val["Class"].values, test["Class"].values
    t = best_threshold(y_val, s_val, val["Amount"])
    print(f"Cost-optimal threshold on validation: {t:.4f}  (assumptions: {DEFAULT})")
    print(compare_policies(y_te, s_te, test["Amount"], t, round(len(test) * 0.002)).to_string())
