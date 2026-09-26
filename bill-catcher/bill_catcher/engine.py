"""Score estimated bills before they are sent, and decide what to do with each.

Input: one row per estimated bill, as MeterHub/Aurora's nightly batch would produce it
(see BILL_COLUMNS). Output: risk, reason codes and an action per bill.

Two layers:
  1. Policy rules that always apply (no real read in 12+ months, outage in period, wild estimate).
  2. A logistic risk score. Day one uses PRIOR_COEF (expert weights) with the intercept calibrated
     so the average risk equals Northwind's real exception rate. learn() refits the weights on
     corrected-bill outcomes: the feedback loop MeterHub has never had.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BILL_COLUMNS = ["bill_id", "account_id", "region", "months_since_actual_read", "estimated_kwh",
                "last_year_kwh", "prior_corrections_12m", "outage_in_period", "crosses_season"]

FEATURES = ["months_since_actual_read", "abs_deviation_pct", "prior_corrections_12m",
            "outage_in_period", "crosses_season", "over_6_months"]
LABELS = {
    "months_since_actual_read": "many months since a real read",
    "abs_deviation_pct": "estimate far from last year's usage",
    "prior_corrections_12m": "account had corrected bills this year",
    "outage_in_period": "outage in bill period (GridWatch)",
    "crosses_season": "estimate spans a season change",
    "over_6_months": "over 6 months since a real read",
}
# Day-one weights (expert judgement, NOT fitted: the data pack has no bill-level history).
# over_6_months starts at 0 so the feedback loop has something to discover.
PRIOR_COEF = {"months_since_actual_read": 0.25, "abs_deviation_pct": 0.025, "prior_corrections_12m": 0.9,
              "outage_in_period": 0.8, "crosses_season": 0.5, "over_6_months": 0.0}

SEND, SELF_READ, READ_OR_VISIT = "SEND", "SELF_READ_PROMPT", "SELF_READ_THEN_FIELD_VISIT"
_RANK = {SEND: 0, SELF_READ: 1, READ_OR_VISIT: 2}


def features(bills):
    last = bills.last_year_kwh.where(bills.last_year_kwh > 0)
    dev = ((bills.estimated_kwh - last).abs() / last * 100).fillna(100).clip(0, 300)
    return pd.DataFrame({
        "months_since_actual_read": bills.months_since_actual_read.clip(0, 36),
        "abs_deviation_pct": dev,
        "prior_corrections_12m": bills.prior_corrections_12m.clip(0, 5),
        "outage_in_period": bills.outage_in_period.astype(int),
        "crosses_season": bills.crosses_season.astype(int),
        "over_6_months": (bills.months_since_actual_read > 6).astype(int),
    }, index=bills.index)


def solve_intercept(z, target_rate):
    """Intercept that makes the mean predicted probability equal target_rate."""
    lo, hi = -30.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(z + mid)))).mean() < target_rate else (lo, mid)
    return (lo + hi) / 2


class Catcher:
    def __init__(self, coef, intercept, source, reference_mean):
        self.coef = {f: float(coef[f]) for f in FEATURES}
        self.intercept = float(intercept)
        self.source = source
        self.reference_mean = {f: float(reference_mean[f]) for f in FEATURES}

    @property
    def w(self):
        return np.array([self.coef[f] for f in FEATURES])

    @classmethod
    def prior(cls, reference_bills, base_rate):
        X = features(reference_bills)
        w = np.array([PRIOR_COEF[f] for f in FEATURES])
        return cls(PRIOR_COEF, solve_intercept(X[FEATURES].values @ w, base_rate),
                   f"prior weights, intercept calibrated to real exception rate {base_rate:.1%}", X.mean())

    def learn(self, bills, needs_correction):
        """Refit on corrected-bill outcomes (1 = bill needed correction)."""
        X = features(bills)
        lr = LogisticRegression(max_iter=2000).fit(X[FEATURES], needs_correction)
        return Catcher(dict(zip(FEATURES, lr.coef_[0])), lr.intercept_[0],
                       f"learned from {len(X):,} corrected-bill outcomes", X.mean())

    def risk(self, X):
        return 1 / (1 + np.exp(-(X[FEATURES].values @ self.w + self.intercept)))

    def reasons(self, X, k=2):
        # contribution relative to an average estimated bill, so reasons explain what is unusual
        contrib = (X[FEATURES].values - np.array([self.reference_mean[f] for f in FEATURES])) * self.w
        out = []
        for row in contrib:
            top = [j for j in np.argsort(-row)[:k] if row[j] > 0.05]
            out.append("; ".join(LABELS[FEATURES[j]] for j in top) or "-")
        return out

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump(self.__dict__, fh, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        return cls(d["coef"], d["intercept"], d["source"], d["reference_mean"])


def decide(bills, catcher, cal, mode="hold"):
    """Risk, reasons and action per bill. mode='shadow' recommends but sends everything."""
    a, u = cal["assumptions"], cal["unit_costs"]
    X = features(bills)
    out = bills.copy()
    out["risk"] = catcher.risk(X)
    out["reasons"] = catcher.reasons(X)

    # economic hold: expected saving from a self-read beats its cost
    saving = cal["saving_per_bad_bill"]
    economic = out.risk * a["self_read_response_rate"] * saving > a["self_read_cost"]
    action = pd.Series(np.where(economic, SELF_READ, SEND), index=out.index)
    rule = pd.Series("", index=out.index)

    # policy rules override the score (never downgrade)
    rules = [
        ("no real read in 12+ months", X.months_since_actual_read >= 12, READ_OR_VISIT),
        ("outage in bill period", X.outage_in_period == 1, SELF_READ),
        ("estimate >60% off last year", X.abs_deviation_pct >= 60, SELF_READ),
    ]
    for name, mask, act in rules:
        upgrade = mask & (action.map(_RANK) < _RANK[act])
        action[upgrade] = act
        rule[mask] = np.where(rule[mask] == "", name, rule[mask] + "; " + name)

    out["action"] = action
    out["rule"] = rule
    miss = 1 - a["self_read_response_rate"]
    out["hold_cost"] = action.map({SEND: 0.0, SELF_READ: a["self_read_cost"],
                                   READ_OR_VISIT: a["self_read_cost"] + miss * u["field_visit"]})
    # probability a bad bill is actually stopped: self-read only works if the customer responds
    out["p_stop"] = action.map({SEND: 0.0, SELF_READ: a["self_read_response_rate"], READ_OR_VISIT: 1.0})
    out["action_taken"] = SEND if mode == "shadow" else out.action
    return out.sort_values("risk", ascending=False)


def summarize(scored, cal, outcome_col="needs_correction"):
    held = scored[scored.action != SEND]
    s = {
        "bills": len(scored),
        "self_read_prompts": int((scored.action == SELF_READ).sum()),
        "read_or_visit": int((scored.action == READ_OR_VISIT).sum()),
        "held_share": len(held) / max(len(scored), 1),
        "expected_bad_bills": float(scored.risk.sum()),
        "expected_bad_stopped": float((held.risk * held.p_stop).sum()),
        "hold_cost": float(held.hold_cost.sum()),
    }
    s["expected_saving"] = s["expected_bad_stopped"] * cal["saving_per_bad_bill"]
    s["expected_net"] = s["expected_saving"] - s["hold_cost"]
    if outcome_col in scored:
        from sklearn.metrics import roc_auc_score
        bad = scored[outcome_col] == 1
        s["actual_bad_bills"] = int(bad.sum())
        s["bad_in_hold_queue"] = int(held[outcome_col].sum())
        s["recall_of_queue"] = s["bad_in_hold_queue"] / max(s["actual_bad_bills"], 1)
        s["precision_of_queue"] = s["bad_in_hold_queue"] / max(len(held), 1)
        s["auc"] = float(roc_auc_score(scored[outcome_col], scored.risk))
    return s
