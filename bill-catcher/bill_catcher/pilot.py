"""90-day pilot in Barrowdale + Dunmoor, scaled to their real monthly estimated-bill volumes.

Month 1: shadow mode. Prior model flags bills but everything is sent; corrections are collected.
Month 2+: hold mode. Model is retrained each month on all outcomes collected so far.
Bills are SYNTHETIC (see simulate.py); volumes, error rate, costs and correction values are real.
"""
import pandas as pd
from sklearn.metrics import roc_auc_score

from .engine import SEND, Catcher, decide
from .simulate import billing_run

PILOT_REGIONS = ["Barrowdale", "Dunmoor"]


def run_pilot(cal, regions=PILOT_REGIONS, months=3, sample=40_000, seed=100):
    saving = cal["saving_per_bad_bill"]
    ref = pd.concat([billing_run(r, cal, 5_000, seed - 1, with_outcomes=False) for r in regions])
    catcher = Catcher.prior(ref, cal["base_exception_rate"])
    prior = catcher
    history = []
    rows = []
    for month in range(1, months + 1):
        mode = "shadow" if month == 1 else "hold"
        runs = [billing_run(r, cal, sample, seed + 10 * month + i) for i, r in enumerate(regions)]
        row = dict(month=month, mode=mode, model="prior" if catcher is prior else "learned",
                   est_bills=0.0, held=0.0, field_visits=0.0, bad_bills=0.0, bad_stopped=0.0,
                   hold_cost=0.0, correction_saving=0.0, wrong_billing_stopped=0.0)
        aucs = []
        for region, bills in zip(regions, runs):
            scale = cal["regions"][region]["est_bills_month"] / len(bills)
            s = decide(bills, catcher, cal, mode)
            aucs.append(roc_auc_score(s.needs_correction, s.risk))
            held = s[s.action != SEND]
            stopped = (held.needs_correction * held.p_stop).sum()
            live = mode == "hold"
            row["est_bills"] += len(s) * scale
            row["held"] += len(held) * scale
            row["field_visits"] += (held.hold_cost - cal["assumptions"]["self_read_cost"]).sum() / cal["unit_costs"]["field_visit"] * scale
            row["bad_bills"] += s.needs_correction.sum() * scale
            row["bad_stopped"] += stopped * scale  # in shadow: what it WOULD have stopped
            row["hold_cost"] += held.hold_cost.sum() * scale * live
            row["correction_saving"] += stopped * saving * scale * live
            row["wrong_billing_stopped"] += stopped * cal["regions"][region]["avg_correction_value"] * scale * live
        row["auc"] = sum(aucs) / len(aucs)
        row["net"] = row["correction_saving"] - row["hold_cost"]
        rows.append(row)
        history.extend(runs)
        catcher = prior.learn(pd.concat(history), pd.concat(history).needs_correction)
    return pd.DataFrame(rows), catcher


def value_case(cal, table, regions=PILOT_REGIONS):
    a, u = cal["assumptions"], cal["unit_costs"]
    steady = table.iloc[-1]
    annual_net = steady.net * 12 - a["catcher_run_per_year"]
    accounts = sum(cal["regions"][r]["accounts"] for r in regions)
    return {
        "exceptions_cut": steady.bad_stopped / steady.bad_bills,
        "annual_net_run_rate": annual_net,
        "annual_wrong_billing_stopped": steady.wrong_billing_stopped * 12,
        "build_cost": a["catcher_build"],
        "payback_months": a["catcher_build"] / (annual_net / 12) if annual_net > 0 else float("inf"),
        "smart_meter_cost": accounts * u["smart_meter"],
        "accounts": accounts,
    }
