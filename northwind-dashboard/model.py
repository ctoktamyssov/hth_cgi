"""Northwind analysis + forecast model.

Everything here is derived from the six synthetic CSVs in ./data.
Any number that is NOT from the data is an ASSUMPTION and lives in
DEFAULT_ASSUMPTIONS so it can be shown, edited and challenged.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = Path(__file__).parent / "data"
LEGACY_REGIONS = ["Barrowdale", "Dunmoor"]  # SYS-01/SYS-06, 0% smart meters
METER_CATS = ["Billing - estimated read", "Metering - no read taken"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load():
    d = {
        "complaints": pd.read_csv(DATA / "northwind_complaints.csv", parse_dates=["date_opened", "date_closed"]),
        "kpis": pd.read_csv(DATA / "northwind_monthly_kpis.csv"),
        "meters": pd.read_csv(DATA / "northwind_meter_reads.csv"),
        "pilot": pd.read_csv(DATA / "northwind_ai_pilot_2025.csv"),
        "systems": pd.read_csv(DATA / "northwind_systems.csv"),
        "costs": pd.read_csv(DATA / "northwind_unit_costs.csv"),
    }
    c = d["complaints"]
    c["month"] = c.date_opened.dt.strftime("%Y-%m")
    c["legacy_region"] = c.region.isin(LEGACY_REGIONS)
    c["meter_driven"] = c.category.isin(METER_CATS)
    k = d["kpis"]
    k["backlog"] = (k.complaints_opened - k.complaints_closed).cumsum()
    return d


def unit_cost(d, starts_with):
    row = d["costs"][d["costs"]["item"].str.startswith(starts_with)]
    return float(row.unit_cost.iloc[0])


# ----------------------------------------------------------------------------
# Diagnosis
# ----------------------------------------------------------------------------
def region_table(d):
    m, c = d["meters"], d["complaints"]
    g = m.groupby("region").agg(
        accounts=("accounts", "mean"),
        estimated_read_rate=("estimated_read_rate", "mean"),
        smart_meter_penetration=("smart_meter_penetration", "mean"),
        exceptions_per_month=("billing_exceptions_raised", "mean"),
        systems=("systems_serving_region", "first"),
    )
    g["exceptions_per_10k"] = g.exceptions_per_month / g.accounts * 1e4
    cc = c.groupby("region").agg(complaints=("complaint_id", "size"), meter_share=("meter_driven", "mean"))
    g = g.join(cc)
    g["complaints_per_10k"] = g.complaints / g.accounts * 1e4
    return g.reset_index().sort_values("estimated_read_rate", ascending=False)


def region_month(d):
    """Join meter reads to complaints at region-month grain — the 'two files' link."""
    c, m = d["complaints"], d["meters"]
    mc = c[c.meter_driven].groupby(["month", "region"]).size().rename("meter_complaints")
    rm = m.set_index(["month", "region"]).join(mc).fillna(0).reset_index()
    rm["meter_complaints_per_10k"] = rm.meter_complaints / rm.accounts * 1e4
    rm["exceptions_per_10k"] = rm.billing_exceptions_raised / rm.accounts * 1e4
    rm["legacy"] = rm.region.isin(LEGACY_REGIONS)
    return rm


def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return slope, intercept, r


def transfer_table(d):
    c = d["complaints"]
    t = c.groupby("transferred_between_systems").agg(
        complaints=("complaint_id", "size"),
        avg_days=("days_to_close", "mean"),
        sla_breach=("sla_breach", "mean"),
        reopened=("reopened", "mean"),
    )
    t.index = t.index.map({0: "Stayed in one system", 1: "Transferred between systems"})
    t["unit_cost"] = [unit_cost(d, "Complaint handled end to end (average)"),
                      unit_cost(d, "Complaint handled end to end (transferred")]
    return t


def source_transfer_table(d):
    c = d["complaints"]
    names = d["systems"].set_index("system_id").system_name
    t = c.groupby("source_system").agg(complaints=("complaint_id", "size"),
                                       transferred=("transferred_between_systems", "mean"),
                                       avg_days=("days_to_close", "mean")).reset_index()
    t["system"] = t.source_system + " " + t.source_system.map(names)
    return t


def category_table(d):
    c = d["complaints"]
    t = c.groupby("category").agg(
        complaints=("complaint_id", "size"),
        info_only=("resolvable_by_information_only", "mean"),
        sla_breach=("sla_breach", "mean"),
        avg_days=("days_to_close", "mean"),
    ).sort_values("complaints", ascending=False)
    t["share"] = t.complaints / t.complaints.sum()
    return t.reset_index()


# ----------------------------------------------------------------------------
# Bill catcher (simulated billing run)
# ----------------------------------------------------------------------------
FEATURES = ["months_since_actual_read", "deviation_vs_last_year_pct", "prior_correction_on_account",
            "outage_in_period", "crosses_season"]
FEATURE_LABELS = {
    "months_since_actual_read": "No actual read for many months",
    "deviation_vs_last_year_pct": "Estimate far from last year's usage",
    "prior_correction_on_account": "Account had a corrected bill before",
    "outage_in_period": "Outage during bill period (GridWatch)",
    "crosses_season": "Estimate spans a season change",
}
TRUE_COEF = np.array([0.32, 0.035, 1.3, 1.1, 0.7])


def simulate_bills(region_est_rate, n, seed, target_exception_rate):
    """Synthetic estimated bills. Feature distributions scale with the region's
    estimated-read rate; the true 'will need correction' rate is calibrated to the
    data (billing exceptions ≈ 4.1% of estimated bills in every region)."""
    rng = np.random.default_rng(seed)
    mean_gap = 1 + 5 * region_est_rate  # legacy regions go longer between real reads
    X = pd.DataFrame({
        "months_since_actual_read": np.clip(rng.geometric(1 / mean_gap, n), 1, 18),
        "deviation_vs_last_year_pct": np.abs(rng.normal(0, 18 + 20 * region_est_rate, n)).round(1),
        "prior_correction_on_account": rng.binomial(1, 0.08 + 0.1 * region_est_rate, n),
        "outage_in_period": rng.binomial(1, 0.03, n),
        "crosses_season": rng.binomial(1, 0.25, n),
    })
    z = X.values @ TRUE_COEF
    # solve intercept so mean probability = target
    lo, hi = -20.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(z + mid)))
        lo, hi = (mid, hi) if p.mean() < target_exception_rate else (lo, mid)
    y = rng.binomial(1, p)
    X.insert(0, "bill_id", [f"EB-{seed % 1000:03d}-{i:05d}" for i in range(n)])
    X["needs_correction"] = y
    return X


def train_catcher(region_est_rate, target_rate, seed=7):
    hist = simulate_bills(region_est_rate, 30000, seed, target_rate)
    new = simulate_bills(region_est_rate, 8000, seed + 1, target_rate)
    model = LogisticRegression(max_iter=1000)
    model.fit(hist[FEATURES], hist.needs_correction)
    new["risk"] = model.predict_proba(new[FEATURES])[:, 1]
    auc = roc_auc_score(new.needs_correction, new.risk)
    contrib = new[FEATURES].values * model.coef_[0]
    top = np.argsort(-contrib, axis=1)[:, :2]
    new["why"] = [" · ".join(FEATURE_LABELS[FEATURES[j]] for j in row) for row in top]
    return model, new.sort_values("risk", ascending=False), auc


def threshold_economics(scored, hold_share, cost_hold, cost_correction, complaint_conv, cost_complaint):
    n_hold = int(len(scored) * hold_share)
    held = scored.head(n_hold)
    caught = int(held.needs_correction.sum())
    total_bad = int(scored.needs_correction.sum())
    precision = caught / n_hold if n_hold else 0.0
    recall = caught / total_bad if total_bad else 0.0
    benefit = caught * (cost_correction + complaint_conv * cost_complaint)
    cost = n_hold * cost_hold
    return dict(n_hold=n_hold, caught=caught, total_bad=total_bad, precision=precision,
                recall=recall, benefit=benefit, cost=cost, net=benefit - cost)


def recall_curve(scored):
    s = scored.reset_index(drop=True)
    cum = s.needs_correction.cumsum()
    share = (np.arange(len(s)) + 1) / len(s)
    return pd.DataFrame({"share_held": share, "recall": cum / cum.iloc[-1]})


# ----------------------------------------------------------------------------
# Forecast + value
# ----------------------------------------------------------------------------
DEFAULT_ASSUMPTIONS = {
    # costs (estimates — not in data pack)
    "catcher_build": 450_000,        # 3 devs + 1 analyst, ~12 weeks, incl. MeterHub integration
    "catcher_run": 150_000,          # hosting, model monitoring, 0.5 FTE analyst
    "case_layer_build": 1_600_000,   # integration layer giving CaseTrack full history from CIS/billing/CRM
    "case_layer_run": 300_000,
    "self_read_cost": 2.0,           # SMS/app prompt + handling per held bill (field visit $92 used only as fallback)
    "field_visit_fallback": 0.05,    # share of held bills that still need a field visit
    "regulator_breach_threshold": 3.0,  # score below which enhanced-monitoring penalty applies (ask COO)
    "score_adjust_speed": 0.10,      # regulator score moves 10% of the way to its 'implied' level each month
}


def score_from_days(days):
    # fitted on 24 months of KPIs: score = 5.316 - 0.0691 * avg_days_to_close  (r ≈ -0.99)
    return np.clip(5.316 - 0.0691 * days, 1, 5)


def forecast(d, levers, a=DEFAULT_ASSUMPTIONS, months=12):
    """Month-by-month backlog simulation.

    levers: catcher_on, catcher_recall, catcher_hold_share, case_layer_on, transfer_cut, smart_meter_share,
            ai_on, ai_containment, surge_fte, surge_months
    """
    k, c, m = d["kpis"], d["complaints"], d["meters"]
    recent = k.tail(12)  # trend over the last 12 months (earlier months include the backlog build-up start)
    t = np.arange(len(recent))
    op_s, op_i = np.polyfit(t, recent.complaints_opened, 1)
    cl_s, cl_i = np.polyfit(t, recent.complaints_closed, 1)
    days_floor = float(k.avg_days_to_close.iloc[0])  # 9.1 days: best observed, when backlog was small
    backlog0 = float(k.backlog.iloc[-1])
    days0 = float(k.avg_days_to_close.iloc[-1])
    score0 = float(k.regulator_satisfaction_score_of_5.iloc[-1])
    little_cal = days0 / (backlog0 / k.complaints_closed.iloc[-1] * 30)  # calibrates Little's law to 38.2 days

    share_est = (c.category == "Billing - estimated read").mean()
    share_meter = c.meter_driven.mean()
    legacy_meter_share = c[c.meter_driven].legacy_region.mean()
    share_info = c.resolvable_by_information_only.mean()
    xfer0 = c.transferred_between_systems.mean()
    cost_avg = unit_cost(d, "Complaint handled end to end (average)")
    cost_xfer = unit_cost(d, "Complaint handled end to end (transferred")
    cost_corr = unit_cost(d, "Manual bill correction")
    agent = unit_cost(d, "Contact centre agent")
    meter_cost = unit_cost(d, "Smart meter installation")
    penalty_q = unit_cost(d, "Regulator penalty")
    ai_cost = unit_cost(d, "AskNorthwind")

    mix_cost0 = xfer0 * cost_xfer + (1 - xfer0) * cost_avg
    per_fte_month = agent / 12 / mix_cost0  # complaints an extra agent closes per month

    months_sorted = sorted(m.month.unique())[-12:]
    last12 = m[m.month.isin(months_sorted)]
    exceptions_yr = last12.billing_exceptions_raised.sum()
    legacy_exc_yr = last12[last12.region.isin(LEGACY_REGIONS)].billing_exceptions_raised.sum()
    legacy_accounts = m[m.region.isin(LEGACY_REGIONS)].groupby("region").accounts.last().sum()
    latest = m[m.month == m.month.max()]
    est_bills_month = float((latest.accounts * latest.estimated_read_rate).sum())

    rows = []
    backlog, score = backlog0, score0
    for i in range(1, months + 1):
        tt = len(recent) - 1 + i
        opened = op_s * tt + op_i
        capacity = cl_s * tt + cl_i
        ramp = lambda start, full: float(np.clip((i - start) / max(full - start, 1), 0, 1))

        prevented = 0.0
        exc_avoided = 0.0
        if levers["catcher_on"]:
            r = levers["catcher_recall"] * ramp(3, 5)  # live after a 90-day build, full by month 5
            prevented += opened * share_est * r
            exc_avoided += exceptions_yr / 12 * r
        sm = levers["smart_meter_share"] * ramp(0, months)  # linear rollout over the year
        if sm > 0:
            # legacy estimated-read rate ~0.60 -> ~0.20 (as in smart regions) for covered accounts
            cut = sm * (1 - 0.20 / 0.60)
            prevented += opened * share_meter * legacy_meter_share * cut
            exc_avoided += legacy_exc_yr / 12 * cut
        if levers["ai_on"]:
            prevented += opened * share_info * levers["ai_containment"] * ramp(1, 3)
        opened_adj = opened - prevented

        xfer = xfer0 * (1 - levers["transfer_cut"] * ramp(4, 9)) if levers["case_layer_on"] else xfer0
        mix_cost = xfer * cost_xfer + (1 - xfer) * cost_avg
        capacity_adj = capacity * mix_cost0 / mix_cost
        if i <= levers["surge_months"]:
            capacity_adj += levers["surge_fte"] * per_fte_month

        closed = min(capacity_adj, backlog + opened_adj)
        backlog = max(backlog + opened_adj - closed, 0)
        days = max(days_floor, little_cal * backlog / max(closed, 1) * 30)
        score = score + a["score_adjust_speed"] * (score_from_days(days) - score)

        # money (this month)
        handling_saved = prevented * mix_cost + closed * (mix_cost0 - mix_cost)
        correction_saved = exc_avoided * cost_corr
        held_bills = est_bills_month * levers["catcher_hold_share"] * ramp(3, 5) if levers["catcher_on"] else 0
        cost = 0.0
        if levers["catcher_on"]:
            cost += (a["catcher_build"] / 3 if i <= 3 else 0) + a["catcher_run"] / 12
            cost += held_bills * (a["self_read_cost"] + a["field_visit_fallback"] * unit_cost(d, "Field meter visit"))
        if levers["case_layer_on"]:
            cost += (a["case_layer_build"] / 6 if i <= 6 else 0) + a["case_layer_run"] / 12
        if sm > 0:
            cost += levers["smart_meter_share"] * legacy_accounts * meter_cost / months
        if levers["ai_on"]:
            cost += ai_cost / 12
        cost += (levers["surge_fte"] * agent / 12) if i <= levers["surge_months"] else 0

        rows.append(dict(month=i, opened=opened_adj, opened_baseline=opened, closed=closed, backlog=backlog,
                         days=days, score=score, prevented=prevented, exceptions_avoided=exc_avoided,
                         handling_saved=handling_saved, correction_saved=correction_saved, cost=cost))
    f = pd.DataFrame(rows)
    # regulator penalty: charged per quarter the score sits below threshold
    q_scores = f.groupby((f.month - 1) // 3).score.last()
    f.attrs["quarters_in_breach"] = int((q_scores < a["regulator_breach_threshold"]).sum())
    f.attrs["penalty_per_quarter"] = penalty_q
    f.attrs["per_fte_month"] = per_fte_month
    return f
