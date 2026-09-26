"""SYNTHETIC MeterHub billing runs, for demos and testing only.

The data pack has no bill-level records. This generator produces estimated bills whose
aggregates match the real pack: region volumes, the region's estimated-read rate, and a
true error rate equal to the real ~4.1% exceptions per estimated bill in every region.

The hidden TRUE_COEF differs from the engine's PRIOR_COEF on purpose (for example a jump
after 6 months without a real read) so the feedback loop has something real to learn.
That difference is planted, not discovered in Northwind data.
"""
import numpy as np
import pandas as pd

from .engine import BILL_COLUMNS, FEATURES, features, solve_intercept

TRUE_COEF = {"months_since_actual_read": 0.12, "abs_deviation_pct": 0.035, "prior_corrections_12m": 1.2,
             "outage_in_period": 1.5, "crosses_season": 0.6, "over_6_months": 1.1}


def billing_run(region, cal, n=20_000, seed=0, with_outcomes=True):
    r = cal["regions"][region]
    est = r["estimated_read_rate"]
    rng = np.random.default_rng(seed)

    # months since a real read: a run of consecutive estimates. Normal accounts are estimated
    # each month with the region's real rate; a small share of meters are hard to access.
    hard = rng.random(n) < cal["assumptions"]["hard_to_access_share"]
    p_read = np.where(hard, 0.10, 1 - est)
    months = np.clip(rng.geometric(p_read), 1, 36)

    last_kwh = rng.lognormal(np.log(700), 0.35, n).round(0)
    drift = 0.12 + 0.25 * est  # legacy 2012 estimation algorithm drifts further (ASSUMPTION)
    est_kwh = (last_kwh * np.clip(1 + rng.normal(0, drift, n) * np.sqrt(months / 2), 0.2, 4)).round(0)

    bills = pd.DataFrame({
        "bill_id": [f"EB-{region[:3].upper()}-{seed:03d}-{i:06d}" for i in range(n)],
        "account_id": [f"ACC-{x:06d}" for x in rng.integers(100_000, 999_999, n)],
        "region": region,
        "months_since_actual_read": months,
        "estimated_kwh": est_kwh,
        "last_year_kwh": last_kwh,
        "prior_corrections_12m": rng.poisson(0.06 + 0.1 * est, n),
        "outage_in_period": rng.binomial(1, 0.03, n),
        "crosses_season": rng.binomial(1, 0.25, n),
    })[BILL_COLUMNS]

    if with_outcomes:
        z = features(bills)[FEATURES].values @ np.array([TRUE_COEF[f] for f in FEATURES])
        p = 1 / (1 + np.exp(-(z + solve_intercept(z, cal["base_exception_rate"]))))
        bills["needs_correction"] = rng.binomial(1, p)
    return bills
