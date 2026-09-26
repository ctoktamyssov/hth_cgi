"""Derive every number the catcher can take from the Northwind data pack.

The pack has no bill-level records, so it cannot train a per-bill model. What it
CAN supply, and what this module extracts:
  - base rate: billing exceptions / estimated bills (flat ~4.1% in every region and month)
  - volumes: estimated bills per month per region
  - severity: average bill correction value on estimated-read complaints, per region
  - complaint conversion: estimated-read complaints per billing exception
  - unit costs: correction, field visit, complaint handling, smart meter
Anything not in the pack is listed under "assumptions" so it can be challenged.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT.parent / "northwind-dashboard" / "data"
CAL_PATH = ROOT / "calibration.json"

ASSUMPTIONS = {
    "self_read_cost": 2.0,            # SMS/app prompt + handling per held bill
    "self_read_response_rate": 0.70,  # share of prompted customers who submit a read in time
    "hard_to_access_share": 0.015,    # accounts whose meter is rarely read (locked access etc.), simulator only
    "catcher_build": 450_000,         # ~12 weeks, 3 devs + 1 analyst (same figure as the dashboard)
    "catcher_run_per_year": 150_000,
}


def _unit_cost(costs, starts_with):
    return float(costs[costs["item"].str.startswith(starts_with)].unit_cost.iloc[0])


def calibrate(data_dir=DEFAULT_DATA):
    data_dir = Path(data_dir)
    m = pd.read_csv(data_dir / "northwind_meter_reads.csv")
    c = pd.read_csv(data_dir / "northwind_complaints.csv", parse_dates=["date_opened"])
    costs = pd.read_csv(data_dir / "northwind_unit_costs.csv")

    m["est_bills"] = m.accounts * m.estimated_read_rate
    last12 = m[m.month.isin(sorted(m.month.unique())[-12:])]
    est = c[c.category == "Billing - estimated read"]

    regions = {}
    for region, g in last12.groupby("region"):
        latest = g.sort_values("month").iloc[-1]
        e = est[est.region == region]
        regions[region] = {
            "accounts": int(latest.accounts),
            "estimated_read_rate": round(float(g.estimated_read_rate.mean()), 4),
            "smart_meter_penetration": float(latest.smart_meter_penetration),
            "est_bills_month": round(float(g.est_bills.mean())),
            "exceptions_month": round(float(g.billing_exceptions_raised.mean())),
            "avg_correction_value": round(float(e.bill_correction_value.mean()), 2),
            "systems": latest.systems_serving_region,
            "legacy": "SYS-01" in latest.systems_serving_region,
        }

    months = c.date_opened.dt.to_period("M").nunique()
    complaints_per_exception = (len(est) / months) / m.groupby("month").billing_exceptions_raised.sum().mean()
    xfer = est.transferred_between_systems.mean()
    unit = {
        "bill_correction": _unit_cost(costs, "Manual bill correction"),
        "field_visit": _unit_cost(costs, "Field meter visit"),
        "smart_meter": _unit_cost(costs, "Smart meter installation"),
        "complaint_avg": _unit_cost(costs, "Complaint handled end to end (average)"),
        "complaint_transferred": _unit_cost(costs, "Complaint handled end to end (transferred"),
    }
    complaint_cost = xfer * unit["complaint_transferred"] + (1 - xfer) * unit["complaint_avg"]

    return {
        "source": str(data_dir),
        "base_exception_rate": round(float(m.billing_exceptions_raised.sum() / m.est_bills.sum()), 5),
        "complaints_per_exception": round(float(complaints_per_exception), 5),
        "estimated_read_complaint_cost": round(float(complaint_cost), 2),
        # what one bad estimated bill costs Northwind if it goes out: a manual correction + chance of a complaint
        "saving_per_bad_bill": round(unit["bill_correction"] + complaints_per_exception * complaint_cost, 2),
        "unit_costs": unit,
        "regions": regions,
        "assumptions": ASSUMPTIONS,
    }


def load_calibration(path=CAL_PATH, data_dir=DEFAULT_DATA):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    cal = calibrate(data_dir)
    path.write_text(json.dumps(cal, indent=2))
    return cal
