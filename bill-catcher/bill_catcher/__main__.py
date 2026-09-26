"""CLI: python -m bill_catcher <command>

  calibrate                 derive rates, volumes and costs from the data pack -> calibration.json
  simulate                  write a SYNTHETIC billing run for a region (for demos)
  score BILLS.csv           risk + reason + action for each bill -> hold queue CSV
  learn OUTCOMES.csv        refit the model on corrected-bill outcomes -> model.json
  pilot                     90-day Barrowdale + Dunmoor pilot and value case
"""
import argparse
import json

import pandas as pd

from .calibrate import CAL_PATH, DEFAULT_DATA, calibrate, load_calibration
from .engine import SEND, Catcher, decide, summarize
from .pilot import PILOT_REGIONS, run_pilot, value_case
from .simulate import billing_run


def money(x):
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"${x:,.0f}"


def cmd_calibrate(args):
    cal = calibrate(args.data)
    CAL_PATH.write_text(json.dumps(cal, indent=2))
    print(f"Wrote {CAL_PATH}")
    print(f"  exception rate per estimated bill : {cal['base_exception_rate']:.2%} (same in every region)")
    print(f"  complaints per exception          : {cal['complaints_per_exception']:.2%}")
    print(f"  saving per bad bill stopped       : ${cal['saving_per_bad_bill']:.2f}")
    print(f"  {'region':<12}{'est. bills/mo':>14}{'exceptions/mo':>15}{'avg correction':>16}")
    for r, v in cal["regions"].items():
        print(f"  {r:<12}{v['est_bills_month']:>14,}{v['exceptions_month']:>15,}{v['avg_correction_value']:>16.2f}"
              + ("  legacy" if v["legacy"] else ""))


def _model(args, cal, bills):
    if args.model:
        return Catcher.from_json(args.model)
    return Catcher.prior(bills, cal["base_exception_rate"])


def cmd_simulate(args):
    cal = load_calibration()
    bills = billing_run(args.region, cal, args.n, args.seed, with_outcomes=args.with_outcomes)
    bills.to_csv(args.out, index=False)
    print(f"Wrote {len(bills):,} SYNTHETIC estimated bills for {args.region} -> {args.out}")


def cmd_score(args):
    cal = load_calibration()
    bills = pd.read_csv(args.bills)
    catcher = _model(args, cal, bills)
    scored = decide(bills, catcher, cal, args.mode)
    queue = scored[scored.action != SEND]
    queue.to_csv(args.out, index=False)
    s = summarize(scored, cal)
    print(f"Model: {catcher.source}   mode: {args.mode}")
    print(f"  bills scored          {s['bills']:,}")
    print(f"  self-read prompts     {s['self_read_prompts']:,}")
    print(f"  self-read, then visit {s['read_or_visit']:,}   (no real read in 12+ months)")
    print(f"  held                  {s['held_share']:.1%}")
    print(f"  expected bad bills    {s['expected_bad_bills']:,.0f}, expected stopped {s['expected_bad_stopped']:,.0f}")
    print(f"  expected net          {money(s['expected_net'])}  ({money(s['expected_saving'])} saved - {money(s['hold_cost'])} hold cost)")
    if "auc" in s:
        print(f"  ACTUAL: {s['bad_in_hold_queue']}/{s['actual_bad_bills']} bad bills in queue "
              f"(recall {s['recall_of_queue']:.0%}, precision {s['precision_of_queue']:.0%}, AUC {s['auc']:.2f})")
    print(f"Hold queue -> {args.out}")
    if args.show <= 0 or queue.empty:
        return
    cols = ["bill_id", "risk", "action", "reasons", "rule"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(queue[cols].head(args.show).to_string(index=False, formatters={"risk": "{:.0%}".format}))


def cmd_learn(args):
    cal = load_calibration()
    df = pd.read_csv(args.outcomes)
    if "needs_correction" not in df:
        raise SystemExit("outcomes file needs a needs_correction column (1 = bill was corrected)")
    prior = Catcher.prior(df, cal["base_exception_rate"])
    learned = prior.learn(df, df.needs_correction)
    learned.to_json(args.out)
    print(f"Wrote {args.out}: {learned.source}")
    print(f"  {'feature':<26}{'prior':>8}{'learned':>9}")
    for f in learned.coef:
        print(f"  {f:<26}{prior.coef[f]:>8.3f}{learned.coef[f]:>9.3f}")


def cmd_pilot(args):
    cal = load_calibration()
    table, _ = run_pilot(cal, args.regions, args.months)
    v = value_case(cal, table, args.regions)
    print(f"90-day pilot: {' + '.join(args.regions)}  (volumes, error rate, costs real; bills SYNTHETIC)\n")
    show = table.copy()
    for col in ["est_bills", "held", "field_visits", "bad_bills", "bad_stopped"]:
        show[col] = show[col].map("{:,.0f}".format)
    for col in ["hold_cost", "correction_saving", "wrong_billing_stopped", "net"]:
        show[col] = show[col].map(money)
    show["auc"] = show["auc"].map("{:.2f}".format)
    print(show.to_string(index=False))
    print("\n(month 1 is shadow mode: bad_stopped is what it WOULD have stopped; nothing held, no cost)\n")
    print(f"Steady state after pilot")
    print(f"  billing exceptions cut      {v['exceptions_cut']:.0%}")
    print(f"  net operating value         {money(v['annual_net_run_rate'])} / yr (after {money(cal['assumptions']['catcher_run_per_year'])} run cost)")
    print(f"  wrong billing stopped       {money(v['annual_wrong_billing_stopped'])} / yr of billing errors never sent to customers")
    print(f"  build cost (ASSUMPTION)     {money(v['build_cost'])}, payback {v['payback_months']:.1f} months")
    print(f"  vs smart meters             {v['accounts']:,} accounts x ${cal['unit_costs']['smart_meter']:.0f} = {money(v['smart_meter_cost'])}")


def main():
    p = argparse.ArgumentParser(prog="bill_catcher", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("calibrate")
    s.add_argument("--data", default=DEFAULT_DATA, help="folder with the Northwind CSVs")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("simulate")
    s.add_argument("--region", default="Barrowdale")
    s.add_argument("--n", type=int, default=20_000)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--with-outcomes", action="store_true", help="include needs_correction (for learn / evaluation)")
    s.add_argument("-o", "--out", default="billing_run.csv")
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("score")
    s.add_argument("bills")
    s.add_argument("--model", help="model.json from `learn` (default: prior weights)")
    s.add_argument("--mode", choices=["hold", "shadow"], default="hold")
    s.add_argument("-o", "--out", default="hold_queue.csv")
    s.add_argument("--show", type=int, default=10)
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("learn")
    s.add_argument("outcomes")
    s.add_argument("-o", "--out", default="model.json")
    s.set_defaults(fn=cmd_learn)

    s = sub.add_parser("pilot")
    s.add_argument("--regions", nargs="+", default=PILOT_REGIONS)
    s.add_argument("--months", type=int, default=3)
    s.set_defaults(fn=cmd_pilot)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
