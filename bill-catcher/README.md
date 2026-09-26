# Bill Catcher: estimated-read exception catcher (Northwind, CGI track)

This tool scores every **estimated** bill before it is sent. Risky bills go to a customer self-read prompt instead of the letterbox. It is the feedback loop MeterHub (SYS-06) has never had: its estimation algorithm is "unchanged since 2012. No feedback loop from corrected bills."

It is the first 90 days of the fix for Barrowdale and Dunmoor, while the smart meter rollout (519k accounts × $148 ≈ **$76.8M**) is phased in.

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m bill_catcher calibrate                     # rates, volumes, costs from ../northwind-dashboard/data
python -m bill_catcher pilot                         # 90-day Barrowdale + Dunmoor pilot + value case
python -m bill_catcher simulate --region Barrowdale --with-outcomes -o run.csv
python -m bill_catcher score run.csv                 # hold queue with risk, reasons and action per bill
python -m bill_catcher learn run.csv                 # refit on corrected-bill outcomes -> model.json
python -m bill_catcher score run.csv --model model.json --mode shadow
```

## How it decides
Input: one row per estimated bill, the fields MeterHub/Aurora's nightly batch already holds:
`bill_id, account_id, region, months_since_actual_read, estimated_kwh, last_year_kwh, prior_corrections_12m, outage_in_period, crosses_season`.

1. **Policy rules** (always apply):
   - no real read in 12+ months → self-read prompt, then a field visit if the customer doesn't respond
   - an outage in the bill period (GridWatch; *"outage data is not used to suppress related billing chasers"*) → self-read prompt
   - an estimate more than 60% off last year's usage → self-read prompt
2. **Risk score**: a logistic model with reason codes per bill. A bill is held when `risk × response rate × $34.72 saving > $2 prompt cost`.
3. **Feedback loop**: `learn` refits the weights on corrected-bill outcomes. `--mode shadow` flags bills without holding anything, so precision can be measured before any bill is held.

## What comes from the data pack vs what doesn't
| Real (from the data pack) | Not in the pack |
|---|---|
| 4.10% of estimated bills become billing exceptions, **flat in every region and month** | Bill-level records: none exist, so bills are **simulated** (`simulate.py`) |
| Estimated bills per month per region (B+D ≈ 319k/month) | Day-one weights (`PRIOR_COEF`): expert judgement |
| Avg correction on estimated-read complaints: **$220 in B+D vs $147 elsewhere** | $2 self-read prompt, 70% response rate |
| 0.83% of exceptions become an estimated-read complaint | $450k build, $150k/yr run (same as the dashboard) |
| $34 correction, $92 field visit, $68/$121 complaint, $148 smart meter | The simulator's hidden "true" weights, set to differ from the priors so the feedback loop has something to learn |

Because the error rate is flat, **region and season don't say which bill is wrong**. Barrowdale and Dunmoor are worse because they estimate 3× more bills, and their errors are about 50% bigger. That's why the model works on bill-level features, not region.

## Pilot result (synthetic bills at real volumes)
- Month 1 shadow, months 2–3 live. The hold queue is about 14% of estimated bills, with a precision of about 19% vs a 4.1% base rate (4.6× lift), AUC about 0.84.
- It **stops about 45% of billing exceptions** in B+D, about $16M/yr of billing errors never sent to customers.
- **Operating savings alone are thin:** about $0.24M/yr net, with payback in about 22 months. Most bills are cheap to correct ($34), and the 12-month field-visit rule costs about $90k/month.
- **The case rests on customers and the regulator:** about 6k fewer wrong bills a month in the two regions driving the complaint growth.

## Pitch framing / Q&A
- *"Is the model accurate?"* Real accuracy is unknown until it is trained on MeterHub history. That's why month 1 is shadow mode: we measure precision before holding a single bill.
- *"Why not just fit smart meters?"* We should, but that costs $77M and has been deferred twice. This costs $450k, sits **beside** Aurora (it reads the batch output and doesn't change the COBOL), and starts learning in week one.
- The penalty/complaint knock-on effect is modelled in the dashboard's Forecast tab, not here.
