# Northwind — Root Cause & Recovery (Hack the Hill III · CGI track)

Streamlit dashboard that argues the client's brief (an AI assistant to triage complaints) targets the symptom, not the cause.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tabs
1. **Diagnosis**: joins `meter_reads` to `complaints` at region-month level. Estimated reads drive complaints (r ≈ 0.79); Barrowdale and Dunmoor (0% smart meters, 1998 billing, 2012 estimation algorithm) estimate about 61% of bills and raise 2.8× more billing exceptions per account. Transferred complaints take 38 vs 23 days and reopen 27% vs 8%. The 2025 AI pilot got worse every month.
2. **Bill Catcher**: a logistic-regression model scores estimated bills *before they are sent* and holds the riskiest ones for a self-read or a field visit. The run is **simulated** (the data pack has no bill-level records), calibrated to the real exception rate of ≈4.1% of estimated bills.
3. **Forecast & Value**: a 12-month backlog and regulator-score simulation with toggles for each intervention, plus the cost, savings, penalty avoidance and payback. Every assumption is listed and editable.

## Model notes
- Regulator score ≈ 5.32 − 0.069 × avg days to close (fitted on 24 months of KPIs), moving 10% of the way toward that level each month (a lag, which is an assumption).
- Days to close is derived from backlog ÷ closures (Little's law), calibrated to 38.2 days, with a floor of 9.1 days.
- Baseline volumes follow the linear trend of the last 12 months.
- All unit costs come from `northwind_unit_costs.csv`. Build and run costs for the new systems are labelled ASSUMPTION.
