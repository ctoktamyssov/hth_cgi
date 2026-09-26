import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import model as M

st.set_page_config(page_title="Northwind — Root Cause & Recovery", layout="wide")

BLUE, ORANGE, GREY, RED, GREEN = "#2E6BD6", "#E8762B", "#9AA3AF", "#C9372C", "#2F9E5B"
TEMPLATE = "plotly_white"


@st.cache_data
def data():
    return M.load()


d = data()
c, k, meters, pilot = d["complaints"], d["kpis"], d["meters"], d["pilot"]
money = lambda x: f"${x/1e6:,.2f}M" if abs(x) >= 1e6 else f"${x:,.0f}"

st.title("Northwind Utilities — the complaints are made upstream")
st.caption("CGI · The Northwind Brief · all figures from the synthetic data pack unless marked ASSUMPTION")

top = st.columns(4)
top[0].metric("Open complaints", f"{int(k.backlog.iloc[-1]):,}", f"+{int(k.backlog.iloc[-1] - k.backlog.iloc[-13]):,} in 12 months", delta_color="inverse")
top[1].metric("Avg days to close", f"{k.avg_days_to_close.iloc[-1]:.1f}", f"from {k.avg_days_to_close.iloc[0]:.1f}", delta_color="off")
top[2].metric("Regulator score", f"{k.regulator_satisfaction_score_of_5.iloc[-1]:.2f} / 5", f"from {k.regulator_satisfaction_score_of_5.iloc[0]:.1f}", delta_color="off")
top[3].metric("Billing exceptions (12 mo)", f"{int(meters[meters.month.isin(sorted(meters.month.unique())[-12:])].billing_exceptions_raised.sum()):,}")

tab1, tab2, tab3 = st.tabs(["1 · Diagnosis", "2 · Bill Catcher (live demo)", "3 · Forecast & Value"])

# =============================================================================
# TAB 1 — DIAGNOSIS
# =============================================================================
with tab1:
    st.subheader("Finding 1 — Estimated meter reads manufacture complaints")
    rt = M.region_table(d)
    rm = M.region_month(d)
    rm["legacy"] = rm.legacy.map({True: "Barrowdale / Dunmoor (no smart meters)", False: "Other regions"})
    slope, intercept, r = M.fit_line(rm.estimated_read_rate, rm.meter_complaints_per_10k)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.scatter(rm, x="estimated_read_rate", y="meter_complaints_per_10k", color="legacy",
                         color_discrete_map={"Barrowdale / Dunmoor (no smart meters)": ORANGE, "Other regions": BLUE}, hover_data=["region", "month"],
                         labels={"estimated_read_rate": "Share of bills based on an estimated read",
                                 "meter_complaints_per_10k": "Meter/estimate complaints per 10k accounts",
                                 "legacy": ""},
                         template=TEMPLATE, height=380)
        xs = np.linspace(rm.estimated_read_rate.min(), rm.estimated_read_rate.max(), 20)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", line=dict(color=GREY, dash="dash"),
                                 name=f"fit, r = {r:.2f}"))
        fig.update_layout(legend=dict(orientation="h", y=-0.25), margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')
        st.caption("Each dot = one region in one month. Joins northwind_meter_reads.csv to northwind_complaints.csv.")
    with col2:
        show = rt[["region", "systems", "smart_meter_penetration", "estimated_read_rate", "exceptions_per_10k", "meter_share"]].copy()
        show.columns = ["Region", "Systems", "Smart meters", "Estimated bills", "Exceptions / 10k acc / mo", "Complaints that are meter-driven"]
        st.dataframe(show.style.format({"Smart meters": "{:.0%}", "Estimated bills": "{:.0%}",
                                        "Exceptions / 10k acc / mo": "{:.0f}", "Complaints that are meter-driven": "{:.0%}"}),
                     hide_index=True, width='stretch')
        leg = rt[rt.region.isin(M.LEGACY_REGIONS)]
        oth = rt[~rt.region.isin(M.LEGACY_REGIONS)]
        st.markdown(
            f"**Barrowdale & Dunmoor** run on the 1998 COBOL billing system and MeterHub, whose estimation "
            f"algorithm is *unchanged since 2012 with no feedback loop from corrected bills*. They estimate "
            f"**{leg.estimated_read_rate.mean():.0%}** of bills (vs {oth.estimated_read_rate.mean():.0%}) and raise "
            f"**{leg.exceptions_per_10k.mean() / oth.exceptions_per_10k.mean():.1f}×** more billing exceptions per account.")

    st.divider()
    st.subheader("Finding 2 — Handoffs between systems make every complaint slower and dearer")
    tt = M.transfer_table(d)
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(tt.style.format({"complaints": "{:,}", "avg_days": "{:.1f}", "sla_breach": "{:.0%}",
                                      "reopened": "{:.0%}", "unit_cost": "${:.0f}"}), width='stretch')
        st.markdown(f"**{c.transferred_between_systems.mean():.0%}** of complaints are transferred. CaseTrack "
                    "*loses the history* of transferred cases and three core systems still sync by *nightly batch file*.")
    with col2:
        st_ = M.source_transfer_table(d)
        fig = px.bar(st_, x="system", y="transferred", text=st_.transferred.map("{:.0%}".format),
                     labels={"transferred": "Share transferred", "system": "Where the complaint was opened"},
                     template=TEMPLATE, height=300, color_discrete_sequence=[BLUE])
        fig.update_layout(yaxis_tickformat=".0%", margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')
        st.caption("Complaints opened in CaseTrack are never transferred; ones opened anywhere else are moved almost half the time.")

    st.divider()
    st.subheader("Finding 3 — The 2025 AI assistant got worse every month")
    col1, col2 = st.columns([3, 2])
    with col1:
        p = pilot.melt(id_vars="month", value_vars=["fully_contained_rate", "repeat_contact_within_7_days_rate",
                                                     "complaint_raised_after_session_rate"])
        p["variable"] = p.variable.map({"fully_contained_rate": "Resolved without an agent",
                                        "repeat_contact_within_7_days_rate": "Customer came back within 7 days",
                                        "complaint_raised_after_session_rate": "Complaint raised after session"})
        fig = px.line(p, x="month", y="value", color="variable", markers=True, template=TEMPLATE, height=320,
                      color_discrete_sequence=[BLUE, ORANGE, RED], labels={"value": "", "month": "", "variable": ""})
        fig.update_layout(yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.25), margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')
    with col2:
        ct = M.category_table(d)
        st.markdown(
            f"CSAT fell **{pilot.assistant_csat_of_5.iloc[0]} → {pilot.assistant_csat_of_5.iloc[-1]}**. It cost "
            f"{money(M.unit_cost(d, 'AskNorthwind'))}/yr.\n\n"
            f"Only **{c.resolvable_by_information_only.mean():.0%}** of complaints can be closed with information alone. "
            f"The other {1 - c.resolvable_by_information_only.mean():.0%} need a corrected bill, a meter visit or a repair — "
            "things a chatbot cannot do. **Automating the answer does not remove the cause.**")
        st.dataframe(ct[["category", "share", "info_only"]].style.format({"share": "{:.0%}", "info_only": "{:.0%}"}),
                     hide_index=True, width='stretch', height=250)

# =============================================================================
# TAB 2 — BILL CATCHER
# =============================================================================
with tab2:
    st.subheader("Catch bad estimated bills before they are sent")
    st.info("**Simulated billing run.** The data pack has no bill-level records, so this demo generates estimated "
            "bills whose error rate is calibrated to Northwind's real figure (billing exceptions ≈ 4.1% of estimated "
            "bills in every region). In production the model trains on MeterHub history + corrected bills — "
            "the feedback loop MeterHub has never had. This proves the pipeline, not the real-world accuracy.")

    exc_per_est = float((meters.billing_exceptions_raised / (meters.accounts * meters.estimated_read_rate)).mean())
    est_compl = (c.category == "Billing - estimated read").sum() / c.month.nunique()
    conv = est_compl / meters.groupby("month").billing_exceptions_raised.sum().mean()

    col1, col2, col3 = st.columns(3)
    region = col1.selectbox("Billing run for region", rt.region.tolist(), index=0)
    hold_share = col2.slider("Hold the riskiest X% of estimated bills", 0.01, 0.30, 0.05, 0.01, format="%.2f")
    visit_share = col3.slider("Held bills needing a field visit (rest use a customer self-read prompt)",
                              0.0, 1.0, M.DEFAULT_ASSUMPTIONS["field_visit_fallback"], 0.05)
    reg_est = float(rt.set_index("region").loc[region, "estimated_read_rate"])
    _, scored, auc = M.train_catcher(reg_est, exc_per_est)

    cost_visit = M.unit_cost(d, "Field meter visit")
    cost_hold = M.DEFAULT_ASSUMPTIONS["self_read_cost"] + visit_share * cost_visit
    cost_corr = M.unit_cost(d, "Manual bill correction")
    cost_compl = 0.349 * M.unit_cost(d, "Complaint handled end to end (transferred") + 0.651 * M.unit_cost(d, "Complaint handled end to end (average)")
    econ = M.threshold_economics(scored, hold_share, cost_hold, cost_corr, conv, cost_compl)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bills held for a real read", f"{econ['n_hold']:,} of {len(scored):,}")
    m2.metric("Bad bills caught (recall)", f"{econ['caught']} / {econ['total_bad']}", f"{econ['recall']:.0%}", delta_color="off")
    m3.metric("Held bills that were bad (precision)", f"{econ['precision']:.0%}")
    m4.metric("Net value this run", money(econ["net"]), f"{money(econ['benefit'])} saved − {money(econ['cost'])} cost", delta_color="off")

    col1, col2 = st.columns([2, 3])
    with col1:
        rc = M.recall_curve(scored)
        fig = px.area(rc, x="share_held", y="recall", template=TEMPLATE, height=320,
                      labels={"share_held": "Share of estimated bills held", "recall": "Share of bad bills caught"},
                      color_discrete_sequence=[BLUE])
        fig.add_vline(x=hold_share, line_dash="dash", line_color=ORANGE)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dot"), name="random"))
        fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Model: logistic regression · AUC {auc:.2f} on an unseen billing run · explainable reason codes per bill.")
    with col2:
        st.markdown("**Hold queue — sent to a self-read prompt or a field visit instead of the customer's letterbox**")
        q = scored.head(econ["n_hold"])[["bill_id", "risk", "months_since_actual_read", "deviation_vs_last_year_pct", "why"]]
        q.columns = ["Bill", "Risk", "Months since real read", "Deviation vs last year %", "Why flagged"]
        st.dataframe(q.head(200).style.format({"Risk": "{:.0%}", "Deviation vs last year %": "{:.0f}"}), hide_index=True, width='stretch', height=320)

    st.markdown(
        f"**Economics per bill** — holding costs {money(cost_hold)} (self-read prompt ${M.DEFAULT_ASSUMPTIONS['self_read_cost']:.0f} "
        f"ASSUMPTION + {visit_share:.0%} × ${cost_visit:.0f} field visit). Each bad bill caught avoids a ${cost_corr:.0f} manual "
        f"correction plus a {conv:.1%} chance of a ~${cost_compl:.0f} complaint. **Holding too many bills loses money** — "
        "push the slider to 30% to see it. Field visits at $92 only pay for the very highest-risk bills.")

# =============================================================================
# TAB 3 — FORECAST & VALUE
# =============================================================================
with tab3:
    st.subheader("Can Northwind reach a regulator score of 4.0 in 12 months?")
    with st.sidebar:
        st.header("Interventions")
        st.caption("Toggle and tune — the forecast and value case update live.")
        catcher_on = st.checkbox("① Bill Catcher (live month 4)", True)
        c_hold = st.slider("Share of estimated bills held", 0.01, 0.30, 0.05, 0.01, disabled=not catcher_on)
        rc_all = M.recall_curve(M.train_catcher(0.35, 0.041)[1])
        c_recall = float(np.interp(c_hold, rc_all.share_held, rc_all.recall))
        st.caption(f"→ catches ~{c_recall:.0%} of bad bills company-wide (tab 2 model)")
        case_on = st.checkbox("② Unified case history (integration layer, live month 5)", True)
        x_cut = st.slider("Cut in cross-system transfers", 0.0, 0.9, 0.6, 0.05, disabled=not case_on)
        surge = st.slider("Temporary backlog team (agents)", 0, 40, 12)
        surge_m = st.slider("…for how many months", 0, 12, 4)
        smart = st.slider("③ Smart meters: share of Barrowdale+Dunmoor fitted this year", 0.0, 1.0, 0.0, 0.05)
        ai_on = st.checkbox("④ Relaunch AI assistant (client's original brief)", False)
        ai_c = st.slider("AI containment of info-only complaints", 0.0, 0.5, 0.12, 0.01, disabled=not ai_on)
        st.divider()
        st.header("ASSUMPTIONS")
        A = dict(M.DEFAULT_ASSUMPTIONS)
        A["catcher_build"] = st.number_input("Bill Catcher build $", value=A["catcher_build"], step=50_000)
        A["catcher_run"] = st.number_input("Bill Catcher run $/yr", value=A["catcher_run"], step=25_000)
        A["case_layer_build"] = st.number_input("Integration layer build $", value=A["case_layer_build"], step=100_000)
        A["case_layer_run"] = st.number_input("Integration layer run $/yr", value=A["case_layer_run"], step=50_000)
        A["regulator_breach_threshold"] = st.number_input("Penalty applies below score", value=A["regulator_breach_threshold"], step=0.1)
        A["field_visit_fallback"] = st.slider("Held bills needing field visit", 0.0, 1.0, A["field_visit_fallback"], 0.05)

    levers = dict(catcher_on=catcher_on, catcher_recall=c_recall, catcher_hold_share=c_hold, case_layer_on=case_on,
                  transfer_cut=x_cut, smart_meter_share=smart, ai_on=ai_on, ai_containment=ai_c,
                  surge_fte=surge, surge_months=surge_m)
    off = dict(levers, catcher_on=False, case_layer_on=False, smart_meter_share=0, ai_on=False, surge_fte=0, surge_months=0)
    ai_only = dict(off, ai_on=True, ai_containment=0.12)
    f = M.forecast(d, levers, A)
    fb = M.forecast(d, off, A)
    fa = M.forecast(d, ai_only, A)

    hist = k[["month", "backlog", "regulator_satisfaction_score_of_5"]].rename(columns={"regulator_satisfaction_score_of_5": "score"})
    fut_months = pd.period_range(pd.Period(k.month.iloc[-1]) + 1, periods=12, freq="M").strftime("%Y-%m")

    last = hist.iloc[-1]

    def series(frame, name):  # anchor each scenario on the last actual month so lines connect
        return pd.DataFrame({"month": [last.month, *fut_months], "backlog": [last.backlog, *frame.backlog.values],
                             "score": [last.score, *frame.score.values], "scenario": name})

    plot = pd.concat([hist.assign(scenario="Actual"), series(fb, "Do nothing"), series(fa, "Client brief: AI assistant only"),
                      series(f, "Our plan")])
    cmap = {"Actual": "#222", "Do nothing": GREY, "Client brief: AI assistant only": RED, "Our plan": GREEN}
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(plot, x="month", y="score", color="scenario", color_discrete_map=cmap, template=TEMPLATE, height=360,
                      labels={"score": "Regulator score / 5", "month": ""})
        fig.add_hline(y=4.0, line_dash="dash", line_color=GREEN, annotation_text="target 4.0")
        fig.update_layout(legend=dict(orientation="h", y=-0.2), margin=dict(t=10), yaxis_range=[1, 5])
        st.plotly_chart(fig, width='stretch')
    with col2:
        fig = px.line(plot, x="month", y="backlog", color="scenario", color_discrete_map=cmap, template=TEMPLATE, height=360,
                      labels={"backlog": "Open complaints", "month": ""})
        fig.update_layout(legend=dict(orientation="h", y=-0.2), margin=dict(t=10))
        st.plotly_chart(fig, width='stretch')
    hit = f[f.score >= 4.0]
    st.markdown(
        f"**Our plan:** score **{f.score.iloc[-1]:.2f}** at month 12 "
        + (f"(crosses 4.0 in month {int(hit.month.iloc[0])})" if len(hit) else "(**does not reach 4.0**)")
        + f" · backlog {int(f.backlog.iloc[-1]):,} · avg {f.days.iloc[-1]:.0f} days to close. "
        f"**AI-only brief:** {fa.score.iloc[-1]:.2f}. **Do nothing:** {fb.score.iloc[-1]:.2f}.")

    st.divider()
    st.subheader("Value case — 12 months")
    pen_q = f.attrs["penalty_per_quarter"]
    pen_avoided = (fb.attrs["quarters_in_breach"] - f.attrs["quarters_in_breach"]) * pen_q
    cost = f.cost.sum() - fb.cost.sum()
    handling = f.handling_saved.sum()
    correction = f.correction_saved.sum()
    benefit = handling + correction + pen_avoided
    v = st.columns(5)
    v[0].metric("Cost (build + run + holds)", money(cost))
    v[1].metric("Complaint handling saved", money(handling))
    v[2].metric("Bill corrections avoided", money(correction))
    v[3].metric("Regulator penalties avoided", money(pen_avoided),
                f"{fb.attrs['quarters_in_breach']}→{f.attrs['quarters_in_breach']} quarters in breach", delta_color="off")
    v[4].metric("Net, year 1", money(benefit - cost))

    f["cum_net"] = (f.handling_saved + f.correction_saved - (f.cost - fb.cost)).cumsum()
    q_pen = []
    for q in range(4):
        b_breach = fb.score.iloc[q * 3 + 2] < A["regulator_breach_threshold"]
        p_breach = f.score.iloc[q * 3 + 2] < A["regulator_breach_threshold"]
        q_pen += [0, 0, pen_q if (b_breach and not p_breach) else 0]
    f["cum_net_pen"] = f.cum_net + np.cumsum(q_pen)
    fig = go.Figure()
    fig.add_bar(x=fut_months, y=f.cum_net_pen, marker_color=[GREEN if v >= 0 else RED for v in f.cum_net_pen], name="incl. penalties")
    fig.add_scatter(x=fut_months, y=f.cum_net, mode="lines+markers", line=dict(color=GREY), name="operating savings only")
    fig.update_layout(template=TEMPLATE, height=300, yaxis_title="Cumulative net $", margin=dict(t=10),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width='stretch')
    pay = f[f.cum_net_pen >= 0]
    st.markdown(f"**Payback:** " + (f"month {int(pay.month.iloc[0])} including avoided penalties" if len(pay) else "not within 12 months")
                + " · operating savings alone: " + money(f.cum_net.iloc[-1]) + " cumulative at month 12.")

    with st.expander("What this plan does NOT fix — and the biggest risks"):
        st.markdown("""
- **Smart meters are the permanent fix** for Barrowdale & Dunmoor (≈519k accounts × $148 ≈ $77M). The Bill Catcher buys time; it does not remove estimation.
- **Aurora Billing (1998 COBOL)** has two developers who understand the rating engine; the Bill Catcher must sit *beside* it, reading its batch output, not modify it.
- **Helix CIS vendor support ends in 18 months** — the integration layer should be designed to survive that migration.
- **Regulator-score model** is a straight-line fit of score vs days-to-close (r ≈ −0.99) with a lag; the real regulator may weigh other things. *Ask the COO.*
- **Biggest risk:** the model's real accuracy is unknown until trained on MeterHub history. Mitigation: shadow-mode for 4 weeks — flag but don't hold — and measure precision before holding any bill.
""")
    with st.expander("All assumptions and their sources"):
        rows = [
            ("Complaint handling cost", "$68 / $121 transferred", "northwind_unit_costs.csv"),
            ("Manual bill correction", "$34", "northwind_unit_costs.csv"),
            ("Field meter visit", "$92", "northwind_unit_costs.csv"),
            ("Smart meter install", "$148", "northwind_unit_costs.csv"),
            ("Regulator penalty", "$2.4M per quarter in breach", "northwind_unit_costs.csv"),
            ("Each billing exception = one manual correction", "1:1", "ASSUMPTION — confirm with COO"),
            ("Penalty applies when score below", f"{A['regulator_breach_threshold']}", "ASSUMPTION — confirm with COO"),
            ("Score vs days-to-close", "score = 5.32 − 0.069 × days", "Fitted on northwind_monthly_kpis.csv"),
            ("Days to close vs backlog", "Little's law, calibrated to 38.2 days, floor 9.1", "Fitted on KPIs"),
            ("Baseline volumes", "Linear trend of last 12 months", "northwind_monthly_kpis.csv"),
            ("Extra agent throughput", f"{f.attrs['per_fte_month']:.0f} complaints / month", "$46k FTE ÷ blended complaint cost"),
            ("Bill Catcher build / run", f"{money(A['catcher_build'])} / {money(A['catcher_run'])} per yr", "ASSUMPTION — ~12 weeks, 3 devs + analyst"),
            ("Integration layer build / run", f"{money(A['case_layer_build'])} / {money(A['case_layer_run'])} per yr", "ASSUMPTION"),
            ("Self-read prompt cost", f"${A['self_read_cost']:.0f} per held bill", "ASSUMPTION"),
            ("Smart meters cut estimated-read rate", "0.60 → 0.20", "Observed in smart-meter regions"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Item", "Value", "Source"]), hide_index=True, width='stretch')
