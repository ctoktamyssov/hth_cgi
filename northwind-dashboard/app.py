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

# =============================================================================
# SIDEBAR INTERVENTIONS & ASSUMPTIONS
# =============================================================================
with st.sidebar:
    st.header("Interventions")
    st.caption("Toggle and tune levers live.")
    catcher_on = st.checkbox("① Bill Catcher (live month 4)", True)
    c_hold = st.slider("Share of estimated bills held", 0.01, 0.30, 0.05, 0.01, disabled=not catcher_on)
    rc_all = M.recall_curve(M.train_catcher(0.35, 0.041)[1])
    c_recall = float(np.interp(c_hold, rc_all.share_held, rc_all.recall))
    st.caption(f"→ catches ~{c_recall:.0%} of bad bills company-wide")
    
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

# =============================================================================
# TOP METRICS
# =============================================================================
top = st.columns(4)
top[0].metric("Open complaints", f"{int(k.backlog.iloc[-1]):,}", f"+{int(k.backlog.iloc[-1] - k.backlog.iloc[-13]):,} in 12 months", delta_color="inverse")
top[1].metric("Avg days to close", f"{k.avg_days_to_close.iloc[-1]:.1f}", f"from {k.avg_days_to_close.iloc[0]:.1f}", delta_color="off")
top[2].metric("Regulator score", f"{k.regulator_satisfaction_score_of_5.iloc[-1]:.2f} / 5", f"from {k.regulator_satisfaction_score_of_5.iloc[0]:.1f}", delta_color="off")
top[3].metric("Billing exceptions (12 mo)", f"{int(meters[meters.month.isin(sorted(meters.month.unique())[-12:])].billing_exceptions_raised.sum()):,}")

tab1, tab2 = st.tabs(["1 · Diagnosis", "2 · Bill Catcher (live demo)"])

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
    hold_share = col2.slider("Hold the riskiest X% of estimated bills", 0.01, 0.30, c_hold, 0.01, format="%.2f")
    visit_share = col3.slider("Held bills needing a field visit (rest use a customer self-read prompt)",
                              0.0, 1.0, A["field_visit_fallback"], 0.05)
    reg_est = float(rt.set_index("region").loc[region, "estimated_read_rate"])
    _, scored, auc = M.train_catcher(reg_est, exc_per_est)

    cost_visit = M.unit_cost(d, "Field meter visit")
    cost_hold = A["self_read_cost"] + visit_share * cost_visit
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
        f"**Economics per bill** — holding costs {money(cost_hold)} (self-read prompt ${A['self_read_cost']:.0f} "
        f"ASSUMPTION + {visit_share:.0%} × ${cost_visit:.0f} field visit). Each bad bill caught avoids a ${cost_corr:.0f} manual "
        f"correction plus a {conv:.1%} chance of a ~${cost_compl:.0f} complaint. **Holding too many bills loses money** — "
        "push the slider to 30% to see it. Field visits at $92 only pay for the very highest-risk bills.")