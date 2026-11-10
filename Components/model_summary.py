"""Compact, research-grounded model-selection explanation for IDS 560.

Call render_model_summary() inside the existing Model Insights tab.
All performance numbers are fixed historical 2016 holdout results; they are
NOT live performance metrics or statistics for the selected store.
"""
from __future__ import annotations

import streamlit as st


def render_model_summary() -> None:
    """Explain the model choice without replacing detailed validation charts."""
    st.markdown(
        "**Why LightGBM?** Our decision-support system needs a demand forecast "
        "before calculating inventory replenishment. Demand is difficult to predict "
        "because approximately **68% of item–store–days had zero sales** in the "
        "M5 data used for this project."
    )

    with st.container(border=True):
        st.markdown("##### Model selection at a glance")
        a, b, c = st.columns(3, gap="small")
        a.metric(
            "LightGBM · 7-day WAPE", "71.62%",
            help="Historical 2016 holdout; lower WAPE is better.",
        )
        b.metric(
            "Recent 7-day mean · WAPE", "74.58%",
            help="Historical 2016 holdout baseline on the same evaluation.",
        )
        c.metric(
            "Error reduction", "2.96 pp",
            help="74.58% − 71.62% = 2.96 percentage points; not a relative percentage reduction.",
        )
        st.markdown(
            "LightGBM achieved **2.96 percentage points lower WAPE** than the "
            "seven-day Recent Mean baseline. Its signed forecast bias was also "
            "closer to zero (**−0.70%** versus **−1.75%**)."
        )
        st.caption(
            "2016 held-out evaluation · 50% item–store sample · Aggregate model results, "
            "not accuracy for the selected store or live forecasts."
        )

    with st.expander("How does this support the inventory decision?", expanded=False):
        st.markdown(
            "The model predicts expected demand; the replenishment policy then "
            "combines demand with **assumed starting stock, safety stock, and review "
            "flags**. The dashboard does not have actual Walmart on-hand inventory "
            "and does not place purchase orders."
        )
        st.markdown(
            "**Why not simply use the lowest error at all costs?** Forecast error "
            "is only one consideration. Systematic bias, uncertainty, and managerial "
            "review matter when a forecast informs an inventory decision."
        )
        st.caption(
            "See the detailed comparisons and horizon results below. Sources: "
            "project's final LightGBM forecasting notebook and research poster."
        )
