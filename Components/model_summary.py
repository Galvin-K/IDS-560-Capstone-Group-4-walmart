"""Concise introduction and model-selection overview for the IDS 560 dashboard.

Call render_model_summary() once, at the beginning of Model Insights.
All results are historical 2016 holdout evaluation, not live accuracy.
"""
from __future__ import annotations

import streamlit as st


def render_model_summary() -> None:
    st.subheader("1. Understanding the Forecasting Models")
    st.write(
        "The system forecasts demand for individual products and stores before calculating "
        "inventory needs. This is challenging: approximately **68% of item–store–days "
        "recorded no sales** in the M5 dataset."
    )

    recent, lightgbm = st.columns(2, gap="medium")
    with recent:
        with st.container(border=True):
            st.markdown("**Recent Mean · Baseline**")
            st.write(
                "Uses average sales over a recent period to estimate future demand. "
                "It provides a simple benchmark for a more advanced model."
            )
    with lightgbm:
        with st.container(border=True):
            st.markdown("**LightGBM · Selected model**")
            st.write(
                "Uses gradient-boosted decision trees to learn patterns involving recent "
                "sales, price, holidays, SNAP days, and other calendar information."
            )

    st.subheader("2. Why We Selected LightGBM")
    st.caption("2016 historical holdout · Seven-day forecast horizon · 50% item–store sample")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3, gap="small")
        c1.metric("LightGBM WAPE", "71.62%", help="Weighted Absolute Percentage Error: lower is better.")
        c2.metric("Recent Mean WAPE", "74.58%", help="The recent seven-day average baseline.")
        c3.metric("Lower error", "2.96 pp", help="74.58% − 71.62% = 2.96 percentage points.")
        st.write(
            "**LightGBM had 2.96 percentage points lower WAPE** than the recent seven-day "
            "mean baseline. Its signed bias was closer to zero (**−0.70%** versus **−1.75%**)."
        )
        st.caption(
            "WAPE is a measure of forecast error, **not model accuracy**; lower is better. "
            "These are aggregate historical results, not accuracy for the selected store."
        )

    st.markdown("#### 3. How to Interpret the Results")
    st.markdown(
        "**WAPE:** Total absolute forecasting error divided by total actual sales. "
        "For example, 71.62% WAPE does **not** mean 71.62% accuracy.  \n"
        "**Bias:** Shows overall overforecasting (+) or underforecasting (−); values "
        "closer to zero indicate less systematic error.  \n"
        "**RMSE:** Forecast error in units that penalizes large misses more heavily.  \n"
        "**P10–P90 coverage:** Percentage of observed daily sales within the predicted "
        "lower and upper ranges (target: 80%)."
    )
    with st.expander("How does this support inventory decisions?", expanded=False):
        st.write(
            "The selected forecast feeds a replenishment policy that uses **assumed starting "
            "inventory, safety stock, and review flags**. The dashboard does not have actual "
            "Walmart on-hand inventory, and it does not place purchase orders."
        )
        st.write(
            "A lower forecasting error alone does not guarantee better replenishment. "
            "Bias, uncertainty, and manager review also matter."
        )
    st.divider()
