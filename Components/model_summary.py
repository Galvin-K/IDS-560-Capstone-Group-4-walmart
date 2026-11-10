"""Plain-language model and metric guide for IDS 560 Model Insights.

Use render_model_summary() at the top of the existing Model Insights tab.
Figures are from the historical 2016 holdout, not the selected store.
"""
from __future__ import annotations

import streamlit as st


def render_model_summary() -> None:
    """Introduce methods and evaluation terms BEFORE showing results."""
    st.subheader("Understanding the Forecasting Models")
    st.write(
        "The goal is to predict demand for each product and store before calculating "
        "replenishment needs. This is challenging because approximately **68% of "
        "item–store–days had zero sales** in the M5 dataset."
    )

    recent, lgbm = st.columns(2, gap="medium")
    with recent:
        with st.container(border=True):
            st.markdown("**Recent Mean · Baseline**")
            st.write(
                "Uses average sales from a recent period to predict upcoming demand. "
                "It's a simple reference point for judging whether more complex models help."
            )
    with lgbm:
        with st.container(border=True):
            st.markdown("**LightGBM · Selected model**")
            st.write(
                "A gradient-boosted tree model that learns patterns from recent sales, "
                "prices, calendar factors, holidays, and SNAP days."
            )

    st.divider()
    st.subheader("Why We Selected LightGBM")
    with st.container(border=True):
        a, b, c = st.columns(3, gap="small")
        a.metric(
            "LightGBM · 7-day WAPE", "71.62%",
            help="2016 holdout; aggregate WAPE. Lower is better.",
        )
        b.metric(
            "Recent 7-day Mean · WAPE", "74.58%",
            help="Baseline on the same 2016 historical evaluation.",
        )
        c.metric(
            "Lower WAPE", "2.96 pp",
            help="74.58 − 71.62 = 2.96 percentage points; not percent relative improvement.",
        )
        st.write(
            "LightGBM had **2.96 percentage points lower WAPE** than the recent "
            "7-day mean baseline. Its signed bias was also closer to zero "
            "(**−0.70%** versus **−1.75%**)."
        )
        st.caption(
            "Historical 2016 holdout · 50% item–store sample · Aggregate evaluation, "
            "not accuracy for the selected store or live forecasts."
        )
    st.markdown("##### How to read the evaluation results")
    st.markdown(
        "**WAPE (Weighted Absolute Percentage Error):** Total absolute forecast error "
        "divided by total actual sales. **Lower is better.** For example, 71.62% WAPE "
        "does **not** mean 71.62% accuracy.\n\n"
        "**Bias:** Whether the model tends to overpredict (+) or underpredict (−) "
        "sales. Values closer to zero indicate less overall systematic error.\n\n"
        "**RMSE (Root Mean Squared Error):** Forecast error measured in units, with "
        "larger mistakes receiving more weight. **Lower is better.**\n\n"
        "**P10–P90 coverage:** The percentage of observed daily sales inside the "
        "model's lower and upper demand estimates; the project's target was 80%."
    )
    with st.expander("How does this support inventory decisions?", expanded=False):
        st.write(
            "LightGBM estimates demand. The replenishment policy then combines "
            "the forecast with **assumed starting inventory, safety stock, and review "
            "flags**. The app uses historical M5 data and simulated inventory; "
            "it does not place actual orders."
        )
        st.write(
            "Lower error alone does not guarantee a better order: systematic bias, "
            "uncertainty and manager review are important when deciding what to replenish."
        )
        st.caption("Source: IDS 560 forecasting evaluation and research poster.")
