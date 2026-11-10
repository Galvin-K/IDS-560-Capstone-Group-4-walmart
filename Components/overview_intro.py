"""Compact Executive Overview introduction for the IDS 560 dashboard.

Historical metrics are fixed 2016 holdout results, not live/store-specific ones.
"""
from __future__ import annotations

import streamlit as st


def render_overview_intro(store: str) -> None:
    """Guide first-time visitors and clearly label evaluation figures."""
    st.markdown(
        "**Retail Inventory Decision Support** — Explore historical product demand, "
        "forecasted sales, and simulated replenishment decisions from the public "
        "Walmart M5 dataset. The planner reviews each recommendation; this prototype "
        "does not place real orders."
    )

    st.markdown("##### Get started")
    steps = st.columns(3, gap="small")
    with steps[0]:
        with st.container(border=True):
            st.markdown("**1 · Explore demand**")
            st.caption(
                f"You're viewing **{store}**. Change stores in the sidebar "
                "and review the seven-day overview below."
            )
    with steps[1]:
        with st.container(border=True):
            st.markdown("**2 · Plan inventory**")
            st.caption(
                "Open **Inventory Planning** for 14-day replenishment decisions "
                "and manual stock adjustments."
            )
    with steps[2]:
        with st.container(border=True):
            st.markdown("**3 · Review evidence**")
            st.caption(
                "Use **Model Insights** for evaluation and **AI Assistant** "
                "for dated supporting evidence."
            )

    with st.expander("Historical model performance · 2016 holdout", expanded=False):
        st.markdown(
            "**How to read these numbers:** WAPE is the sum of absolute forecast "
            "errors divided by actual sales; **lower is better**. A WAPE of 71.6% "
            "is **not** 71.6% accuracy. **Recent Mean** is a simple baseline that "
            "predicts using average recent sales. **Bias** shows whether forecasts "
            "are systematically too high or too low."
        )
        a, b, c = st.columns(3)
        a.metric("7-day LightGBM WAPE", "71.6%")
        b.metric("7-day Recent Mean WAPE", "74.6%")
        c.metric("7-day LightGBM bias", "−0.7%")
        st.caption(
            "Fixed 2016 holdout results for the 50% stratified M5 sample, "
            "not store-specific or live performance. More details in Model Insights."
        )

    st.caption(
        "**Data scope:** Historical M5 sales through 2016; forecast and inventory "
        "examples are demonstrations. Starting stock is simulated, not actual Walmart "
        "inventory. Selling-price estimates are not supplier purchasing costs."
    )
