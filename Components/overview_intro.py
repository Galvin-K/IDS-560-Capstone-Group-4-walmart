"""Executive Overview introduction for the UIC IDS 560 Streamlit capstone.

Historical holdout metrics are fixed, documented evaluation results, NOT metrics
for the user's selected store or the deployed January 2016 demonstration week.
"""
from __future__ import annotations

import streamlit as st


def render_overview_intro(store: str) -> None:
    """Render a compact guide and historical model-performance context."""
    st.markdown(
        "**Retail Inventory Decision Support** — Explore historical product demand, "
        "forecasted sales, and simulated replenishment decisions from the public "
        "Walmart M5 dataset. The planner reviews every recommendation; this prototype "
        "does not place real orders."
    )

    st.markdown("##### Get started")
    steps = st.columns(3, gap="small")
    with steps[0]:
        with st.container(border=True):
            st.markdown("**1 · Explore demand**")
            st.caption(f"You're viewing **{store}**. Change stores in the sidebar and review the 7-day overview below.")
    with steps[1]:
        with st.container(border=True):
            st.markdown("**2 · Plan inventory**")
            st.caption("Open **Inventory Planning** to inspect 14-day replenishment decisions and manually adjust stock.")
    with steps[2]:
        with st.container(border=True):
            st.markdown("**3 · Review evidence**")
            st.caption("Use **Model Insights** for validation and **AI Assistant** for dated supporting evidence.")

    with st.expander("Historical model performance · 2016 holdout", expanded=False):
        a, b, c = st.columns(3)
        a.metric("7-day LightGBM WAPE", "71.6%")
        b.metric("7-day Recent Mean WAPE", "74.6%")
        c.metric("7-day LightGBM bias", "−0.7%")
        st.caption(
            "Fixed evaluation results for the 50% stratified M5 sample; "
            "lower WAPE is better. These values are not recomputed for the "
            "selected store, and do not measure live replenishment results."
        )

    st.caption(
        "**Data scope:** Historical M5 sales through 2016; forecast and inventory "
        "examples are demonstrations. Starting stock is simulated, not actual Walmart "
        "inventory. Selling-price estimates are not supplier purchasing costs."
    )
