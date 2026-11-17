"""Navigation guide for the IDS 560 Executive Overview."""
from __future__ import annotations

import streamlit as st


def render_overview_intro(store: str) -> None:
    """Display navigation cards openly, without redundant model or data summaries."""
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
