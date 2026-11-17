"""Four-step navigation guide for the IDS 560 Executive Overview."""
from __future__ import annotations

import html
import streamlit as st


def render_overview_intro(store: str) -> None:
    """Show task, short description, and exact destination for each step."""
    cards = [
        (
            "1 · Explore demand",
            f"Compare stores and spot potential shortages for {store}.",
            "Executive Overview / Demand Forecasting",
            "#EAF2FF", "#173B70", "#BCD4F4",
        ),
        (
            "2 · Plan inventory",
            "Check suggested orders and inventory needs.",
            "Inventory Planning → Monitoring & Ordering",
            "#EAF6F0", "#195A43", "#B9E2CD",
        ),
        (
            "3 · Review orders",
            "Inspect flagged orders and make a decision.",
            "Inventory Planning → Review & Approval",
            "#FFF4DB", "#7A520F", "#EAD39B",
        ),
        (
            "4 · Explore evidence",
            "Check model results or ask the AI Assistant when needed.",
            "Model Insights / AI Assistant",
            "#F0ECFB", "#594682", "#DAD0F1",
        ),
    ]
    columns = st.columns(4, gap="small")
    for column, (heading, description, destination, background, heading_color, border) in zip(columns, cards):
        with column:
            st.markdown(
                f'''<div style="background:{background};border:1px solid {border};border-radius:10px;
                min-height:158px;padding:15px;box-sizing:border-box;overflow-wrap:anywhere;">
                <div style="font-weight:700;color:{heading_color};font-size:14px;margin-bottom:9px;">{html.escape(heading)}</div>
                <div style="color:#475569;font-size:12.5px;line-height:1.5;margin-bottom:11px;">{html.escape(description)}</div>
                <div style="border-top:1px solid {border};padding-top:9px;color:{heading_color};font-size:12px;font-weight:650;line-height:1.4;">{html.escape(destination)}</div>
                </div>''',
                unsafe_allow_html=True,
            )
