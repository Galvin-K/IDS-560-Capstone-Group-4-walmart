"""Navigation guide for the IDS 560 Executive Overview."""
from __future__ import annotations

import html
import streamlit as st


def render_overview_intro(store: str) -> None:
    """Display three colored navigation cards; no model or data summaries."""
    cards = [
        (
            "1 · Explore demand",
            f"You're viewing {store}. Change stores in the sidebar and review the 7-day overview below.",
            "#EAF2FF", "#173B70", "#BCD4F4",
        ),
        (
            "2 · Plan inventory",
            "Open Inventory Planning to inspect 14-day replenishment decisions and manually adjust stock.",
            "#EAF6F0", "#195A43", "#B9E2CD",
        ),
        (
            "3 · Review evidence",
            "Use Model Insights for validation and AI Assistant for dated supporting evidence.",
            "#FFF4DB", "#7A520F", "#EAD39B",
        ),
    ]
    columns = st.columns(3, gap="small")
    for column, (heading, description, background, heading_color, border) in zip(columns, cards):
        with column:
            st.markdown(
                f'''<div style="background:{background};border:1px solid {border};border-radius:10px;min-height:126px;padding:16px;box-sizing:border-box;">
                <div style="font-weight:700;color:{heading_color};font-size:15px;margin-bottom:12px;">{html.escape(heading)}</div>
                <div style="color:#475569;font-size:13px;line-height:1.5;">{html.escape(description)}</div>
                </div>''',
                unsafe_allow_html=True,
            )
