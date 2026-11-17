"""Navigation guide for the IDS 560 Executive Overview."""
from __future__ import annotations

from html import escape
import streamlit as st


def render_overview_intro(store: str) -> None:
    """Show three always-visible white navigation cards on Executive Overview.

    White card surfaces with subtle navy, gold, and blue top accents match
    the dashboard presentation while preserving readability.
    """
    safe_store = escape(str(store))
    st.html(f"""
    <style>
      .eo-nav-guide {{
        display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
        gap:16px; width:100%; margin:2px 0 28px;
      }}
      .eo-nav-guide .eo-nav-card {{
        box-sizing:border-box; min-width:0; background:#fff !important;
        border:1px solid #d2dbe6; border-top:4px solid #1f3368; border-radius:9px;
        padding:13px 15px 17px; min-height:116px;
      }}
      .eo-nav-guide .eo-nav-card:nth-child(1) {{border-top-color:#1f3368;}}
      .eo-nav-guide .eo-nav-card:nth-child(2) {{border-top-color:#f4bd35;}}
      .eo-nav-guide .eo-nav-card:nth-child(3) {{border-top-color:#4877b9;}}
      .eo-nav-guide .eo-nav-title {{
        color:#172e58; font-size:16px; font-weight:700;
        line-height:1.45; margin:0 0 15px;
      }}
      .eo-nav-guide .eo-nav-description {{
        color:#617189; font-size:14px; line-height:1.55; margin:0;
      }}
      @media(max-width:770px) {{
        .eo-nav-guide {{grid-template-columns:1fr; gap:10px;}}
        .eo-nav-guide .eo-nav-card {{min-height:0;}}
      }}
    </style>
    <div class="eo-nav-guide" aria-label="Dashboard navigation guide">
      <section class="eo-nav-card">
        <div class="eo-nav-title">1 · Explore demand</div>
        <p class="eo-nav-description">You're viewing {safe_store}. Change stores in the sidebar and review the 7-day overview below.</p>
      </section>
      <section class="eo-nav-card">
        <div class="eo-nav-title">2 · Plan inventory</div>
        <p class="eo-nav-description">Open <strong>Inventory Planning</strong> to inspect 14-day replenishment decisions and manually adjust stock.</p>
      </section>
      <section class="eo-nav-card">
        <div class="eo-nav-title">3 · Review evidence</div>
        <p class="eo-nav-description">Use <strong>Model Insights</strong> for validation and <strong>AI Assistant</strong> for dated supporting evidence.</p>
      </section>
    </div>
    """)
