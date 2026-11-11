"""Executive Overview introduction and historical model evaluation context."""
from __future__ import annotations
from html import escape
import streamlit as st


def render_overview_intro(store: str) -> None:
    """Show actionable, compact guidance without changing dashboard calculations."""
    store_name = escape(str(store))
    st.html(f"""
    <style>
    .ds-intro {{font-family:'Aptos','Segoe UI',system-ui,sans-serif;color:#1d355a;}}
    .ds-intro .ds-intro-summary {{font-size:14px;line-height:1.7;color:#405672;margin:0 0 20px;}}
    .ds-intro .ds-journey-title {{font-size:15px;font-weight:650;margin:0 0 12px;letter-spacing:-.01em;}}
    .ds-intro .ds-journey-grid {{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;}}
    .ds-intro .ds-journey-card {{position:relative;background:#fff;border:1px solid #dbe4ef;border-radius:13px;padding:17px 16px 15px;min-height:146px;box-shadow:0 5px 16px rgba(21,48,101,.045);}}
    .ds-intro .ds-journey-card:nth-child(1) {{border-top:3px solid #286bc1;}}
    .ds-intro .ds-journey-card:nth-child(2) {{border-top:3px solid #ffc220;}}
    .ds-intro .ds-journey-card:nth-child(3) {{border-top:3px solid #6587b5;}}
    .ds-intro .ds-number {{display:inline-flex;align-items:center;justify-content:center;width:27px;height:27px;border-radius:9px;background:#edf3fc;font-size:12px;font-weight:650;color:#20487f;}}
    .ds-intro .ds-card-title {{font-size:14px;font-weight:650;color:#1d355a;margin:12px 0 5px;}}
    .ds-intro .ds-card-text {{font-size:12.5px;line-height:1.6;color:#63748d;margin:0;}}
    .ds-intro .ds-card-tag {{display:inline-block;font-size:11px;margin-top:11px;padding:3px 7px;border-radius:5px;background:#f2f6fc;color:#315b93;font-weight:600;}}
    @media(max-width:820px) {{.ds-intro .ds-journey-grid{{grid-template-columns:1fr;}}.ds-intro .ds-journey-card{{min-height:0;}}}}
    </style>
    <section class="ds-intro" aria-label="How to use the inventory dashboard">
      <p class="ds-intro-summary">Explore historical demand, assess possible stock gaps, and review simulated replenishment decisions. <strong>These are planning demonstrations, not actual Walmart purchase orders.</strong></p>
      <div class="ds-journey-title">Start with a business question</div>
      <div class="ds-journey-grid">
        <article class="ds-journey-card">
          <span class="ds-number">01</span>
          <div class="ds-card-title">Where might stock run short?</div>
          <p class="ds-card-text">Review the seven-day gap for <strong>{store_name}</strong>, or choose another store using the sidebar.</p>
          <span class="ds-card-tag">Executive Overview</span>
        </article>
        <article class="ds-journey-card">
          <span class="ds-number">02</span>
          <div class="ds-card-title">What should we replenish?</div>
          <p class="ds-card-text">Open the 14-day plan to review order quantities, safety stock, and items requiring approval.</p>
          <span class="ds-card-tag">Inventory Planning</span>
        </article>
        <article class="ds-journey-card">
          <span class="ds-number">03</span>
          <div class="ds-card-title">What supports the decision?</div>
          <p class="ds-card-text">Check validation findings in Model Insights, then explore dated evidence with the AI Assistant.</p>
          <span class="ds-card-tag">Model Insights · AI Assistant</span>
        </article>
      </div>
    </section>
    """)

    with st.expander("Historical model performance · 2016 holdout", expanded=False):
        st.markdown(
            "**Before interpreting the results:** WAPE measures total absolute forecast error "
            "relative to actual sales (**lower is better**). It is **not** an accuracy score. "
            "Recent Mean uses recent average sales as a baseline. Bias indicates whether "
            "forecasts tend to run high or low."
        )
        a, b, c = st.columns(3)
        a.metric("LightGBM · 7-day WAPE", "71.6%")
        b.metric("Recent Mean · 7-day WAPE", "74.6%")
        c.metric("LightGBM · 7-day bias", "−0.7%")
        st.caption(
            "2016 holdout · 50% stratified item–store sample. Aggregate evaluation, "
            "not live or store-specific performance."
        )

    st.caption(
        "**Data scope:** Historical M5 sales through 2016. Starting inventory is simulated "
        "rather than actual Walmart stock. Selling-price estimates are not supplier purchasing costs."
    )
