"""Option C — navy-accented Walmart M5 dashboard theme.

UI-only: no changes to model outputs, inventory logic, data or RAG.
Call render_dashboard_theme() AFTER legacy CSS and BEFORE the page header.
Optional: replace the existing Walmart title block with render_dashboard_banner().
"""

import streamlit as st


def render_dashboard_theme() -> None:
    """Apply a restrained neutral, navy, blue and amber theme."""
    st.html("""
<style>
/* OPTION_C_NAUTICAL: single source for app-wide visual overrides */
:root {
  --c-navy: #132c60;
  --c-navy-deep: #102451;
  --c-blue: #1767b8;
  --c-yellow: #ffc220;
  --c-canvas: #f7f9fc;
  --c-surface: #ffffff;
  --c-sidebar: #f1f5fb;
  --c-border: #dce5f1;
  --c-muted: #61738e;
  --c-shadow: 0 4px 15px rgba(18,42,86,.065);
}
/* Overall canvas: deliberately off-white, never flat gray */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--c-canvas) !important;
    color: var(--c-navy-deep);
}
[data-testid="stHeader"] { background: rgba(247,249,252,.97) !important; }
.main .block-container { max-width: 1300px !important; padding-top: 2.2rem; padding-bottom: 3rem; }
/* Sidebar */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
  background: var(--c-sidebar) !important;
}
[data-testid="stSidebar"] { border-right: 1px solid var(--c-border) !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--c-navy-deep) !important; }
/* Type */
.main h1, .main h2, .main h3 { color: var(--c-navy-deep); letter-spacing:-.018em; }
.main [data-testid="stCaptionContainer"], .main small { color:var(--c-muted); }
/* Active tab: yellow indicator, blue/navy text */
button[data-baseweb="tab"] { color:#3f5170 !important; font-weight:600; padding-bottom:11px; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--c-blue) !important; font-weight:760; }
[data-baseweb="tab-highlight"] { background-color:var(--c-yellow) !important; height:4px !important; border-radius:3px !important; }
/* Native metrics: light lift and clear border */
[data-testid="stMetric"] {
  background:var(--c-surface) !important;
  border:1px solid var(--c-border) !important;
  box-shadow:var(--c-shadow);
  border-radius:13px !important;
  padding:19px 17px !important;
}
[data-testid="stMetricValue"] { color:var(--c-navy-deep) !important; font-weight:740 !important; }
[data-testid="stMetricLabel"] { color:#344867 !important; }
/* Panels / cards / expanders */
[data-testid="stExpander"] {
  border:1px solid var(--c-border) !important; border-radius:11px !important;
  background:#fff !important; overflow:hidden;
}
[data-testid="stExpander"] summary { background:#f9fbff !important; color:var(--c-navy-deep) !important; }
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-color:var(--c-border) !important;
  border-radius:13px !important;
}
/* Native selection / input */
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-testid="stNumberInput"] input {
  border-color:#cdd9e8 !important;
  border-radius:9px !important;
  background:var(--c-surface) !important;
}
/* Dark navy selected store and primary action; soft gold highlight on hover */
div.stButton > button, [data-testid="stDownloadButton"] button { border-radius:9px !important; }
div.stButton > button[kind="primary"], button[data-testid="stFormSubmitButton"] {
  background:var(--c-navy) !important; border-color:var(--c-navy) !important;
  color:#fff !important; font-weight:650 !important;
}
div.stButton > button[kind="primary"]:hover, button[data-testid="stFormSubmitButton"]:hover {
  background:#1d4385 !important; border-color:var(--c-yellow) !important;
}
/* Chart backgrounds and tables */
[data-testid="stPlotlyChart"] {
  background:var(--c-surface) !important; border:1px solid var(--c-border) !important;
  border-radius:13px !important; padding:9px 8px !important;
  box-shadow:var(--c-shadow);
}
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius:11px !important; overflow:hidden; }
/* Header for optional navy brand banner */
.ids-c-hero {
  display:flex; justify-content:space-between; align-items:center; gap:22px;
  width:100%; padding:22px 28px; margin:0 0 22px;
  background:linear-gradient(115deg,#102452 0%,#1a3977 80%,#284c92 100%);
  border-radius:13px; color:#fff; position:relative; overflow:hidden;
  box-shadow:0 8px 22px rgba(16,36,82,.12);
}
.ids-c-hero::after {
  content:""; position:absolute; width:22px; height:155%; top:-25%; right:38px;
  background:var(--c-yellow); transform:skew(-28deg);
}
.ids-c-hero h1 { color:#fff !important; font-size:clamp(22px,2.6vw,33px); line-height:1.2; margin:0 0 5px !important; }
.ids-c-hero p { color:#e7efff !important; font-size:13px; margin:0; }
.ids-c-mark { width:148px; max-width:100%; height:auto; object-fit:contain; }
.ids-c-hero-content { z-index:1; flex:1; }
.ids-c-chip { position:relative; z-index:1; background:rgba(255,255,255,.10);
  border:1px solid rgba(255,255,255,.23); color:#fff; font-size:12px; padding:7px 10px;
  border-radius:30px; margin-right:40px; white-space:nowrap; }
/* Optional section banners: apply only where explicitly requested */
.ids-c-section {
  background:var(--c-navy); color:#fff; border-radius:9px 9px 0 0;
  padding:11px 15px; font-size:17px; font-weight:750; letter-spacing:.01em;
}
/* Responsive behavior */
@media(max-width: 720px) {
 .main .block-container { padding-left:1rem; padding-right:1rem; }
 .ids-c-hero { flex-wrap:wrap; padding:20px; }
 .ids-c-chip { display:none; }
 .ids-c-hero::after { right:14px; }
}
@media(prefers-reduced-motion:reduce) {
  * { scroll-behavior:auto !important; }
}
</style>
""")


def render_dashboard_banner() -> None:
    """Optional replacement for existing app's inline Walmart header (not an extra heading)."""
    st.html('''
<div class="ids-c-hero" aria-label="Demand and Inventory Planner">
  <div class="ids-c-hero-content">
    <h1>Demand &amp; Inventory Planner</h1>
    <p>Walmart M5 historical data · Forecasting &amp; replenishment decision support</p>
  </div>
  <span class="ids-c-chip">IDS 560 · Academic prototype</span>
</div>
''')
