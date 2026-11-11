
import streamlit as st


def render_dashboard_theme() -> None:
    """Apply a Walmart-inspired visual system with sidebar navigation."""
    st.markdown(
        """
<style>
:root {
  --wm-navy: #102b72;
  --wm-navy-2: #173f90;
  --wm-blue: #0071ce;
  --wm-blue-soft: #eaf2ff;
  --wm-yellow: #ffc220;
  --wm-yellow-soft: #fff7db;
  --wm-bg: #f7f9fc;
  --wm-panel: #ffffff;
  --wm-border: #dbe4f1;
  --wm-text: #102451;
  --wm-muted: #64748b;
  --wm-shadow: 0 8px 24px rgba(16, 36, 81, 0.08);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
  background: var(--wm-bg) !important;
  color: var(--wm-text) !important;
}
[data-testid="stHeader"] {
  background: rgba(247,249,252,0.96) !important;
  backdrop-filter: blur(6px);
}
.main .block-container {
  max-width: 1380px !important;
  padding-top: 1.85rem !important;
  padding-bottom: 4rem !important;
}

/* Typography */
.main h1, .main h2, .main h3 {
  color: var(--wm-text) !important;
  letter-spacing: -0.02em;
}
.main p, .main li, .main label, .main span, .main div {
  color: inherit;
}
.main [data-testid="stCaptionContainer"], .main small {
  color: var(--wm-muted) !important;
}

/* Sidebar */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
  background: linear-gradient(180deg, #f6f9ff 0%, #eef4fb 100%) !important;
}
[data-testid="stSidebar"] {
  border-right: 1px solid var(--wm-border) !important;
}
[data-testid="stSidebar"] .block-container {
  padding-top: 1.2rem !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
  color: var(--wm-text) !important;
}

/* Sidebar navigation radio */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
  gap: 0.4rem;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  border: 1px solid transparent !important;
  border-radius: 12px !important;
  padding: 0.55rem 0.75rem !important;
  background: transparent !important;
  transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: #ffffff !important;
  border-color: var(--wm-border) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
  display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
  font-weight: 700 !important;
  color: var(--wm-text) !important;
  margin: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div,
[data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div + div {
  color: var(--wm-text) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: linear-gradient(90deg, #fff0ad 0%, #ffe07b 100%) !important;
  border-color: #f2cb55 !important;
  box-shadow: inset 3px 0 0 var(--wm-yellow), 0 2px 8px rgba(16,36,81,0.05);
}

/* Inputs */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] input,
textarea {
  background: #fff !important;
  border: 1px solid var(--wm-border) !important;
  border-radius: 12px !important;
}

/* Expanders / cards */
[data-testid="stExpander"] {
  border: 1px solid var(--wm-border) !important;
  background: var(--wm-panel) !important;
  border-radius: 12px !important;
  overflow: hidden;
  box-shadow: 0 2px 10px rgba(16,36,81,0.03);
}
[data-testid="stExpander"] summary {
  background: #fff !important;
  color: var(--wm-text) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-color: var(--wm-border) !important;
  border-radius: 14px !important;
}

/* Metrics */
[data-testid="stMetric"] {
  background: var(--wm-panel) !important;
  border: 1px solid var(--wm-border) !important;
  border-radius: 14px !important;
  padding: 18px 18px 16px 18px !important;
  box-shadow: var(--wm-shadow);
}
[data-testid="stMetricLabel"] {
  color: #385073 !important;
  font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
  color: var(--wm-text) !important;
  font-weight: 800 !important;
}

/* Dataframes and plots */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  background: #fff !important;
  border: 1px solid var(--wm-border) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
}
[data-testid="stPlotlyChart"] {
  background: #fff !important;
  border: 1px solid var(--wm-border) !important;
  border-radius: 14px !important;
  padding: 10px 10px 4px 10px !important;
  box-shadow: var(--wm-shadow);
}

/* Buttons */
div.stButton > button,
[data-testid="stDownloadButton"] button,
button[data-testid="stFormSubmitButton"] {
  border-radius: 10px !important;
  font-weight: 700 !important;
}
div.stButton > button[kind="primary"],
button[data-testid="stFormSubmitButton"] {
  background: var(--wm-navy) !important;
  color: white !important;
  border: 1px solid var(--wm-navy) !important;
}
div.stButton > button[kind="primary"]:hover,
button[data-testid="stFormSubmitButton"]:hover {
  background: var(--wm-navy-2) !important;
  border-color: var(--wm-yellow) !important;
}

/* Tabs if still used elsewhere */
button[data-baseweb="tab"] {
  color: #425779 !important;
  font-weight: 600 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--wm-text) !important;
}
[data-baseweb="tab-highlight"] {
  background: var(--wm-yellow) !important;
  height: 4px !important;
  border-radius: 999px !important;
}

/* Footer separation */
.smartstock-footer {
  margin-top: 7rem !important;
}

/* Hero banner */
.wm-hero {
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.25rem;
  padding: 1.2rem 1.5rem;
  margin: 0 0 1.25rem 0;
  border-radius: 16px;
  background: linear-gradient(110deg, #10265f 0%, #173d88 72%, #244d9a 100%);
  box-shadow: 0 10px 28px rgba(16,36,81,0.12);
}
.wm-hero::before {
  content: "";
  position: absolute;
  right: 72px;
  top: -24px;
  width: 18px;
  height: 180px;
  background: var(--wm-yellow);
  transform: rotate(35deg);
  opacity: 0.95;
}
.wm-hero::after {
  content: "";
  position: absolute;
  right: 42px;
  top: -26px;
  width: 11px;
  height: 180px;
  background: #ffe58a;
  transform: rotate(35deg);
  opacity: 0.85;
}
.wm-brand {
  display: flex;
  align-items: center;
  gap: 1rem;
  color: #fff;
  position: relative;
  z-index: 1;
  flex-wrap: wrap;
}
.wm-brand-title {
  font-size: 2rem;
  font-weight: 800;
  line-height: 1.1;
  color: #fff;
}
.wm-brand-subtitle {
  font-size: 0.98rem;
  font-weight: 500;
  color: #e8efff;
}
.wm-wordmark {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  font-size: 2rem;
  font-weight: 800;
  line-height: 1;
  color: #fff;
  white-space: nowrap;
}
.wm-wordmark .spark {
  color: var(--wm-yellow);
  font-size: 2.2rem;
}
.wm-divider {
  width: 1px;
  align-self: stretch;
  background: rgba(255,255,255,0.35);
}
.wm-badge {
  position: relative;
  z-index: 1;
  color: #fff;
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 999px;
  padding: 0.5rem 0.8rem;
  font-size: 0.8rem;
  font-weight: 700;
  white-space: nowrap;
}

@media (max-width: 900px) {
  .wm-hero {
    padding: 1.15rem 1.1rem;
  }
  .wm-divider, .wm-badge {
    display: none;
  }
  .wm-wordmark, .wm-brand-title {
    font-size: 1.55rem;
  }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_banner() -> None:
    st.markdown(
        """
<div class="wm-hero">
  <div class="wm-brand">
    <div class="wm-wordmark">Walmart <span class="spark">✳</span></div>
    <div class="wm-divider"></div>
    <div>
      <div class="wm-brand-title">SmartStock AI</div>
      <div class="wm-brand-subtitle">Demand Forecasting &amp; Inventory Planning</div>
    </div>
  </div>
  <div class="wm-badge">IDS 560 · Academic Prototype</div>
</div>
        """,
        unsafe_allow_html=True,
    )
