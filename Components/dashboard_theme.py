"""Walmart-inspired white/blue/yellow visual theme for the IDS 560 Streamlit dashboard.

Call render_dashboard_theme() after the existing DASHBOARD_STYLE_V1 CSS block
(and before the dashboard header) so these visual overrides take precedence.
No forecasting, inventory, RAG, or app state is modified.
"""

import streamlit as st


def render_dashboard_theme() -> None:
    """Apply a white-and-blue dashboard theme with Walmart-yellow accents."""
    st.markdown(
        """
<style>
/* IDS560_LIGHT_THEME_V1: white canvas / Walmart-inspired blue and yellow accents */
:root {
  --ids-navy: #142759;
  --ids-blue: #0071CE;
  --ids-yellow: #FFC220;
  --ids-orange: #FF9F1C;
  --ids-text: #122446;
  --ids-muted: #566782;
  --ids-border: #DFE7F1;
  --ids-sidebar: #F7FAFD;
}

/* Main page: remove the old gray-blue wash */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
  background: #FFFFFF !important;
  color: var(--ids-text);
}
[data-testid="stHeader"] {
  background: #FFFFFF !important;
}
.main .block-container {
  max-width: 1280px;
}

/* Sidebar and text hierarchy */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
  background: var(--ids-sidebar) !important;
}
[data-testid="stSidebar"] {
  border-right: 1px solid var(--ids-border) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
.main h1, .main h2, .main h3 {
  color: var(--ids-navy);
}
.main [data-testid="stCaptionContainer"],
.main small {
  color: var(--ids-muted);
}

/* Tabs: clear active state; yellow accent for active tab */
button[data-baseweb="tab"] {
  color: var(--ids-muted) !important;
  font-weight: 550;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--ids-navy) !important;
  border-bottom-color: var(--ids-yellow) !important;
  font-weight: 700;
}
[data-baseweb="tab-highlight"] {
  background-color: var(--ids-yellow) !important;
}

/* Metrics: white cards with fine borders and legible numbers */
[data-testid="stMetric"] {
  background: #FFFFFF !important;
  border: 1px solid var(--ids-border) !important;
  border-radius: 12px !important;
  padding: 16px !important;
  box-shadow: 0 2px 7px rgba(20,39,89,.035);
}
[data-testid="stMetricValue"] {
  color: var(--ids-navy) !important;
}

/* Expanders and native bordered containers */
[data-testid="stExpander"] {
  background: #FFFFFF !important;
  border: 1px solid var(--ids-border) !important;
  border-radius: 10px !important;
  overflow: hidden;
}
[data-testid="stExpander"] summary {
  background: #FAFCFF !important;
  color: var(--ids-navy) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
  border-color: var(--ids-border);
}

/* Tables, select boxes, inputs */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border-radius: 10px;
  overflow: hidden;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
  background: #FFFFFF !important;
  border-color: var(--ids-border) !important;
}

/* Charts should look like part of the same white canvas */
[data-testid="stPlotlyChart"] {
  background: #FFFFFF !important;
  border: 1px solid #E9EEF5;
  border-radius: 12px !important;
  padding: 8px !important;
}

/* Native buttons: restrained, consistent styling */
div.stButton > button,
[data-testid="stDownloadButton"] button {
  border-radius: 9px !important;
}
div.stButton > button[kind="primary"],
button[data-testid="stFormSubmitButton"] {
  background: var(--ids-navy) !important;
  border-color: var(--ids-navy) !important;
  color: white !important;
}
div.stButton > button[kind="primary"]:hover,
button[data-testid="stFormSubmitButton"]:hover {
  background: var(--ids-blue) !important;
  border-color: var(--ids-blue) !important;
}

/* Preserve the footer's navy surface, but replace its red accents with yellow. */
.smartstock-footer {
  border-top-color: var(--ids-yellow) !important;
}
.smartstock-footer .ss-button {
  background: var(--ids-yellow) !important;
  color: #142759 !important;
}
.smartstock-footer .ss-button:hover {
  background: var(--ids-orange) !important;
  color: #142759 !important;
}
.smartstock-footer .ss-member:hover {
  border-color: var(--ids-yellow) !important;
}
/* High contrast for the primary CTA without yellow-on-white text */
div.stButton > button[kind="primary"]:focus-visible,
button[data-testid="stFormSubmitButton"]:focus-visible {
  outline: 3px solid var(--ids-yellow);
  outline-offset: 2px;
}
@media (max-width: 650px) {
  [data-testid="stMetric"] { padding: 12px !important; }
  [data-testid="stPlotlyChart"] { padding: 4px !important; }
}
</style>
""",
        unsafe_allow_html=True,
    )
