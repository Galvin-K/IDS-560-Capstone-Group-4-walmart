"""Visual-only styling for the IDS 560 Retail Inventory Decision Support app."""
import streamlit as st


def render_dashboard_theme() -> None:
    # Avoid forcing fonts on icon containers; Streamlit uses a separate icon font.
    st.markdown("""
<style>
:root {
  --ds-ink: #1d355a;
  --ds-navy: #18376b;
  --ds-blue: #286bc1;
  --ds-yellow: #ffc220;
  --ds-bg: #f7f9fc;
  --ds-line: #dfe6ef;
  --ds-muted: #63748d;
  --ds-card: #ffffff;
}
[data-testid="stAppViewContainer"] { background: var(--ds-bg) !important; }
[data-testid="stHeader"] { background: rgba(247,249,252,.96) !important; }
.main .block-container { max-width:1400px!important; padding-top:1.55rem!important; padding-bottom:3.5rem!important; }
/* Unified type, without changing Streamlit's icon glyphs */
[data-testid="stMarkdownContainer"] :is(p,li,h1,h2,h3,h4),
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] :is(p,h1,h2,h3,h4),
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] p {
 font-family: "Aptos", "Segoe UI Variable", "Segoe UI", system-ui, sans-serif !important;
}
.main :is(h1,h2,h3) { color:var(--ds-ink)!important; letter-spacing:-.018em!important; }
.main h1 {font-size:1.8rem!important;font-weight:650!important;}
.main h2 {font-size:1.45rem!important;font-weight:630!important;}
.main h3 {font-size:1.14rem!important;font-weight:620!important;}
.main p {line-height:1.6;}
[data-testid="stCaptionContainer"] p {color:var(--ds-muted)!important;font-size:.85rem!important;}
/* Consistent section grouping, easier to scan than plain continuous text */
.main [data-testid="stHeadingWithActionElements"]:has(h2){
  margin-bottom:0.85rem!important;
}
.main [data-testid="stHeadingWithActionElements"]:has(h2) h2 {
  font-size:1.38rem!important;
}
.main [data-testid="stHeadingWithActionElements"]:has(h3) {
  margin-top:.75rem!important;
}
/* Refined sections: a navy guide line and subtle separation instead of empty headings */
.main [data-testid="stHeadingWithActionElements"]:has(h2) {
  border-left:3px solid #3465a5; padding:8px 12px 8px 13px; margin-top:1.25rem; background:#edf3fb; border-radius:0 8px 8px 0;
}
.main [data-testid="stHeadingWithActionElements"]:has(h3) {
  padding-bottom:3px; border-bottom:1px solid #e3e9f2;
}
/* Navigation */
[data-testid="stSidebar"],[data-testid="stSidebarContent"] {background:#f1f5fb!important;}
[data-testid="stSidebar"] {border-right:1px solid var(--ds-line)!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {gap:.30rem!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  display:flex!important; align-items:center!important; border:1px solid transparent!important;
  padding:.66rem .82rem!important; border-radius:10px!important; background:transparent!important;
  min-height:40px!important; width:100%; transition:background .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label [data-baseweb="radio"],
[data-testid="stSidebar"] [data-testid="stRadio"] label [role="radio"],
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child:has(input) {display:none!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:#fff!important;border-color:var(--ds-line)!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] label:has([aria-checked="true"]) {
 background:#fff1b9!important; border-color:#efd37b!important;
 box-shadow:inset 3px 0 0 var(--ds-yellow)!important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {font-size:.93rem!important;font-weight:550!important;color:var(--ds-ink)!important;margin:0!important;}
/* Cards / data */
[data-testid="stMetric"] {background:#fff!important;border:1px solid var(--ds-line)!important;border-radius:12px!important;padding:14px 17px!important;box-shadow:0 3px 10px rgba(23,46,100,.04)!important;}
[data-testid="stMetricLabel"] {font-size:.85rem!important;font-weight:500!important;color:#4e617b!important;}
[data-testid="stMetricValue"] {
  color:var(--ds-ink)!important;font-size:clamp(1.15rem,1.55vw,1.52rem)!important;
  font-weight:520!important;line-height:1.35!important;letter-spacing:-.025em!important;
  font-variant-numeric:tabular-nums!important;
}
[data-testid="stExpander"] {background:#fff!important;border:1px solid var(--ds-line)!important;border-radius:10px!important;overflow:hidden!important;}
[data-testid="stExpander"] summary {background:#fff!important;color:var(--ds-ink)!important;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:var(--ds-line)!important;border-radius:12px!important;}
[data-testid="stPlotlyChart"] {background:#fff!important;border:1px solid var(--ds-line)!important;border-radius:12px!important;padding:9px!important;box-shadow:0 2px 8px rgba(23,46,100,.035)!important;}
[data-testid="stDataFrame"],[data-testid="stTable"] {background:#fff!important;border:1px solid var(--ds-line)!important;border-radius:10px!important;overflow:hidden!important;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-testid="stNumberInput"] input {background:#fff!important;border:1px solid #d0dbe8!important;border-radius:9px!important;}
div.stButton>button,[data-testid="stDownloadButton"] button {border-radius:9px!important;font-weight:550!important;}
div.stButton>button[kind="primary"],button[data-testid="stFormSubmitButton"] {background:var(--ds-navy)!important;color:#fff!important;border-color:var(--ds-navy)!important;}
/* Consistent, compact banner */
.ds-banner {
 display:flex;justify-content:space-between;align-items:center;gap:20px;
 background:linear-gradient(110deg,#172e64,#254981);color:#fff;
 border-radius:12px;padding:22px 27px;margin:0 -14px 26px;width:calc(100% + 28px);position:relative;overflow:hidden;
}
.ds-banner::after {content:"";position:absolute;width:12px;height:140%;background:var(--ds-yellow);right:38px;top:-20%;transform:skew(-31deg);}
.ds-banner-main {position:relative;z-index:1;display:flex;align-items:center;gap:19px;min-width:0;}
.ds-banner-mark {color:#fff;font-size:17px;font-weight:600;white-space:nowrap;}
.ds-banner-mark .ds-spark {color:var(--ds-yellow);margin-left:4px;font-size:19px;}
.ds-banner-rule {width:1px;height:38px;background:rgba(255,255,255,.3);flex-shrink:0;}
.ds-banner-title {font-size:clamp(18px,2vw,23px);font-weight:610;line-height:1.25;letter-spacing:-.018em;color:#fff;}
.ds-banner-subtitle {font-size:12px;color:#dfe9fc;margin-top:4px;}
.ds-banner-badge {color:#dfe9fc;font-size:11px;margin-right:48px;z-index:1;white-space:nowrap;}
.smartstock-footer {margin-top:6.5rem!important;}
@media(max-width:780px){.ds-banner{padding:16px;margin-left:0;width:100%;}.ds-banner-main{gap:11px;flex-wrap:wrap}.ds-banner-rule,.ds-banner-badge{display:none}.ds-banner-title{font-size:19px}.ds-banner::after{right:7px;opacity:.5}}
@media(prefers-reduced-motion:reduce){[data-testid="stSidebar"] [data-testid="stRadio"] label {transition:none!important;}}

/* V5: left navigation uses meaningful icons rather than radio circles. */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {
  display:flex !important;
  align-items:center !important;
  gap:9px !important;
  padding:10px 12px !important;
  min-height:45px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label::before {
  content:"";
  display:inline-block;
  flex:0 0 20px;
  width:20px;
  height:20px;
  background-size:20px 20px;
  background-repeat:no-repeat;
  background-position:center;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label [data-baseweb="radio"],
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label [role="radio"],
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label > div:first-child:has(input) {
  display:none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(1)::before { background-image: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2318376b%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22m3%2010%209-7%209%207v10a1%201%200%200%201-1%201h-5v-7H9v7H4a1%201%200%200%201-1-1z%22%2F%3E%3C%2Fsvg%3E"); }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(2)::before { background-image: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2318376b%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M4%2020V12h4v8M10%2020V7h4v13M16%2020V4h4v16M3%2020h18%22%2F%3E%3C%2Fsvg%3E"); }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(3)::before { background-image: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2318376b%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22m12%202%209%205v10l-9%205-9-5V7zM3%207l9%205%209-5M12%2012v10%22%2F%3E%3C%2Fsvg%3E"); }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(4)::before { background-image: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2318376b%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22m12%203%209%205-9%205-9-5zM3%2012l9%205%209-5M3%2016l9%205%209-5%22%2F%3E%3C%2Fsvg%3E"); }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:nth-child(5)::before { background-image: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2318376b%22%20stroke-width%3D%221.8%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpath%20d%3D%22M21%2011.5a8.5%208.5%200%200%201-8.5%208.5%209%209%200%200%201-4-.9L3%2021l1.9-5.5a8.5%208.5%200%201%201%2016.1-4z%22%2F%3E%3Cpath%20d%3D%22M8%2011.5h.01M12%2011.5h.01M16%2011.5h.01%22%20stroke-width%3D%222.5%22%20stroke-linecap%3D%22round%22%2F%3E%3C%2Fsvg%3E"); }
/* V5: denser dashboard-style table panels, without interfering with iframe internals. */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border:1px solid #d5e0ef !important;
  border-radius:12px !important;
  background:#fff !important;
  box-shadow:0 5px 16px rgba(21,48,101,.07) !important;
  overflow:hidden !important;
}
[data-testid="stDataFrame"]:focus-within {
  outline:2px solid rgba(40,107,193,.25) !important;
  outline-offset:1px !important;
}
/* Make table and chart sections easier to distinguish from the neutral canvas. */
.main [data-testid="stHeadingWithActionElements"]:has(h2) {
  background:#18376b !important;
  border-left:4px solid #ffc220 !important;
  border-radius:8px !important;
  padding:10px 14px !important;
  margin-top:1.9rem !important;
  margin-bottom:1.0rem !important;
}
.main [data-testid="stHeadingWithActionElements"]:has(h2) h2,
.main [data-testid="stHeadingWithActionElements"]:has(h2) a {
  color:#fff !important;
}
/* Keep top-level page headings clean rather than forcing every title into a panel. */
.main [data-testid="stHeadingWithActionElements"]:has(h1) {
  margin-bottom:.7rem !important;
}
/* Move the banner up without modifying any of its internal elements. */
.ds-banner {
  margin-top:-18px !important;
}
@media(max-width:780px) {
 .ds-banner {margin-top:-8px !important;}
}

</style>
""", unsafe_allow_html=True)


def render_dashboard_banner() -> None:
    st.markdown("""
<div class="ds-banner" role="banner" aria-label="Retail Inventory Decision Support">
  <div class="ds-banner-main">
    <div class="ds-banner-mark">Walmart M5 <span class="ds-spark">✳</span></div>
    <div class="ds-banner-rule" aria-hidden="true"></div>
    <div>
      <div class="ds-banner-title">Retail Inventory Decision Support</div>
      <div class="ds-banner-subtitle">Forecasting · Inventory planning · Evidence-grounded insights</div>
    </div>
  </div>
  <div class="ds-banner-badge">IDS 560 · Academic project</div>
</div>
""", unsafe_allow_html=True)
