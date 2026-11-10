"""Reusable SmartStock AI footer for Streamlit.

Call render_footer() as the LAST UI function in your app.py.
Requires Streamlit with st.html support (1.33+).
All CSS selectors are scoped to .smartstock-footer.
"""
from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

ASSET_DIR = Path(__file__).resolve().parents[1] / "Assets" / "Teams"
GITHUB_URL = "https://github.com/Galvin-K/IDS-560-Capstone-Group-4-walmart/"
ADVISOR_NAME = "Professor Fatemeh Sarayloo"
ADVISOR_EMAIL = "fsaraylo@uic.edu"

TEAM = [
    {
        "name": "Keta Patel",
        "email": "kpate650@uic.edu",
        "photo": "Keta Headshot.png",
        "qr": "Keta_QR code.jpeg",
        "linkedin": "https://www.linkedin.com/in/keta2002",
    },
    {
        "name": "Galvin Kao",
        "email": "hkao9@uic.edu",
        "photo": "Galvin Headshot.jpeg",
        "qr": "Galvin QR code.jpeg",
        "linkedin": "https://www.linkedin.com/in/galvin-kao",
    },
    {
        "name": "Tyler Yuen",
        "email": "tcyuen2@uic.edu",
        "photo": "Tyler Headshot.jpeg",
        "qr": "Tyler QR Code.png",
        "linkedin": "",
    },
    {
        "name": "Fan Mengxi",
        "email": "mfan20@uic.edu",
        "photo": "Mengxi Headshot.jpeg",
        "qr": "Mengxi QR code.jpeg",
        "linkedin": "https://www.linkedin.com/in/regina-fan-45a0b9146",
    },
    {
        "name": "Mengyi Wang",
        "email": "mwang233@uic.edu",
        "photo": "Mengyi Headshot.jpeg",
        "qr": "Mengyi QR Code.jpeg",
        "linkedin": "https://www.linkedin.com/in/carmen-wang-889334208",
    },
]


def _data_uri(filename: str, mime: str) -> str:
    path = ASSET_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing footer asset: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _safe_link(url: str) -> str:
    parsed = urlparse(url)
    return escape(url, quote=True) if parsed.scheme == "https" and parsed.netloc else ""


def _member_card(member: dict) -> str:
    name = escape(member["name"])
    email = escape(member["email"], quote=True)
    photo = _data_uri(member["photo"], "image/webp")
    qr = _data_uri(member["qr"], "image/png")
    linkedin = _safe_link(member.get("linkedin", ""))
    if linkedin:
        linked_qr = f'<a href="{linkedin}" target="_blank" rel="noopener noreferrer" aria-label="Open {name} LinkedIn"><img class="ss-qr" src="{qr}" alt="LinkedIn QR code for {name}" loading="lazy"></a>'
        direct = f'<a class="ss-linkedin" href="{linkedin}" target="_blank" rel="noopener noreferrer">LinkedIn profile ↗</a>'
    else:
        linked_qr = f'<img class="ss-qr" src="{qr}" alt="LinkedIn QR code supplied for {name}" loading="lazy">'
        direct = '<span class="ss-linkedin ss-muted">Scan LinkedIn QR</span>'
    return f'''<article class="ss-member">
        <img class="ss-photo" src="{photo}" alt="Headshot of {name}" loading="lazy">
        <h3>{name}</h3>
        <a class="ss-email" href="mailto:{email}">{email}</a>
        <div class="ss-qr-frame">{linked_qr}</div>
        {direct}
    </article>'''


def render_footer() -> None:
    """Render a non-sticky, scroll-to-bottom footer for the current Streamlit page."""
    cards = "\n".join(_member_card(member) for member in TEAM)
    github = _safe_link(GITHUB_URL)
    advisor_email = escape(ADVISOR_EMAIL, quote=True)
    html = '''
<style>
.smartstock-footer, .smartstock-footer * { box-sizing: border-box; }
.smartstock-footer { width:100%; margin-top:4rem; padding:34px 26px 22px; background:#0c245c; border-top:5px solid #c8102e; color:#f7f9ff; border-radius:16px 16px 6px 6px; font-family:system-ui,-apple-system,Segoe UI,sans-serif; }
.smartstock-footer .ss-top { text-align:center; margin:0 auto 26px; max-width:850px; }
.smartstock-footer .ss-eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:11px; color:#ffc2ca; font-weight:700; }
.smartstock-footer .brand-title {
    color: #ffffff;
    font-size: clamp(22px, 2.5vw, 28px);
    font-weight: 800;
    margin: 6px 0;
    line-height: 1.2;
    text-align: center;
}
.smartstock-footer h2 { color:white; margin:6px 0; font-size:clamp(25px,3vw,34px); font-weight:800; }
.smartstock-footer .ss-desc { color:#d6e1f6; margin:5px 0; font-size:14px; line-height:1.55; }
.smartstock-footer .ss-divider { height:1px; background:rgba(255,255,255,.20); margin:25px 0; }
.smartstock-footer .ss-heading { text-align:center; font-size:21px; font-weight:750; margin:0 0 19px; color:white; }
.smartstock-footer .ss-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; }
.smartstock-footer .ss-member { display:flex; flex-direction:column; align-items:center; text-align:center; padding:20px 9px 17px; min-width:0; background:#17366e; border:1px solid #35548c; border-radius:14px; transition:transform .18s,border-color .18s; }
.smartstock-footer .ss-member:hover { transform:translateY(-3px); border-color:#d56477; }
.smartstock-footer .ss-photo { width:110px; height:110px; aspect-ratio:1; border-radius:50%; border:3px solid #fff; object-fit:cover; object-position: center 34%; margin:0 0 12px; background:#dce7f7; }
.smartstock-footer .ss-member h3 { color:#fff; font-size:15px; font-weight:700; line-height:1.3; margin:0 0 6px; }
.smartstock-footer a { text-decoration:none; }
.smartstock-footer a:hover { text-decoration:underline; }
.smartstock-footer .ss-email { color:#d7e6ff; font-size:12px; overflow-wrap:anywhere; line-height:1.5; min-height:35px; }
.smartstock-footer .ss-qr-frame { background:#fff; padding:8px; border-radius:8px; margin:10px 0 7px; line-height:0; }
.smartstock-footer .ss-qr { width:90px; height:90px; object-fit:contain; image-rendering:auto; }
.smartstock-footer .ss-linkedin { color:#fff; font-size:12px; font-weight:600; }
.smartstock-footer .ss-muted { color:#c4d2ec; font-weight:400; }
.smartstock-footer .ss-github { text-align:center; margin:22px auto 14px; max-width:750px; padding:21px 16px; border-radius:14px; background:#17366e; border:1px solid #35548c; }
.smartstock-footer .ss-github h3 { color:white; font-size:18px; margin:0 0 7px; }
.smartstock-footer .ss-github p { color:#d6e1f6; font-size:13px; line-height:1.5; margin:0 0 15px; }
.smartstock-footer .ss-button { display:inline-block; color:#fff; background:#c8102e; padding:11px 21px; border-radius:8px; font-size:13px; font-weight:750; }
.smartstock-footer .ss-button:hover { background:#a90d27; text-decoration:none; }
.smartstock-footer .ss-advisor { text-align:center; font-size:13px; line-height:1.8; color:#d6e1f6; }
.smartstock-footer .ss-advisor strong { color:white; }
.smartstock-footer .ss-advisor a { color:#fff; }
.smartstock-footer .ss-bottom { text-align:center; font-size:11px; line-height:1.7; color:#c4d2ec; }
@media(max-width:1100px){ .smartstock-footer .ss-grid { grid-template-columns:repeat(3,minmax(0,1fr)); } }
@media(max-width:650px){ .smartstock-footer { padding:24px 14px 18px; } .smartstock-footer .ss-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; } .smartstock-footer .ss-photo { width:92px;height:92px; } .smartstock-footer .ss-member { padding:15px 6px; } .smartstock-footer .ss-qr { width:82px;height:82px; } }
@media(max-width:380px){ .smartstock-footer .ss-grid { grid-template-columns:1fr; } }
@media(prefers-reduced-motion:reduce){ .smartstock-footer .ss-member { transition:none; } }
</style>
<section class="smartstock-footer" aria-label="SmartStock AI team and project information">
    <div class="ss-top">
    <h2 class="brand-title">IDS 560 Analytics Strategy and Practice</h2>
      <div class="ss-eyebrow">University of Illinois Chicago</div>
      <p class="ss-desc">Retrieval-Augmented Decision Support for Retail Inventory Replenishment</p>
      <p class="ss-desc">Integrating demand forecasting with evidence-grounded explanations</p>
    </div>
    <div class="ss-divider"></div>
    <h2 class="ss-heading">Meet Our Team</h2>
    <div class="ss-grid">__CARDS__</div>
    <div class="ss-github">
        <h3>Explore the Project on GitHub</h3>
        <p>Source code, demand forecasting, retrieval-augmented explanations, and the Streamlit dashboard</p>
        <a class="ss-button" href="__GITHUB__" target="_blank" rel="noopener noreferrer">View GitHub Repository ↗</a>
    </div>
    <div class="ss-divider"></div>
    <div class="ss-advisor">
        <div><strong>Faculty Advisor:</strong> __ADVISOR__ · <a href="mailto:__ADVISOR_EMAIL__">__ADVISOR_EMAIL__</a></div>
        <div><strong>University of Illinois Chicago</strong> · IDS 560 Analytics Capstone · Fall 2026</div>
    </div>
    <div class="ss-divider"></div>
    <div class="ss-bottom">© 2026 IDS 560 Group 4 · Academic Project · Built with Python, Streamlit, LightGBM &amp; RAG</div>
</section>'''
    html = (html.replace("__CARDS__", cards)
                .replace("__GITHUB__", github)
                .replace("__ADVISOR__", escape(ADVISOR_NAME))
                .replace("__ADVISOR_EMAIL__", advisor_email))
    st.html(html)
