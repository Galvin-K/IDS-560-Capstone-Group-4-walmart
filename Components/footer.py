"""Reusable IDS 560 Group 4 footer for Streamlit.

Call render_footer() at the end of app.py.
Requires a Streamlit version supporting st.html().
"""
from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT_DIR / "Assets" / "Teams"
UIC_LOGO_PATH = ROOT_DIR / "Assets" / "UIC_logo.png"

GITHUB_URL = "https://github.com/Galvin-K/IDS-560-Capstone-Group-4-walmart/"
ADVISOR_NAME = "Professor Fatemeh Sarayloo"
ADVISOR_EMAIL = "fsaraylo@uic.edu"

TEAM = [
    {"name": "Keta Patel", "email": "kpate650@uic.edu", "photo": "Keta Headshot.png", "qr": "Keta_QR code.jpeg", "linkedin": "https://www.linkedin.com/in/keta2002"},
    {"name": "Galvin Kao", "email": "hkao9@uic.edu", "photo": "Galvin Headshot.jpeg", "qr": "Galvin QR code.jpeg", "linkedin": "https://www.linkedin.com/in/galvin-kao"},
    {"name": "Tyler Yuen", "email": "tcyuen2@uic.edu", "photo": "Tyler Headshot.jpeg", "qr": "Tyler QR Code.png", "linkedin": ""},
    {"name": "Fan Mengxi", "email": "mfan20@uic.edu", "photo": "Mengxi Headshot.jpeg", "qr": "Mengxi QR code.jpeg", "linkedin": "https://www.linkedin.com/in/regina-fan-45a0b9146"},
    {"name": "Mengyi Wang", "email": "mwang233@uic.edu", "photo": "Mengyi Headshot.jpeg", "qr": "Mengyi QR Code.jpeg", "linkedin": "https://www.linkedin.com/in/carmen-wang-889334208"},
]


def _data_uri(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing footer asset: {path}")
    mime_by_suffix = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    mime = mime_by_suffix.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported image format: {path.name}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _safe_link(url: str) -> str:
    parsed = urlparse(url)
    return escape(url, quote=True) if parsed.scheme == "https" and parsed.netloc else ""


def _member_card(member: dict) -> str:
    name = escape(member["name"])
    email = escape(member["email"], quote=True)
    photo = _data_uri(ASSET_DIR / member["photo"])
    qr = _data_uri(ASSET_DIR / member["qr"])
    linkedin = _safe_link(member.get("linkedin", ""))

    linkedin_link = (
        f'<a href="{linkedin}" target="_blank" rel="noopener noreferrer">LinkedIn ↗</a>'
        if linkedin else '<span class="ss-muted">LinkedIn via QR</span>'
    )
    return f'''<article class="ss-member">
       <img class="ss-photo" src="{photo}" alt="Headshot of {name}" loading="lazy">
       <div class="ss-member-meta">
         <div class="ss-name">{name}</div>
         <a class="ss-email" href="mailto:{email}">{email}</a>
         <div class="ss-links">{linkedin_link}
           <details class="ss-qr-toggle"><summary>QR code</summary>
             <div class="ss-qr-box"><img src="{qr}" alt="LinkedIn QR code for {name}" loading="lazy"></div>
           </details>
         </div>
       </div>
    </article>'''


def render_footer() -> None:
    """Compact footer, with all original contacts and QR codes preserved."""
    cards = "\n".join(_member_card(member) for member in TEAM)
    uic_logo = _data_uri(UIC_LOGO_PATH)
    github = _safe_link(GITHUB_URL)
    advisor_email = escape(ADVISOR_EMAIL, quote=True)
    html = '''<style>
.smartstock-footer,.smartstock-footer *{box-sizing:border-box}
.smartstock-footer{
 width:100%;margin-top:6.5rem;padding:19px 22px 16px;background:#172e64;
 border-top:4px solid #ffc220;color:#f7f9ff;border-radius:12px;
 font-family:"Aptos","Segoe UI Variable","Segoe UI",system-ui,sans-serif;
}
.smartstock-footer a{color:#e7efff;text-decoration:none}
.smartstock-footer a:hover{text-decoration:underline}
.smartstock-footer .ss-topline{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap}
.smartstock-footer .ss-uic-logo{width:165px;max-width:100%;display:block;background:#fff;border-radius:5px;padding:4px}
.smartstock-footer .ss-project{flex:1;min-width:230px}
.smartstock-footer .ss-project strong{font-size:17px;font-weight:600;letter-spacing:-.01em;color:#fff}
.smartstock-footer .ss-project p{font-size:12px;line-height:1.45;margin:3px 0;color:#d5e2f9}
.smartstock-footer .ss-github-link{background:#ffc220;color:#142a55!important;border-radius:7px;font-size:12px;font-weight:600;padding:9px 13px;white-space:nowrap}
.smartstock-footer .ss-section-title{font-size:15px;font-weight:600;margin:14px 0 11px;color:#fff}
.smartstock-footer .ss-team{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px}
.smartstock-footer .ss-member{
 min-width:0;display:flex;flex-direction:column;align-items:center;gap:6px;text-align:center;
 border:1px solid rgba(230,240,255,.16);border-radius:10px;padding:11px 7px;
 background:rgba(255,255,255,.05);
}
.smartstock-footer .ss-photo{width:64px;height:64px;border:2px solid #d8e5fa;border-radius:50%;object-fit:cover;object-position:center 33%}
.smartstock-footer .ss-member-meta{min-width:0}
.smartstock-footer .ss-name{font-weight:600;font-size:12px;color:#fff}
.smartstock-footer .ss-email{font-size:10px;overflow-wrap:anywhere;display:block;line-height:1.4;margin:3px 0}
.smartstock-footer .ss-links{display:flex;align-items:center;justify-content:center;gap:9px;flex-wrap:wrap;font-size:10px;margin-top:5px}
.smartstock-footer .ss-qr-toggle{position:relative}
.smartstock-footer .ss-qr-toggle summary{cursor:pointer;color:#e9f0ff;list-style:none;text-decoration:underline;text-underline-offset:2px}
.smartstock-footer .ss-qr-toggle summary::-webkit-details-marker{display:none}
.smartstock-footer .ss-qr-box{margin-top:8px;background:#fff;padding:6px;border-radius:7px;line-height:0}
.smartstock-footer .ss-qr-box img{width:96px;height:96px;object-fit:contain}
.smartstock-footer .ss-muted{color:#cfddf4}
.smartstock-footer .ss-bottom{
 border-top:1px solid rgba(245,250,255,.17);margin-top:15px;padding-top:12px;
 display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;
 color:#cbd9ef;font-size:11px;line-height:1.45;
}
.smartstock-footer .ss-bottom a{color:#e3eeff}
@media(max-width:1020px){.smartstock-footer .ss-team{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:650px){.smartstock-footer{padding:16px 12px}.smartstock-footer .ss-team{grid-template-columns:repeat(2,minmax(0,1fr))}.smartstock-footer .ss-uic-logo{width:145px}}
@media(max-width:360px){.smartstock-footer .ss-team{grid-template-columns:1fr}}
</style>
<section class="smartstock-footer" aria-label="Project team and resources">
 <div class="ss-topline">
   <img class="ss-uic-logo" src="__UIC_LOGO__" alt="University of Illinois Chicago logo">
   <div class="ss-project">
     <strong>IDS 560 · Retail Inventory Decision Support</strong>
     <p>Historical Walmart M5 forecasting · Simulated replenishment · Evidence-grounded explanations</p>
   </div>
   <a class="ss-github-link" href="__GITHUB__" target="_blank" rel="noopener noreferrer">GitHub Repository ↗</a>
 </div>
 <div class="ss-section-title">Meet Our Team</div>
 <div class="ss-team">__CARDS__</div>
 <div class="ss-bottom">
   <span><strong>Faculty Advisor:</strong> __ADVISOR__ · <a href="mailto:__ADVISOR_EMAIL__">__ADVISOR_EMAIL__</a></span>
   <span>University of Illinois Chicago · IDS 560 · Fall 2026</span>
 </div>
</section>'''
    html = (html.replace("__UIC_LOGO__", uic_logo)
                .replace("__CARDS__", cards)
                .replace("__GITHUB__", github)
                .replace("__ADVISOR__", escape(ADVISOR_NAME))
                .replace("__ADVISOR_EMAIL__", advisor_email))
    st.html(html)
