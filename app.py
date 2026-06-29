import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_score, recall_score,
)

BASE_DIR = Path(__file__).parent
PIMA_PATH = BASE_DIR / "data" / "pima_diabetes.csv"
MAP_PATH  = BASE_DIR / "data" / "diabetes_global.csv"

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesIQ | Global Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME PALETTES ────────────────────────────────────────────────────────────
# Professional clinical palette: deep blue + slate neutrals + single indigo accent.
THEMES = {
    "dark": {
        "bg":         "#0e1525",   # app background
        "bg_sidebar": "#0a0f1c",   # sidebar background
        "surface":    "#161f33",   # card / panel background
        "surface_2":  "#1b2740",   # raised / input background
        "border":     "#24314d",   # hairline borders
        "border_2":   "#2e3e5e",   # stronger borders
        "text":       "#f1f5fb",   # primary text / values
        "text_muted": "#a7b6cc",   # body text
        "text_dim":   "#6f8199",   # captions / labels
        "accent":     "#4f7cff",   # primary indigo-blue accent
        "accent_2":   "#3a63d8",   # accent gradient end
        "accent_soft":"rgba(79,124,255,0.10)",
        "shadow":     "0 4px 28px rgba(0,0,0,0.40)",
        "grid":       "rgba(36,49,77,0.7)",
    },
    "light": {
        "bg":         "#f4f6fb",
        "bg_sidebar": "#ffffff",
        "surface":    "#ffffff",
        "surface_2":  "#f0f3f9",
        "border":     "#e2e8f2",
        "border_2":   "#cdd7e6",
        "text":       "#0f1b30",
        "text_muted": "#42536b",
        "text_dim":   "#7587a0",
        "accent":     "#3a63d8",
        "accent_2":   "#2a4fc0",
        "accent_soft":"rgba(58,99,216,0.08)",
        "shadow":     "0 2px 16px rgba(15,27,48,0.08)",
        "grid":       "rgba(205,215,230,0.9)",
    },
}

# Categorical / semantic chart colours (theme-independent, tuned to read on both bg).
ACCENT_SERIES = "#4f7cff"   # primary
SERIES_2      = "#f59e0b"   # amber (secondary categorical)
SERIES_3      = "#10b981"   # emerald
POSITIVE      = "#10b981"
RISK_HI       = "#ef4444"
RISK_MID      = "#f59e0b"


def get_theme():
    """Active theme dict, defaulting to dark."""
    mode = st.session_state.get("theme_mode", "dark")
    return THEMES.get(mode, THEMES["dark"])


def inject_css():
    """Theme-driven CSS using the active palette. Re-runs on every render so the
    sidebar light/dark toggle re-themes instantly."""
    t = get_theme()
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --bg:{t['bg']}; --bg-sidebar:{t['bg_sidebar']};
    --surface:{t['surface']}; --surface-2:{t['surface_2']};
    --border:{t['border']}; --border-2:{t['border_2']};
    --text:{t['text']}; --text-muted:{t['text_muted']}; --text-dim:{t['text_dim']};
    --accent:{t['accent']}; --accent-2:{t['accent_2']}; --accent-soft:{t['accent_soft']};
    --shadow:{t['shadow']}; --grid:{t['grid']};
}}

/* FOUNDATION */
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
section.main,.main .block-container {{
    background-color:var(--bg) !important;
    font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif !important;
}}
.block-container {{ padding:2.6rem 2.5rem 3rem 2.5rem !important; max-width:1480px !important; }}

/* HEADER / FOOTER */
header[data-testid="stHeader"],.stApp > header {{ background-color:var(--bg) !important; }}
#MainMenu,footer {{ visibility:hidden; }}

/* SIDEBAR */
[data-testid="stSidebar"] {{
    background:var(--bg-sidebar) !important;
    border-right:1px solid var(--border) !important;
}}
[data-testid="stSidebar"] label {{ color:var(--text-dim) !important; font-size:.84rem !important; font-weight:500 !important; }}
[data-testid="stSidebar"] span  {{ color:var(--text-dim) !important; }}
[data-testid="stSidebar"] p     {{ color:var(--text-dim) !important; font-size:.75rem !important; }}
[data-testid="stRadio"] > div > label {{ color:var(--text-muted) !important; font-size:.88rem !important; transition:color .15s; }}
[data-testid="stRadio"] > div > label:hover {{ color:var(--accent) !important; }}

/* TYPOGRAPHY */
.stApp h1 {{ color:var(--text) !important; font-weight:800 !important; letter-spacing:-.02em !important; }}
.stApp h2 {{ color:var(--text) !important; font-weight:700 !important; letter-spacing:-.01em !important; }}
.stApp h3 {{ color:var(--text-muted) !important; font-weight:600 !important; }}
.stApp h4,.stApp h5,.stApp h6 {{ color:var(--text-dim) !important; font-weight:500 !important; }}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{ color:var(--text-muted); line-height:1.7; font-size:.88rem; }}
[data-testid="stMarkdownContainer"] strong {{ color:var(--text) !important; }}
p,label {{ color:var(--text-muted) !important; }}

/* METRIC CARDS */
[data-testid="metric-container"],[data-testid="stMetric"] {{
    background:var(--surface) !important;
    border:1px solid var(--border) !important;
    border-left:3px solid var(--accent) !important;
    border-radius:12px !important;
    padding:16px 20px !important;
    box-shadow:var(--shadow) !important;
}}
[data-testid="stMetricLabel"] {{
    color:var(--text-dim) !important; font-size:.72rem !important;
    font-weight:600 !important; text-transform:uppercase !important; letter-spacing:.06em !important;
}}
[data-testid="stMetricValue"] {{
    color:var(--text) !important; font-size:1.75rem !important;
    font-weight:800 !important; letter-spacing:-.02em !important; line-height:1.15 !important;
}}
[data-testid="stMetricDelta"] {{ color:var(--text-dim) !important; font-size:.78rem !important; font-weight:500 !important; }}
[data-testid="stMetricDelta"] svg {{ display:none; }}

/* TABS */
[data-testid="stTabs"] {{ border-bottom:1px solid var(--border) !important; }}
[data-testid="stTabs"] button {{
    color:var(--text-dim) !important; font-size:.85rem !important; font-weight:500 !important;
    background:transparent !important; border:none !important;
    padding:.5rem 1.1rem !important; transition:color .15s !important;
}}
[data-testid="stTabs"] button:hover {{ color:var(--text-muted) !important; }}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color:var(--accent) !important; font-weight:700 !important;
    border-bottom:2px solid var(--accent) !important;
}}

/* SECTION HEADER */
div.sec-hdr {{
    display:flex; align-items:center; gap:10px;
    border-bottom:1px solid var(--border) !important;
    padding-bottom:11px !important; margin-bottom:1.4rem !important;
}}
div.sec-hdr .sec-hdr-txt {{
    color:var(--text) !important; font-size:1.22rem !important;
    font-weight:700 !important; letter-spacing:-.01em !important;
}}
div.sec-hdr svg {{ color:var(--accent); flex:0 0 auto; }}
div.sec-sub {{ color:var(--text-dim); font-size:.82rem; margin-top:-1rem; margin-bottom:1.4rem; }}

/* CALLOUT */
.callout {{
    display:flex; gap:11px; align-items:flex-start;
    background:var(--accent-soft); border-left:3px solid var(--accent);
    border-radius:0 8px 8px 0; padding:12px 16px; margin:14px 0;
    color:var(--text-muted); font-size:.86rem; line-height:1.6;
}}
.callout svg {{ color:var(--accent); flex:0 0 auto; margin-top:2px; }}

/* TYPE CARDS */
.type-card {{
    background:var(--surface);
    border:1px solid var(--border); border-radius:14px;
    padding:26px 20px; text-align:center; height:100%;
    transition:border-color .25s,transform .2s,box-shadow .2s;
    box-shadow:var(--shadow);
}}
.type-card:hover {{ border-color:var(--accent); transform:translateY(-3px); box-shadow:var(--shadow); }}
.type-card .tc-icon {{ color:var(--accent); margin-bottom:12px; display:inline-flex; }}
.type-card h3 {{ color:var(--text) !important; font-size:1.05rem !important; font-weight:700 !important; margin-bottom:4px; }}
.type-card .tc-pct {{ color:var(--accent) !important; font-size:1.9rem !important; font-weight:800 !important; margin:6px 0 10px; display:block; letter-spacing:-.02em; }}
.type-card p {{ color:var(--text-dim) !important; font-size:.82rem !important; line-height:1.6; }}

/* MISC */
hr {{ border-color:var(--border) !important; margin:1.4rem 0 !important; }}
[data-testid="stAlert"] {{ background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:10px !important; }}
[data-testid="stDataFrame"] {{ border-radius:10px !important; overflow:hidden; border:1px solid var(--border); }}
[data-testid="stNumberInput"] input {{ background:var(--surface-2) !important; color:var(--text) !important; border-color:var(--border-2) !important; border-radius:8px !important; }}
[data-testid="stSelectbox"] > div > div {{ background:var(--surface-2) !important; color:var(--text) !important; border-radius:8px !important; }}
.stButton > button {{
    background:linear-gradient(135deg,var(--accent),var(--accent-2)) !important;
    color:#ffffff !important; font-weight:700 !important; font-size:.88rem !important;
    border:none !important; border-radius:9px !important; letter-spacing:.02em !important;
    transition:opacity .15s,transform .15s,background .15s !important;
}}
.stButton > button:hover {{ opacity:.90 !important; transform:translateY(-1px) !important; }}

/* Secondary (outline) buttons in the sidebar — theme toggle & logout */
[data-testid="stSidebar"] .stButton > button {{
    background:transparent !important;
    color:var(--text-muted) !important;
    border:1px solid var(--border-2) !important;
    font-weight:600 !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    border-color:var(--accent) !important; color:var(--accent) !important;
    opacity:1 !important;
}}

/* THEME TOGGLE (segmented) */
[data-testid="stSidebar"] [role="radiogroup"][aria-label="theme"] {{ gap:0; }}
</style>
""", unsafe_allow_html=True)


PASSWORD = "diabetes2026"

# ── CHART COLOUR TOKENS ───────────────────────────────────────────────────────
# Semantic categorical colours (stable across themes).
TEAL    = ACCENT_SERIES   # primary series (kept name for minimal churn)
ORANGE  = SERIES_2        # secondary categorical
RED     = RISK_HI         # high-risk / diabetic
FONT_FAM = "Inter, -apple-system, sans-serif"

# Sequential scales for choropleth / importance bars.
ACCENT_SCALE = [[0, "#1e3a8a"], [0.5, "#3a63d8"], [1.0, "#7aa2ff"]]
TEAL_SCALE   = ACCENT_SCALE                       # kept name for minimal churn
RISK_SCALE   = [[0, "#1e3a8a"], [0.5, SERIES_2],  [1.0, RISK_HI]]


def _c():
    """Live theme-derived chart colours (background, text, grid)."""
    t = get_theme()
    return dict(
        paper=t["surface"], plot="rgba(0,0,0,0)",
        text=t["text"], text_muted=t["text_muted"], text_dim=t["text_dim"],
        grid=t["grid"], border=t["border"], accent=t["accent"], bg=t["bg"],
    )


def _fade(hex_color, alpha):
    """Return an rgba() string for a #rrggbb colour at the given opacity.
    Used to de-emphasise secondary marks (figure-ground / focal point)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Theme-aware neutral tokens used throughout the chart builders. Initialised to
# the dark palette and reassigned each render by refresh_theme_tokens() so the
# light/dark toggle re-colours every chart instantly.
BLUE_LT  = THEMES["dark"]["text"]        # bright text on bars
BLUE_MID = THEMES["dark"]["text_muted"]  # axis tick labels
BLUE_DIM = THEMES["dark"]["text_dim"]    # captions / faint labels
PAPER_BG = THEMES["dark"]["surface"]     # chart card background
CARD_BG  = THEMES["dark"]["surface"]
FONT_CLR = THEMES["dark"]["text"]
PLOT_BG  = "rgba(0,0,0,0)"
GRID_CLR = THEMES["dark"]["grid"]


def refresh_theme_tokens():
    """Sync the module-level chart tokens to the active theme. Call after
    inject_css() on every run."""
    global BLUE_LT, BLUE_MID, BLUE_DIM, PAPER_BG, CARD_BG, FONT_CLR, GRID_CLR
    t = get_theme()
    BLUE_LT  = t["text"]
    BLUE_MID = t["text_muted"]
    BLUE_DIM = t["text_dim"]
    PAPER_BG = t["surface"]
    CARD_BG  = t["surface"]
    FONT_CLR = t["text"]
    GRID_CLR = t["grid"]


# ── INLINE SVG ICONS (Lucide-style, currentColor stroke) ──────────────────────
_ICON_PATHS = {
    "overview":    '<path d="M3 12l9-9 9 9"/><path d="M5 10v10h14V10"/><path d="M9 20v-6h6v6"/>',
    "dashboard":   '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
    "globe":       '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a15 15 0 0 1 0 18"/><path d="M12 3a15 15 0 0 0 0 18"/>',
    "trends":      '<path d="M3 3v18h18"/><path d="M19 9l-5 5-4-4-4 4"/>',
    "users":       '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "risk":        '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "model":       '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
    "info":        '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "logo":        '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>',
    "logout":      '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="M16 17l5-5-5-5"/><path d="M21 12H9"/>',
    "lightbulb":   '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5.76.76 1.23 1.52 1.41 2.5"/>',
    "sun":         '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',
    "moon":        '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"/>',
    "bolt":        '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "drop":        '<path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>',
    "baby":        '<path d="M9 12h.01M15 12h.01M10 16c.5.3 1.2.5 2 .5s1.5-.2 2-.5"/><path d="M19 6.3a9 9 0 0 1 1.8 3.9 2 2 0 0 1 0 3.6 9 9 0 0 1-17.6 0 2 2 0 0 1 0-3.6A9 9 0 0 1 12 3c2 0 3.5 1.1 3.5 2.5s-.9 2.5-2 2.5c-.8 0-1.5-.4-1.5-1"/>',
    "stethoscope": '<path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6 6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/>',
}


def icon(name, size=18, stroke=2, cls=""):
    """Return an inline SVG string for the given icon, inheriting currentColor."""
    paths = _ICON_PATHS.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" '
        f'class="ic {cls}">{paths}</svg>'
    )


def _layout(h=360, t=44, b=36, l=10, r=10, grid="none"):
    """Base Plotly layout.

    grid: which axis keeps a faint reference grid. Default "none" — gridlines are
    clutter and we prefer direct value labels. Use "x"/"y" only when a position
    must be judged against a scale (line/scatter/ROC), never when bars are labeled.
    """
    c = _c()
    faint = _fade(c["text_dim"], 0.18)

    def axis(show_grid):
        return dict(
            showgrid=show_grid, gridcolor=faint, gridwidth=1,
            zerolinecolor="rgba(0,0,0,0)",
            linecolor="rgba(0,0,0,0)",
            tickcolor="rgba(0,0,0,0)",
            tickfont=dict(color=c["text_dim"], size=11, family=FONT_FAM),
        )
    return dict(
        paper_bgcolor=c["paper"], plot_bgcolor=c["plot"],
        font=dict(color=c["text_muted"], family=FONT_FAM, size=12),
        height=h, margin=dict(t=t, b=b, l=l, r=r),
        xaxis=axis(grid == "x"), yaxis=axis(grid == "y"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=c["border"], borderwidth=0,
            font=dict(color=c["text_muted"], size=11, family=FONT_FAM),
        ),
        hoverlabel=dict(
            bgcolor=c["paper"], bordercolor=c["accent"],
            font=dict(color=c["text"], size=12, family=FONT_FAM),
        ),
    )


# ── LOGIN ─────────────────────────────────────────────────────────────────────
def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        c1, c2, c3 = st.columns([1, 1.2, 1])
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='text-align:center;padding:44px 36px;
                        background:var(--surface);
                        border-radius:18px;border:1px solid var(--border);
                        box-shadow:var(--shadow);'>
              <div style='display:inline-flex;align-items:center;justify-content:center;
                          width:64px;height:64px;border-radius:16px;margin-bottom:16px;
                          background:var(--accent-soft);color:var(--accent);'>
                {icon('logo', 34, 1.8)}
              </div>
              <h1 style='color:var(--text);margin:0 0 6px;font-size:2rem;
                         font-weight:800;letter-spacing:-.03em;
                         font-family:Inter,-apple-system,sans-serif;'>DiabetesIQ</h1>
              <p style='color:var(--accent);font-size:.8rem;font-weight:600;letter-spacing:.1em;
                        text-transform:uppercase;margin-bottom:6px;'>
                Global Diabetes Intelligence Platform
              </p>
              <p style='color:var(--text-dim);font-size:.82rem;margin-bottom:0;'>
                MSBA382 — Healthcare Analytics · OSB 2026
              </p>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            pwd = st.text_input("Password", type="password",
                                placeholder="Enter dashboard password",
                                label_visibility="collapsed")
            if st.button("Login  →", use_container_width=True, type="primary"):
                if pwd == PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        st.stop()


# ── DATA LOADERS ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading country data…")
def load_map_data():
    try:
        import urllib.request, json
        url = ("https://api.worldbank.org/v2/country/all/indicator/"
               "SH.STA.DIAB.ZS?format=json&per_page=20000&date=2000:2024")
        with urllib.request.urlopen(url, timeout=20) as r:
            records = json.loads(r.read())[1]
        rows = []
        for rec in records:
            if (rec.get("value") is not None
                    and rec.get("countryiso3code")
                    and len(rec["countryiso3code"]) == 3):
                rows.append({
                    "Entity":     rec["country"]["value"],
                    "Code":       rec["countryiso3code"],
                    "Year":       int(rec["date"]),
                    "Prevalence": round(float(rec["value"]), 2),
                })
        df = pd.DataFrame(rows)
        return df[df["Year"].isin([2011, 2024])].reset_index(drop=True)
    except Exception:
        return pd.read_csv(MAP_PATH).query(
            "Year in [2011, 2024]").reset_index(drop=True)


@st.cache_data
def get_global_trend():
    return pd.DataFrame({
        "Year":      [2000, 2003, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2024],
        "Prevalence":[6.4,  7.1,  7.6,  8.0,  8.3,  8.3,  8.5,  8.8,  9.3, 10.5, 10.5],
        "Adults_M":  [151,  194,  246,  285,  366,  387,  415,  425,  463,  537,  589],
    })


@st.cache_data
def get_regional_data():
    return pd.DataFrame({
        "Region":      ["MENA","N. America & Caribbean","South-East Asia",
                        "Western Pacific","S. & Central America","Europe","Africa"],
        "Code":        ["MENA","NAC","SEA","WP","SACA","EUR","AFR"],
        "Prevalence":  [16.7, 12.9, 11.9, 11.6, 10.7,  8.4,  5.0],
        "Adults_M":    [  73,   51,   90,  206,   32,   61,   24],
        "Undiagnosed": [53.5, 20.1, 52.5, 51.6, 39.5, 33.5, 65.0],
    })


@st.cache_data
def get_demographics():
    ages = ["20-24","25-29","30-34","35-39","40-44","45-49",
            "50-54","55-59","60-64","65-69","70-74","75-79"]
    return pd.DataFrame({
        "AgeGroup": ages,
        "Male":  [2.1,3.2,5.0,8.1,12.4,16.8,20.3,22.5,23.8,24.6,25.1,25.3],
        "Female":[1.7,2.8,4.5,7.4,11.9,16.1,19.8,21.9,23.2,24.0,24.6,24.3],
    })


PIMA_FALLBACK_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

def _clean_pima(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan).fillna(df[c].median())
    df["Status"] = df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"})
    return df

@st.cache_data(show_spinner="Loading Pima dataset…")
def load_pima():
    if PIMA_PATH.exists():
        return _clean_pima(pd.read_csv(PIMA_PATH))
    try:
        return _clean_pima(pd.read_csv(PIMA_FALLBACK_URL))
    except Exception:
        return None


@st.cache_data(show_spinner="Training ML models…")
def get_trained_models():
    try:
        df = pd.read_csv(PIMA_PATH) if PIMA_PATH.exists() else pd.read_csv(PIMA_FALLBACK_URL)
        for c in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
            if c in df.columns:
                df[c] = df[c].replace(0, np.nan).fillna(df[c].median())
        feats = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                 "Insulin","BMI","DiabetesPedigreeFunction","Age"]
        X, y = df[feats], df["Outcome"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                   random_state=42, stratify=y)
        scaler = StandardScaler()
        Xs_tr, Xs_te = scaler.fit_transform(X_tr), scaler.transform(X_te)
        lr = LogisticRegression(max_iter=1000, random_state=42).fit(Xs_tr, y_tr)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
        def _m(model, Xte, yte):
            yp = model.predict(Xte)
            ypr = model.predict_proba(Xte)[:, 1]
            return dict(acc=accuracy_score(yte,yp), auc=roc_auc_score(yte,ypr),
                        prec=precision_score(yte,yp), rec=recall_score(yte,yp),
                        prob=ypr, pred=yp)
        imp = (pd.Series(rf.feature_importances_, index=feats)
               .sort_values(ascending=False).reset_index())
        imp.columns = ["Feature","Importance"]
        return dict(lr=lr, rf=rf, scaler=scaler, feats=feats,
                    Xte=X_te, Xte_sc=Xs_te, yte=y_te,
                    lr_m=_m(lr,Xs_te,y_te), rf_m=_m(rf,X_te,y_te), imp=imp)
    except Exception:
        return None


# ── UI HELPERS ────────────────────────────────────────────────────────────────
def sec(title, sub="", icon_name=None):
    ico = f"<span class='sec-hdr-ico'>{icon(icon_name, 22, 2)}</span>" if icon_name else ""
    st.markdown(
        f"<div class='sec-hdr'>{ico}<span class='sec-hdr-txt'>{title}</span></div>",
        unsafe_allow_html=True)
    if sub:
        st.markdown(f"<div class='sec-sub'>{sub}</div>", unsafe_allow_html=True)

def callout(text):
    st.markdown(
        f"<div class='callout'>{icon('lightbulb', 17, 2)}<span>{text}</span></div>",
        unsafe_allow_html=True)

def no_pima():
    st.info("**Pima dataset not loaded.**  \n"
            "Save `diabetes.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) "
            "as `data/pima_diabetes.csv` and refresh.", icon="⚠️")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_global_trend(trend, h=340):
    fig = go.Figure()
    # Shaded fill first (draws behind the line)
    fig.add_trace(go.Scatter(
        x=trend["Year"], y=trend["Prevalence"],
        mode="lines+markers",
        line=dict(color=TEAL, width=2.5),
        marker=dict(size=7, color=TEAL, line=dict(color=PAPER_BG, width=2)),
        fill="tozeroy", fillcolor="rgba(79,124,255,0.08)",
        hovertemplate="<b>%{x}</b><br>Prevalence: <b>%{y:.1f}%</b><extra></extra>",
        name="Global avg.",
    ))
    # Annotation on last point
    last = trend.iloc[-1]
    fig.add_annotation(x=last["Year"], y=last["Prevalence"],
                       text=f"<b>{last['Prevalence']}%</b>",
                       showarrow=True, arrowhead=0, arrowcolor=TEAL,
                       font=dict(color=TEAL, size=12), ax=30, ay=-24)
    fig.update_layout(
        title=dict(text="Global Diabetes Prevalence — IDF Atlas 2000–2024",
                   font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
        xaxis_title="", yaxis_title="Prevalence (%)",
        showlegend=False,
        **_layout(h, grid="y"),   # faint horizontal grid only — judge the % level
    )
    return fig


def fig_adults_bar(trend, h=320):
    """Adults with diabetes, in millions, per IDF Atlas edition.

    Data-viz choices:
      • Categorical x with every year ticked → labels align under their own bar.
      • Direct value labels on each bar → the y-axis is removed (no double lookup).
      • Single flat colour (Gestalt: similarity = one series); endpoints 2000 &
        2024 emphasised to carry the "151M → 589M" story (focal point).
      • No gridlines, no title chrome → maximise data-ink ratio.
    """
    c = _c()
    df = trend.copy()
    years = df["Year"].astype(str).tolist()
    first, last = years[0], years[-1]
    # Mute the in-between bars; highlight the two endpoints that tell the story.
    bar_colors = [
        c["accent"] if y in (first, last) else _fade(c["accent"], 0.40)
        for y in years
    ]
    txt_colors = [
        c["text"] if y in (first, last) else c["text_dim"]
        for y in years
    ]
    fig = go.Figure(go.Bar(
        x=years, y=df["Adults_M"],
        text=[f"{v}M" for v in df["Adults_M"]],
        textposition="outside",
        textfont=dict(color=txt_colors, size=10, family=FONT_FAM),
        marker=dict(color=bar_colors, line_width=0),
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>Adults with diabetes: <b>%{y}M</b><extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Number of Adults with Diabetes (Millions)",
                   font=dict(color=c["text_muted"], size=13, family=FONT_FAM)),
        bargap=0.32,
        **_layout(h, l=10, r=10, b=30),
    )
    # Align every label under its bar; strip the redundant y-axis entirely.
    fig.update_xaxes(type="category", tickmode="array", tickvals=years,
                     ticktext=years, gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(color=c["text_dim"], size=10))
    fig.update_yaxes(visible=False, range=[0, df["Adults_M"].max() * 1.14])
    return fig


def fig_region_bar(reg, h=340):
    df = reg.sort_values("Prevalence")
    fig = px.bar(
        df, x="Prevalence", y="Code", orientation="h",
        text="Prevalence", color="Prevalence",
        color_continuous_scale=RISK_SCALE,
        labels={"Prevalence":"","Code":""},
        custom_data=["Region"],
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%", textposition="outside",
        textfont=dict(color=BLUE_LT, size=11, family=FONT_FAM),
        marker_line_width=0,
        hovertemplate="<b>%{customdata[0]}</b><br>Prevalence: <b>%{x:.1f}%</b><extra></extra>",
    )
    fig.update_layout(
        title=dict(text="Prevalence by IDF Region — 2024",
                   font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
        coloraxis_showscale=False,
        **_layout(h, l=70, r=55),
    )
    # Bars are directly labeled → the value axis is redundant; hide it.
    fig.update_xaxes(visible=False, range=[0, df["Prevalence"].max() * 1.15])
    fig.update_yaxes(tickfont=dict(color=BLUE_MID, size=11))
    return fig


def fig_world_map(map_df, year=2024, h=460):
    df = map_df[map_df["Year"] == year].copy()
    df["Rank"] = df["Prevalence"].rank(ascending=False).astype(int)
    fig = px.choropleth(
        df, locations="Code", color="Prevalence",
        hover_name="Entity",
        hover_data={"Prevalence": ":.1f", "Rank": True, "Code": False},
        color_continuous_scale=[
            [0.0,"#1e3a8a"],[0.25,"#3a63d8"],[0.5,"#7aa2ff"],
            [0.72,SERIES_2],[1.0,RISK_HI],
        ],
        range_color=[float(df["Prevalence"].min()),
                     float(df["Prevalence"].quantile(0.97))],
        labels={"Prevalence":"Prevalence (%)"},
    )
    c = _c()
    land = get_theme()["surface_2"]
    fig.update_traces(marker_line_width=0.4, marker_line_color=c["bg"])
    fig.update_layout(
        paper_bgcolor=c["bg"],
        geo=dict(
            bgcolor=c["bg"], lakecolor=c["bg"],
            showframe=False, showcoastlines=False,
            landcolor=land, showocean=True, oceancolor=c["bg"],
            showcountries=True, countrycolor=c["bg"],
            showlakes=False,
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Prevalence (%)", font=dict(color=c["text_muted"], size=11)),
            tickfont=dict(color=c["text_dim"], size=10),
            thickness=10, len=0.55, x=1.01,
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=0, b=0, l=0, r=0), height=h,
    )
    return fig


def fig_age_gender(demo, h=340):
    fig = go.Figure([
        go.Bar(name="Male", x=demo["AgeGroup"], y=demo["Male"],
               marker=dict(color=TEAL, line=dict(width=0)),
               hovertemplate="<b>Age %{x}</b><br>Male: <b>%{y:.1f}%</b><extra></extra>"),
        go.Bar(name="Female", x=demo["AgeGroup"], y=demo["Female"],
               marker=dict(color=ORANGE, line=dict(width=0)),
               hovertemplate="<b>Age %{x}</b><br>Female: <b>%{y:.1f}%</b><extra></extra>"),
    ])
    fig.update_layout(
        barmode="group", bargap=0.28, bargroupgap=0.06,
        title=dict(text="Prevalence by Age Group & Gender — 2024",
                   font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
        yaxis_title="Prevalence (%)",
        **_layout(h, grid="y"),   # 24 grouped bars → reference grid beats clutter
    )
    fig.update_xaxes(tickfont=dict(color=BLUE_DIM, size=10))
    return fig


def fig_type_donut(h=320):
    fig = go.Figure(go.Pie(
        labels=["Type 2", "Type 1", "Gestational", "Other"],
        values=[92, 6, 3, 2], hole=0.58,
        marker=dict(colors=[RED, TEAL, ORANGE, "#6a4c93"],
                    line=dict(color=PAPER_BG, width=2)),
        textinfo="percent",
        textposition="inside",
        insidetextorientation="horizontal",
        textfont=dict(size=12, color="#ffffff", family=FONT_FAM),
        pull=[0.04, 0, 0, 0],
        hovertemplate="<b>%{label}</b><br>Share: <b>%{percent}</b><extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Global Diabetes Type Distribution",
                   font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
        paper_bgcolor=PAPER_BG, height=h,
        margin=dict(t=50, b=10, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=BLUE_MID, size=11),
                    orientation="v", x=1.02, xanchor="left"),
        annotations=[dict(text="<b>Type</b>", x=0.5, y=0.5,
                          font=dict(size=14, color=BLUE_MID, family=FONT_FAM),
                          showarrow=False)],
    )
    return fig


def fig_scatter(df, xf, yf, h=380):
    fig = px.scatter(df, x=xf, y=yf, color="Status",
                     color_discrete_map={"No Diabetes": TEAL, "Diabetes": RED},
                     opacity=0.55, labels={"Status":""},
                     custom_data=["Age","Status"])
    fig.update_traces(
        marker=dict(size=5, line=dict(width=0)),
        hovertemplate=(
            f"<b>{xf}: %{{x:.0f}}</b><br>{yf}: %{{y:.1f}}<br>"
            "Age: %{customdata[0]:.0f}<br>Status: <b>%{customdata[1]}</b><extra></extra>"
        ),
    )
    if xf == "Glucose":
        fig.add_vline(x=140, line_dash="dot", line_color=ORANGE, line_width=1.5,
                      annotation_text="Threshold: 140 mg/dL",
                      annotation_font=dict(color=ORANGE, size=10),
                      annotation_position="top right")
    fig.update_layout(
        title=dict(text=f"{xf} vs {yf} — Diabetic vs Non-Diabetic",
                   font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
        **_layout(h, grid="y"),   # scatter: keep a faint grid to read positions
    )
    return fig


def fig_feature_importance(imp_df, h=340):
    feat_lbl = {
        "Glucose":"Plasma Glucose","BMI":"Body Mass Index","Age":"Age",
        "DiabetesPedigreeFunction":"Pedigree Fn.","Pregnancies":"Pregnancies",
        "Insulin":"Serum Insulin","SkinThickness":"Skin Thickness",
        "BloodPressure":"Blood Pressure",
    }
    df = imp_df.copy()
    df["Feature"] = df["Feature"].map(feat_lbl)
    fig = px.bar(df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale=TEAL_SCALE,
                 text="Importance", labels={"Importance":"","Feature":""},
                 custom_data=["Feature"])
    fig.update_traces(
        texttemplate="%{text:.3f}", textposition="outside",
        textfont=dict(color=BLUE_MID, size=10, family=FONT_FAM),
        marker_line_width=0,
        hovertemplate="<b>%{customdata[0]}</b><br>Importance: <b>%{x:.4f}</b><extra></extra>",
    )
    fig.update_layout(
        title=dict(text="Feature Importance — Random Forest",
                   font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
        yaxis_autorange="reversed", coloraxis_showscale=False,
        **_layout(h, l=130, r=60),
    )
    # Bars carry their importance value → drop the redundant x-axis.
    fig.update_xaxes(visible=False, range=[0, df["Importance"].max() * 1.18])
    fig.update_yaxes(tickfont=dict(color=BLUE_MID, size=11))
    return fig


def fig_roc(md, h=340):
    fig = go.Figure()
    # Diagonal reference
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="rgba(90,122,144,0.35)", dash="dot", width=1.5))
    for name, prob, color in [
        ("Logistic Regression", md["lr_m"]["prob"], ORANGE),
        ("Random Forest",       md["rf_m"]["prob"], TEAL),
    ]:
        fpr, tpr, _ = roc_curve(md["yte"], prob)
        auc = roc_auc_score(md["yte"], prob)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{name}  AUC = {auc:.3f}",
            mode="lines", line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=f"rgba{tuple(list(bytes.fromhex(color[1:])) + [15])}",
            hovertemplate="FPR: %{x:.2f}<br>TPR: <b>%{y:.2f}</b><extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="ROC Curve — LR vs Random Forest",
                   font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        **_layout(h, grid="y"),   # diagnostic plot: grid aids reading the tradeoff
    )
    return fig


def fig_corr(df, h=400):
    feats = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
             "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    short = ["Preg","Gluc","BP","Skin","Ins","BMI","DPF","Age","Out"]
    corr = df[feats].corr()
    th = get_theme()
    mid = "#eef2f8" if st.session_state.get("theme_mode") == "light" else "#1a2740"
    cell_txt = th["text"]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=short, y=short,
        colorscale=[[0,RED],[0.5,mid],[1,TEAL]],
        zmid=0, texttemplate="%{z:.2f}",
        textfont=dict(size=10, color=cell_txt, family=FONT_FAM),
        colorbar=dict(
            tickfont=dict(color=BLUE_DIM, size=10),
            title=dict(text="r", font=dict(color=BLUE_MID, size=11)),
            thickness=10, len=0.7,
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
        ),
        hovertemplate="<b>%{y} × %{x}</b><br>r = <b>%{z:.3f}</b><extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Clinical Feature Correlation Matrix",
                   font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
        paper_bgcolor=PAPER_BG, font=dict(color=FONT_CLR, family=FONT_FAM),
        height=h, margin=dict(t=50, b=10, l=10, r=10),
        xaxis=dict(tickfont=dict(color=BLUE_MID, size=10), side="bottom"),
        yaxis=dict(tickfont=dict(color=BLUE_MID, size=10), autorange="reversed"),
    )
    return fig


def fig_confusion(md, h=300):
    cm = confusion_matrix(md["yte"], md["rf_m"]["pred"])
    labels = ["No Diabetes","Diabetes"]
    fig = px.imshow(
        cm, x=labels, y=labels,
        color_continuous_scale=[[0,CARD_BG],[1,TEAL]],
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_traces(textfont=dict(size=24, color="#ffffff", family=FONT_FAM),
                      hovertemplate="Actual: <b>%{y}</b><br>Predicted: <b>%{x}</b><br>Count: <b>%{z}</b><extra></extra>")
    fig.update_layout(
        paper_bgcolor=PAPER_BG, height=h,
        margin=dict(t=30,b=30,l=10,r=10),
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(color=BLUE_MID, size=12)),
        yaxis=dict(tickfont=dict(color=BLUE_MID, size=12)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def page_overview(trend, regional):
    sec("Global Diabetes Overview",
        "Key burden statistics — IDF Diabetes Atlas 2024 (11th Edition)",
        icon_name="overview")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Adults with Diabetes",  "537 Million",  delta="1 in 10 adults globally")
    k2.metric("Global Prevalence",     "10.5%",        delta="+4.1 pp since 2000")
    k3.metric("Deaths per Year",       "6.7 Million",  delta="1 death every 5 seconds")
    k4.metric("Healthcare Spend",      "$966 Billion", delta="+316B vs 2015")
    st.markdown("<br>", unsafe_allow_html=True)
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Undiagnosed",       "240 Million",  delta="1 in 2 unaware")
    k6.metric("Projected by 2045", "828 Million",  delta="+54% from today")
    k7.metric("Children with T1D", "1.2 Million",  delta="Under age 20")
    k8.metric("MENA Prevalence",   "16.7%",        delta="Highest region globally")

    st.markdown("<br>", unsafe_allow_html=True)
    callout("The Middle East & North Africa (MENA) region carries the highest diabetes "
            "burden globally at 16.7% — nearly double Africa's rate (5.0%).")

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Understanding Diabetes Types", icon_name="info")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='type-card'>
          <span class='tc-icon'>{icon('bolt', 30, 1.8)}</span>
          <h3>Type 1</h3>
          <span class='tc-pct'>~6%</span>
          <p>Autoimmune — body produces no insulin.<br>Diagnosed in childhood or young adulthood.<br>Requires lifelong insulin therapy.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='type-card'>
          <span class='tc-icon'>{icon('drop', 30, 1.8)}</span>
          <h3>Type 2</h3>
          <span class='tc-pct'>~92%</span>
          <p>Insulin resistance or insufficient production.<br>Strongly linked to obesity and sedentary lifestyle.<br>Largely preventable and manageable.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='type-card'>
          <span class='tc-icon'>{icon('baby', 30, 1.8)}</span>
          <h3>Gestational</h3>
          <span class='tc-pct'>~3–5%</span>
          <p>Develops during pregnancy when insulin demand rises.<br>Usually resolves after delivery but raises future Type 2 risk for mother and child.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_global_trend(trend, 320), use_container_width=True)
    with c2:
        st.plotly_chart(fig_region_bar(regional, 320), use_container_width=True)


def page_dashboard(map_df, trend, regional, demo, pima):
    sec("Analytics Dashboard", "All key visualizations at a glance",
        icon_name="dashboard")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Adults with Diabetes", "537M",   delta="1 in 10 adults")
    k2.metric("Global Prevalence",    "10.5%",  delta="+4.1 pp since 2000")
    k3.metric("Deaths / Year",        "6.7M",   delta="1 every 5 seconds")
    k4.metric("Undiagnosed",          "240M",   delta="~1 in 2 unaware")
    k5.metric("MENA Rate",            "16.7%",  delta="Highest region")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='display:flex;align-items:center;gap:8px;color:{BLUE_DIM};"
        f"font-size:.8rem;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:.07em;margin-bottom:4px;'>{icon('globe', 16, 2)}"
        f"<span>GLOBAL DIABETES PREVALENCE — 2024 (% OF ADULTS 20–79)</span></p>",
        unsafe_allow_html=True)
    if not map_df.empty:
        st.plotly_chart(fig_world_map(map_df, year=2024, h=440),
                        use_container_width=True)
    else:
        st.warning("Map data unavailable.")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_global_trend(trend, 320), use_container_width=True)
    with c2:
        st.plotly_chart(fig_region_bar(regional, 320), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(fig_age_gender(demo, 320), use_container_width=True)
    with c4:
        st.plotly_chart(fig_type_donut(320), use_container_width=True)

    if pima is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(fig_scatter(pima, "Glucose", "BMI", 320),
                            use_container_width=True)
        with c6:
            md = get_trained_models()
            if md:
                st.plotly_chart(fig_feature_importance(md["imp"], 320),
                                use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c7, c8 = st.columns(2)
        with c7:
            st.plotly_chart(fig_corr(pima, 360), use_container_width=True)
        with c8:
            md = get_trained_models()
            if md:
                st.plotly_chart(fig_roc(md, 360), use_container_width=True)
    else:
        no_pima()


def page_global_map(map_df):
    sec("Global Geographic Distribution",
        "Diabetes prevalence (% adults 20–79) — World Bank / IDF Atlas",
        icon_name="globe")

    if map_df.empty:
        st.warning("Map data could not be loaded.")
        return

    available = sorted(map_df["Year"].unique().tolist(), reverse=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        sel_yr = st.radio("Year", available, horizontal=True, format_func=str)
    with c2:
        n = len(map_df[map_df["Year"] == sel_yr])
        st.metric("Countries", n, delta=f"data points")

    st.plotly_chart(fig_world_map(map_df, year=sel_yr, h=500),
                    use_container_width=True)
    callout(f"In {sel_yr}, MENA and Pacific Island nations show the highest prevalence. "
            "Pakistan leads at 31.4% (2024). Hover over any country for its exact rate.")

    df_yr = map_df[map_df["Year"] == sel_yr].copy()
    df_yr["Rank"] = df_yr["Prevalence"].rank(ascending=False).astype(int)

    st.markdown("#### Top 20 Countries by Prevalence")
    top20 = df_yr.nlargest(20, "Prevalence").sort_values("Prevalence")
    fig_top = px.bar(
        top20, x="Prevalence", y="Entity", orientation="h",
        color="Prevalence", color_continuous_scale=RISK_SCALE,
        text="Prevalence", labels={"Prevalence":"Prevalence (%)","Entity":""},
        custom_data=["Entity","Rank"],
    )
    fig_top.update_traces(
        texttemplate="%{text:.1f}%", textposition="outside",
        textfont=dict(color=BLUE_MID, size=10, family=FONT_FAM),
        marker_line_width=0,
        hovertemplate="<b>%{customdata[0]}</b><br>Rank #%{customdata[1]}<br>Prevalence: <b>%{x:.1f}%</b><extra></extra>",
    )
    fig_top.update_layout(
        coloraxis_showscale=False,
        **_layout(520, l=140, r=60),
    )
    fig_top.update_xaxes(visible=False, range=[0, top20["Prevalence"].max() * 1.15])
    fig_top.update_yaxes(tickfont=dict(color=BLUE_MID, size=10))
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.plotly_chart(fig_top, use_container_width=True)
    with c2:
        tbl = (df_yr.nlargest(20, "Prevalence").reset_index(drop=True)
               [["Rank","Entity","Prevalence"]]
               .rename(columns={"Entity":"Country","Prevalence":"Rate (%)"}))
        tbl["Rate (%)"] = tbl["Rate (%)"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=520)


def page_trends(map_df, trend, regional):
    sec("Trends Over Time",
        "Global trajectory and country comparisons — IDF Atlas 2000–2024",
        icon_name="trends")

    tab1, tab2, tab3 = st.tabs([
        "Global Trend", "Country Comparison (2011 vs 2024)", "Regional Breakdown",
    ])

    with tab1:
        st.plotly_chart(fig_global_trend(trend, 360), use_container_width=True)
        st.plotly_chart(fig_adults_bar(trend, 320), use_container_width=True)
        callout("Global cases nearly quadrupled from 151M (2000) to 589M (2024), "
                "driven by population growth, ageing, and rising obesity rates.")

    with tab2:
        if map_df.empty:
            st.warning("Country data unavailable.")
        else:
            codes_both = (map_df.groupby("Code")["Year"].nunique()
                          .loc[lambda s: s == 2].index.tolist())
            df_both = map_df[map_df["Code"].isin(codes_both)].copy()
            all_ctry = sorted(df_both["Entity"].unique().tolist())
            defaults = [c for c in ["Pakistan","Kuwait","Lebanon","Saudi Arabia",
                                    "United States","France","China","India"]
                        if c in all_ctry][:8]
            selected = st.multiselect("Select countries to compare",
                                      all_ctry, default=defaults)
            if not selected:
                st.info("Select at least one country to see the comparison.")
            else:
                sub = df_both[df_both["Entity"].isin(selected)].copy()
                pivot = (sub.pivot(index="Entity", columns="Year", values="Prevalence")
                         .reset_index())
                sub["Year_str"] = sub["Year"].astype(str)
                order = (pivot.assign(chg=lambda d: d[2024]-d[2011])
                         .sort_values("chg", ascending=False)["Entity"].tolist())
                sub["Entity"] = pd.Categorical(sub["Entity"], categories=order, ordered=True)
                sub = sub.sort_values("Entity")
                fig_cmp = px.bar(
                    sub, x="Entity", y="Prevalence",
                    color="Year_str", barmode="group",
                    color_discrete_map={"2011":"#1a5c78","2024":RED},
                    text="Prevalence",
                    labels={"Prevalence":"Prevalence (%)","Entity":"","Year_str":"Year"},
                )
                fig_cmp.update_traces(
                    texttemplate="%{text:.1f}%", textposition="outside",
                    textfont=dict(color=BLUE_DIM, size=10, family=FONT_FAM),
                    marker_line_width=0,
                )
                fig_cmp.update_layout(
                    title=dict(text="Diabetes Prevalence: 2011 vs 2024",
                               font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
                    legend_title_text="Year",
                    bargap=0.30, bargroupgap=0.08,
                    **_layout(400, b=44))
                # Every bar is labeled → hide the redundant y value-axis.
                fig_cmp.update_yaxes(visible=False,
                                     range=[0, sub["Prevalence"].max() * 1.18])
                fig_cmp.update_xaxes(tickangle=-35,
                                     tickfont=dict(color=BLUE_MID, size=10))
                st.plotly_chart(fig_cmp, use_container_width=True)
                if 2011 in pivot.columns and 2024 in pivot.columns:
                    pivot["Change (pp)"] = (pivot[2024] - pivot[2011]).round(1)
                    tbl = pivot[["Entity", 2011, 2024, "Change (pp)"]].copy()
                    tbl.columns = ["Country","2011 (%)","2024 (%)","Change (pp)"]
                    st.dataframe(tbl.sort_values("Change (pp)", ascending=False),
                                 use_container_width=True, hide_index=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_region_bar(regional, 360), use_container_width=True)
        with c2:
            un = regional.sort_values("Undiagnosed", ascending=False)
            fig_u = px.bar(
                un, x="Undiagnosed", y="Code", orientation="h",
                text="Undiagnosed", color="Undiagnosed",
                color_continuous_scale=RISK_SCALE,
                labels={"Undiagnosed":"","Code":""},
                custom_data=["Region"],
            )
            fig_u.update_traces(
                texttemplate="%{text:.1f}%", textposition="outside",
                textfont=dict(color=BLUE_MID, size=11, family=FONT_FAM),
                marker_line_width=0,
                hovertemplate="<b>%{customdata[0]}</b><br>Undiagnosed: <b>%{x:.1f}%</b><extra></extra>",
            )
            fig_u.update_layout(
                title=dict(text="Undiagnosed Rate by Region — 2024",
                           font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
                coloraxis_showscale=False,
                **_layout(360, l=60, r=55))
            fig_u.update_xaxes(visible=False, range=[0, un["Undiagnosed"].max() * 1.15])
            fig_u.update_yaxes(tickfont=dict(color=BLUE_MID, size=11))
            st.plotly_chart(fig_u, use_container_width=True)
        callout("Africa has the highest undiagnosed rate (65%) — 2 in 3 diabetics "
                "remain unaware. North America has the lowest at 20%.")


def page_demographics(demo, pima):
    sec("Demographics Analysis",
        "Distribution by age group, gender, and diabetes type — IDF Atlas 2024",
        icon_name="users")

    tab1, tab2, tab3, tab4 = st.tabs([
        "By Age Group", "By Gender", "Diabetes Type", "Pima Dataset Profile",
    ])

    with tab1:
        st.plotly_chart(fig_age_gender(demo, 420), use_container_width=True)
        callout("Prevalence rises steeply from ~2% at age 20 to ~25% at age 75. "
                "Ages 40–60 represent the critical prevention window.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Pie(
                labels=["Male", "Female"],
                values=[11.3, 10.9], hole=0.6,
                marker=dict(colors=[TEAL, ORANGE],
                            line=dict(color=PAPER_BG, width=2)),
                textinfo="label+percent",
                textfont=dict(size=13, color="#ffffff", family=FONT_FAM),
                hovertemplate="<b>%{label}</b><br>Prevalence: <b>%{value:.1f}%</b><extra></extra>",
            ))
            fig.update_layout(
                title=dict(text="Global Prevalence by Gender — 2024",
                           font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
                paper_bgcolor=PAPER_BG, height=320,
                margin=dict(t=50,b=10,l=10,r=10), showlegend=False,
                annotations=[dict(text="<b>Gender</b>", x=0.5, y=0.5,
                                  font=dict(size=14, color=BLUE_MID, family=FONT_FAM),
                                  showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            abs_df = pd.DataFrame({"Gender":["Male","Female"],"Adults (M)":[278,259]})
            fig2 = px.bar(abs_df, x="Gender", y="Adults (M)", color="Gender",
                          color_discrete_map={"Male":TEAL,"Female":ORANGE},
                          text="Adults (M)", labels={"Adults (M)":"Adults (Millions)","Gender":""})
            fig2.update_traces(
                texttemplate="%{text}M", textposition="outside",
                textfont=dict(color=BLUE_MID, size=12, family=FONT_FAM),
                marker_line_width=0,
            )
            fig2.update_layout(
                title=dict(text="Adults with Diabetes by Gender",
                           font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
                showlegend=False, **_layout(320, t=50))
            st.plotly_chart(fig2, use_container_width=True)
        callout("9.8M more men than women have diabetes globally. "
                "In MENA, female burden is disproportionately higher.")

    with tab3:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.plotly_chart(fig_type_donut(380), use_container_width=True)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            tbl = pd.DataFrame({
                "Type":           ["Type 2 (T2D)","Type 1 (T1D)","Gestational","Other"],
                "Share":          ["~92%","~6%","~3%","~1-2%"],
                "Typical Onset":  ["Adult 30+","Childhood","Pregnancy","Variable"],
                "Insulin Needed": ["Sometimes","Always","Sometimes","Variable"],
                "Preventable":    ["Largely yes","No","Partially","No"],
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            callout("Type 2 is the dominant subtype — largely preventable through "
                    "diet, exercise, and weight management.")

    with tab4:
        if pima is None:
            no_pima()
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(pima, x="Age", color="Status",
                                   barmode="overlay", nbins=25, opacity=0.7,
                                   color_discrete_map={"No Diabetes":TEAL,"Diabetes":RED},
                                   labels={"Age":"Age (years)","Status":""})
                fig.update_traces(marker_line_width=0)
                fig.update_layout(
                    title=dict(text="Age Distribution by Diabetes Status",
                               font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
                    legend_title_text="",
                    **_layout(360))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                summary = pd.DataFrame({
                    "Metric": ["Total patients","With diabetes","Without diabetes",
                               "Mean age — diabetic","Mean age — non-diabetic",
                               "Mean glucose — diabetic","Mean BMI — diabetic"],
                    "Value": [
                        len(pima), int(pima["Outcome"].sum()),
                        int((pima["Outcome"]==0).sum()),
                        f"{pima[pima['Outcome']==1]['Age'].mean():.1f} yrs",
                        f"{pima[pima['Outcome']==0]['Age'].mean():.1f} yrs",
                        f"{pima[pima['Outcome']==1]['Glucose'].mean():.0f} mg/dL",
                        f"{pima[pima['Outcome']==1]['BMI'].mean():.1f} kg/m²",
                    ],
                })
                st.markdown("**Dataset Profile**")
                st.dataframe(summary, use_container_width=True, hide_index=True, height=290)


def page_risk_factors(pima):
    sec("Risk Factors Analysis",
        "Clinical indicators — Pima Indians Diabetes Dataset (N=768)",
        icon_name="risk")

    if pima is None:
        no_pima()
        return

    feats = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
             "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    feat_lbl = {
        "Glucose":"Plasma Glucose (mg/dL)","BMI":"BMI (kg/m²)","Age":"Age (years)",
        "Insulin":"Serum Insulin (µU/mL)","BloodPressure":"Blood Pressure (mmHg)",
        "SkinThickness":"Skin Thickness (mm)","DiabetesPedigreeFunction":"Pedigree Fn.",
        "Pregnancies":"Pregnancies",
    }

    tab1, tab2, tab3 = st.tabs([
        "Correlation Heatmap", "Feature Distributions", "Scatter Analysis",
    ])

    with tab1:
        st.plotly_chart(fig_corr(pima, 460), use_container_width=True)
        callout("Glucose has the strongest correlation with diabetes outcome (r = 0.49), "
                "followed by BMI (r = 0.29) and Age (r = 0.24).")

    with tab2:
        sel = st.selectbox("Feature", feats, format_func=lambda x: feat_lbl[x])
        c1, c2 = st.columns(2)
        with c1:
            fig_h = px.histogram(pima, x=sel, color="Status",
                                 barmode="overlay", nbins=30, opacity=0.7,
                                 color_discrete_map={"No Diabetes":TEAL,"Diabetes":RED},
                                 labels={"Status":""})
            fig_h.update_traces(marker_line_width=0)
            fig_h.update_layout(
                title=dict(text=f"Distribution: {feat_lbl[sel]}",
                           font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
                legend_title_text="", **_layout(360))
            st.plotly_chart(fig_h, use_container_width=True)
        with c2:
            fig_b = px.box(pima, x="Status", y=sel, color="Status",
                           color_discrete_map={"No Diabetes":TEAL,"Diabetes":RED},
                           points="outliers", labels={"Status":""})
            fig_b.update_traces(
                marker=dict(size=3, opacity=0.4, line=dict(width=0)),
            )
            fig_b.update_layout(
                title=dict(text=f"Box Plot: {feat_lbl[sel]}",
                           font=dict(color=FONT_CLR, size=13, family=FONT_FAM)),
                showlegend=False, **_layout(360))
            st.plotly_chart(fig_b, use_container_width=True)

    with tab3:
        ca, cb = st.columns(2)
        with ca:
            xf = st.selectbox("X axis", feats, index=1, key="sx",
                              format_func=lambda x: feat_lbl[x])
        with cb:
            yf = st.selectbox("Y axis", feats, index=5, key="sy",
                              format_func=lambda x: feat_lbl[x])
        st.plotly_chart(fig_scatter(pima, xf, yf, 460), use_container_width=True)


def page_predictive(pima):
    sec("Predictive Analytics",
        "Logistic Regression vs Random Forest — Pima Indians Diabetes Dataset",
        icon_name="model")

    if pima is None:
        no_pima()
        return

    md = get_trained_models()
    if md is None:
        st.error("Model training failed. Verify `data/pima_diabetes.csv` is valid.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LR Accuracy",  f"{md['lr_m']['acc']:.1%}", delta=f"AUC {md['lr_m']['auc']:.3f}")
    m2.metric("RF Accuracy",  f"{md['rf_m']['acc']:.1%}", delta=f"AUC {md['rf_m']['auc']:.3f}")
    m3.metric("RF Precision", f"{md['rf_m']['prec']:.1%}")
    m4.metric("RF Recall",    f"{md['rf_m']['rec']:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_roc(md, 360), use_container_width=True)
    with c2:
        st.plotly_chart(fig_feature_importance(md["imp"], 360), use_container_width=True)

    tn, fp, fn, tp = confusion_matrix(md["yte"], md["rf_m"]["pred"]).ravel()
    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns([1, 1.6])
    with ca:
        st.plotly_chart(fig_confusion(md, 300), use_container_width=True)
    with cb:
        dot = lambda col: (f"<span style='display:inline-block;width:8px;height:8px;"
                           f"border-radius:50%;background:{col};margin-right:9px;"
                           f"vertical-align:middle;'></span>")
        st.markdown(f"""
        <div style='background:var(--surface);border:1px solid var(--border);
                    border-radius:12px;padding:18px 20px;'>
          <div style='color:var(--text);font-weight:700;font-size:.92rem;
                      margin-bottom:14px;'>Random Forest — Confusion Matrix Breakdown</div>
          <div style='color:var(--text-muted);font-size:.86rem;line-height:2;'>
            {dot(POSITIVE)}True Negatives (correctly healthy): <strong>{tn}</strong><br>
            {dot(POSITIVE)}True Positives (correctly diabetic): <strong>{tp}</strong><br>
            {dot(RISK_HI)}False Positives (false alarm): <strong>{fp}</strong><br>
            {dot(RISK_HI)}False Negatives (missed): <strong>{fn}</strong>
          </div>
          <div style='color:var(--text-dim);font-size:.84rem;margin-top:14px;
                      padding-top:12px;border-top:1px solid var(--border);'>
            Recall: correctly identifies <strong style='color:var(--text);'>{tp/(tp+fn):.0%}</strong> of diabetic cases.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    sec("Patient Risk Assessment", icon_name="stethoscope")
    callout("Enter a patient's clinical measurements to estimate their individual diabetes risk probability.")
    st.markdown(
        f"<p style='display:flex;align-items:center;gap:7px;color:{RISK_MID};"
        f"font-size:.8rem;margin-bottom:1rem;'>{icon('risk', 15, 2)}"
        "<span>For educational purposes only — not a clinical diagnostic tool.</span></p>",
        unsafe_allow_html=True)

    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            preg    = st.number_input("Pregnancies",            0,   20,    2)
            glucose = st.number_input("Plasma Glucose (mg/dL)", 50,  250,  120)
            bp      = st.number_input("Blood Pressure (mmHg)",  30,  130,   72)
        with c2:
            skin    = st.number_input("Skin Thickness (mm)",     5,   60,   23)
            insulin = st.number_input("Serum Insulin (µU/mL)",  10,  600,   94)
            bmi     = st.number_input("BMI (kg/m²)",          15.0, 65.0, 32.0, step=0.1)
        with c3:
            dpf     = st.number_input("Diabetes Pedigree Fn.", 0.05, 2.5,  0.47, step=0.01)
            age     = st.number_input("Age (years)",            18,   90,   35)
        submitted = st.form_submit_button("Assess Risk  →",
                                          use_container_width=True, type="primary")

    if submitted:
        inp   = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        rf_p  = float(md["rf"].predict_proba(inp)[0][1])
        lr_p  = float(md["lr"].predict_proba(md["scaler"].transform(inp))[0][1])
        avg_p = (rf_p + lr_p) / 2
        color, ico_name, label = (
            (POSITIVE, "drop", "LOW RISK")      if avg_p < 0.35 else
            (RISK_MID, "risk", "MODERATE RISK") if avg_p < 0.65 else
            (RISK_HI,  "risk", "HIGH RISK")
        )
        st.markdown(f"""
        <div style='background:var(--surface);
                    border:1px solid var(--border);border-top:3px solid {color};
                    border-radius:16px;padding:28px 32px;text-align:center;
                    margin-top:16px;box-shadow:var(--shadow);'>
          <div style='display:inline-flex;align-items:center;justify-content:center;
                      width:56px;height:56px;border-radius:14px;margin-bottom:12px;
                      background:{color}1a;color:{color};'>{icon(ico_name, 28, 2)}</div>
          <div style='color:{color};font-size:1.6rem;font-weight:800;
                      letter-spacing:-.02em;margin-bottom:6px;'>{label}</div>
          <div style='color:var(--text);font-size:2.4rem;font-weight:800;
                      letter-spacing:-.03em;margin-bottom:6px;'>{avg_p:.1%}</div>
          <div style='color:var(--text-dim);font-size:.82rem;'>
            Random Forest: {rf_p:.1%} &nbsp;·&nbsp; Logistic Regression: {lr_p:.1%}
          </div>
        </div>""", unsafe_allow_html=True)


def page_about():
    sec("About DiabetesIQ",
        "Data sources, methodology, and project information",
        icon_name="info")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Data Sources")
        st.dataframe(pd.DataFrame({
            "Source":    ["World Bank API","IDF Atlas 2024 (11th Ed.)","Pima Indians DB"],
            "Variables": ["Prevalence % by country/year",
                          "Regional, age, gender, type stats",
                          "8 clinical features + Outcome (768 pts)"],
            "Coverage":  ["253 countries — 2011 & 2024",
                          "215 countries, global estimates",
                          "Female Pima patients, Arizona, USA"],
        }), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("#### Methodology")
        st.markdown("""
- **Country map**: World Bank `SH.STA.DIAB.ZS` — age-standardised % adults 20–79
- **Global trend**: IDF Atlas editions 2000–2024 (published aggregate estimates)
- **Pima preprocessing**: Biological zeros imputed with column median
- **ML pipeline**: 80/20 stratified split · StandardScaler for LR · RF on raw features
- **Risk score**: Ensemble average of LR and RF probability outputs
- **Limitation**: Pima dataset is female-only, single ethnic group
        """)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Project Info")
        st.markdown("""
| | |
|---|---|
| **Dashboard** | DiabetesIQ v3.0 |
| **Course** | MSBA382 — Healthcare Analytics |
| **School** | Suliman S. Olayan School of Business |
| **Data** | IDF Atlas 2024 (11th Edition) |
        """)
    with c4:
        st.markdown("#### References")
        st.markdown("""
- IDF Diabetes Atlas 11th Edition (2024). [diabetesatlas.org](https://diabetesatlas.org)
- World Bank Open Data. [data.worldbank.org](https://data.worldbank.org/indicator/SH.STA.DIAB.ZS)
- Smith et al. (1988). ADAP Algorithm for Diabetes Onset. *SCAMC*.
- Scikit-learn: Pedregosa et al., JMLR 12 (2011) 2825–2830.
        """)
    st.markdown(
        f"<p style='text-align:center;color:{BLUE_DIM};font-size:.78rem;margin-top:32px;'>"
        "Built with Streamlit · MSBA382 Healthcare Analytics · OSB 2026</p>",
        unsafe_allow_html=True)


# ── THEME TOGGLE ──────────────────────────────────────────────────────────────
def theme_toggle():
    """Sidebar light/dark switch. Stores choice in session_state; the next render
    re-injects CSS for an instant re-theme."""
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "dark"
    current = st.session_state.theme_mode
    other = "light" if current == "dark" else "dark"
    label_ico = icon("sun" if other == "light" else "moon", 15, 2)
    st.markdown(
        f"<p style='color:var(--text-dim);font-size:.68rem;font-weight:600;"
        f"letter-spacing:.08em;text-transform:uppercase;text-align:center;"
        f"margin:0 0 6px;'>Appearance</p>", unsafe_allow_html=True)
    if st.button(f"Switch to {other.capitalize()} mode",
                 use_container_width=True, key="theme_btn"):
        st.session_state.theme_mode = other
        st.rerun()


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "dark"
    inject_css()
    refresh_theme_tokens()

    check_login()

    with st.sidebar:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:11px;padding:22px 8px 14px;'>
          <div style='display:inline-flex;align-items:center;justify-content:center;
                      width:42px;height:42px;border-radius:11px;
                      background:var(--accent-soft);color:var(--accent);flex:0 0 auto;'>
            {icon('logo', 24, 1.8)}
          </div>
          <div style='text-align:left;'>
            <div style='color:var(--text);font-size:1.08rem;font-weight:800;
                        letter-spacing:-.02em;line-height:1.1;'>DiabetesIQ</div>
            <div style='color:var(--accent);font-size:.62rem;font-weight:600;
                        letter-spacing:.1em;text-transform:uppercase;margin-top:2px;'>
              Diabetes Intelligence
            </div>
          </div>
        </div>
        <hr style='border-color:var(--border);margin:0 0 10px;'>
        """, unsafe_allow_html=True)

        PAGES = [
            "Overview",
            "Dashboard",
            "Global Map",
            "Trends",
            "Demographics",
            "Risk Factors",
            "Predictive Analytics",
            "About",
        ]
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")

        st.markdown("<hr style='border-color:var(--border);margin:10px 0;'>",
                    unsafe_allow_html=True)
        theme_toggle()

        st.markdown(f"""
        <hr style='border-color:var(--border);margin:10px 0;'>
        <p style='color:var(--text-dim);font-size:.7rem;text-align:center;line-height:1.6;'>
          Source: World Bank · IDF Atlas 2024<br>MSBA382 · OSB 2026
        </p>
        """, unsafe_allow_html=True)

        if st.button("Log out", use_container_width=True, key="logout_btn"):
            st.session_state.authenticated = False
            st.rerun()

    map_df   = load_map_data()
    trend    = get_global_trend()
    regional = get_regional_data()
    demo     = get_demographics()
    pima     = load_pima()

    if   page == PAGES[0]: page_overview(trend, regional)
    elif page == PAGES[1]: page_dashboard(map_df, trend, regional, demo, pima)
    elif page == PAGES[2]: page_global_map(map_df)
    elif page == PAGES[3]: page_trends(map_df, trend, regional)
    elif page == PAGES[4]: page_demographics(demo, pima)
    elif page == PAGES[5]: page_risk_factors(pima)
    elif page == PAGES[6]: page_predictive(pima)
    elif page == PAGES[7]: page_about()


if __name__ == "__main__":
    main()
