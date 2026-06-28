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

# Anchor all file paths to this script's directory so Streamlit Cloud works correctly
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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 1. BACKGROUNDS — force dark theme regardless of config.toml */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.main .block-container { background-color:#0a1628 !important; }

/* 2. HEADER / FOOTER */
header[data-testid="stHeader"],
.stApp > header               { background-color:#0a1628 !important; }
#MainMenu, footer              { visibility:hidden; }
.block-container               { padding-top:1.5rem !important; }

/* 3. SIDEBAR */
[data-testid="stSidebar"]      { background:#0d1e3a !important; }

/* 4. METRIC CARDS */
[data-testid="metric-container"] {
    background:#1a2744 !important; border:1px solid #00d4aa;
    border-radius:10px; padding:14px 18px;
}
[data-testid="stMetricLabel"] { color:#8ab4c9 !important; font-size:.82rem !important; }
[data-testid="stMetricValue"] { color:#ffffff !important; font-size:1.7rem !important; font-weight:700 !important; }
[data-testid="stMetricDelta"] { color:#00d4aa !important; }

/* 5. TABS */
[data-testid="stTabs"] button               { color:#8ab4c9 !important; background:transparent; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color:#00d4aa !important; border-bottom:2px solid #00d4aa !important;
}

/* 6. GENERAL TEXT — headings and body (safe: .sec-hdr is a <div>, not <h*>, so no conflict) */
.stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6  { color:#cce8f0 !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span { color:#cce8f0; }
[data-testid="stMarkdownContainer"] strong { color:#ffffff; }
p, label                              { color:#cce8f0 !important; }

/* 7. CUSTOM COMPONENTS — must use !important to beat Streamlit's inline styles */
.sec-hdr {
    color:#00d4aa !important; font-size:1.5rem; font-weight:700;
    border-bottom:2px solid #2a3a5c; padding-bottom:4px; margin-bottom:18px;
}
.callout {
    background:#1a2744; border-left:4px solid #00d4aa;
    border-radius:6px; padding:12px 16px; margin:10px 0; color:#cce8f0;
}
.type-card {
    background:#1a2744; border-radius:10px; padding:16px;
    text-align:center; border:1px solid #2a3a5c; height:100%;
}
.type-card h3 { color:#00d4aa !important; margin-bottom:8px; }
.type-card p  { color:#a0b8cc; font-size:.86rem; line-height:1.5; }

/* 8. MISC */
hr                             { border-color:#2a3a5c !important; }
[data-testid="stAlert"]        { background:#1a2744 !important; }
</style>
""", unsafe_allow_html=True)

PASSWORD  = "diabetes2026"
PLOT_BG   = "#1a2744"
PAPER_BG  = "#1a2744"
FONT_CLR  = "#cce8f0"
GRID_CLR  = "#2a3a5c"


def _layout(height=360, t=30, b=40, l=10, r=10):
    return dict(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_CLR), height=height,
        margin=dict(t=t, b=b, l=l, r=r),
        xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
        legend=dict(bgcolor="#0a1628", bordercolor="#2a3a5c", borderwidth=1),
    )


# ── LOGIN ─────────────────────────────────────────────────────────────────────
def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        c1, c2, c3 = st.columns([1, 1.3, 1])
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align:center;padding:36px 28px;background:#1a2744;
                        border-radius:16px;border:1px solid #2a3a5c;'>
              <div style='font-size:3rem;'>🩺</div>
              <h1 style='color:#fff;margin:8px 0 4px;font-size:1.8rem;'>DiabetesIQ</h1>
              <p style='color:#8ab4c9;margin-bottom:20px;font-size:.93rem;'>
                Global Diabetes Intelligence Platform<br>
                <em>MSBA382 — Healthcare Analytics · OSB 2026</em>
              </p>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            pwd = st.text_input("Password", type="password",
                                placeholder="Enter dashboard password")
            if st.button("Login", use_container_width=True, type="primary"):
                if pwd == PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        st.stop()   # valid here: blocks unauthenticated rendering


# ── DATA LOADERS ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading country data…")
def load_map_data():
    """World Bank API: 247 countries (2011) and 253 countries (2024)."""
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
        # Fallback to saved CSV
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


PIMA_FALLBACK_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
)

def _clean_pima(df: pd.DataFrame) -> pd.DataFrame:
    """Impute biological zeros and add Status column."""
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan).fillna(df[c].median())
    df["Status"] = df["Outcome"].map({0: "No Diabetes", 1: "Diabetes"})
    return df

@st.cache_data(show_spinner="Loading Pima dataset…")
def load_pima():
    """Local file → public URL fallback. Always returns a DataFrame or None."""
    # 1) try local file (works on localhost and if file is properly committed)
    if PIMA_PATH.exists():
        return _clean_pima(pd.read_csv(PIMA_PATH))
    # 2) fallback: download from plotly's public mirror (same dataset, same headers)
    try:
        df = pd.read_csv(PIMA_FALLBACK_URL)
        return _clean_pima(df)
    except Exception:
        return None


# NOTE: @st.cache_data (not cache_resource) handles DataFrame args correctly.
@st.cache_data(show_spinner="Training ML models…")
def get_trained_models():
    """Self-contained: loads Pima (local → URL fallback), trains LR + RF."""
    try:
        if PIMA_PATH.exists():
            df = pd.read_csv(PIMA_PATH)
        else:
            df = pd.read_csv(PIMA_FALLBACK_URL)
        for c in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
            if c in df.columns:
                df[c] = df[c].replace(0, np.nan).fillna(df[c].median())

        feats = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                 "Insulin","BMI","DiabetesPedigreeFunction","Age"]
        X, y = df[feats], df["Outcome"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        Xs_tr = scaler.fit_transform(X_tr)
        Xs_te = scaler.transform(X_te)

        lr = LogisticRegression(max_iter=1000, random_state=42).fit(Xs_tr, y_tr)
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr, y_tr)

        def _m(model, Xte, yte):
            yp   = model.predict(Xte)
            ypr  = model.predict_proba(Xte)[:, 1]
            return dict(acc=accuracy_score(yte, yp),
                        auc=roc_auc_score(yte, ypr),
                        prec=precision_score(yte, yp),
                        rec=recall_score(yte, yp),
                        prob=ypr, pred=yp)

        imp = (pd.Series(rf.feature_importances_, index=feats)
               .sort_values(ascending=False)
               .reset_index())
        imp.columns = ["Feature", "Importance"]

        return dict(
            lr=lr, rf=rf, scaler=scaler, feats=feats,
            Xte=X_te, Xte_sc=Xs_te, yte=y_te,
            lr_m=_m(lr, Xs_te, y_te),
            rf_m=_m(rf, X_te,  y_te),
            imp=imp,
        )
    except FileNotFoundError:
        return None


# ── UI HELPERS ────────────────────────────────────────────────────────────────
def sec(title, sub=""):
    st.markdown(f"<div class='sec-hdr'>{title}</div>", unsafe_allow_html=True)
    if sub:
        st.markdown(
            f"<p style='color:#8ab4c9;margin-top:-14px;margin-bottom:14px;font-size:.88rem;'>{sub}</p>",
            unsafe_allow_html=True)

def callout(text):
    st.markdown(f"<div class='callout'>💡 {text}</div>", unsafe_allow_html=True)

def no_pima():
    st.info(
        "**Pima dataset not loaded.**  \n"
        "Download `diabetes.csv` from [Kaggle]"
        "(https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) "
        "and save it as `data/pima_diabetes.csv`, then refresh.",
        icon="⚠️")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS  (return Figure; called from both Dashboard and detail pages)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_global_trend(trend, h=340):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["Year"], y=trend["Prevalence"],
        mode="lines+markers",
        line=dict(color="#00d4aa", width=3),
        marker=dict(size=8, color="#00d4aa"),
        fill="tozeroy", fillcolor="rgba(0,212,170,.12)",
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Global Diabetes Prevalence Trend — IDF Atlas 2000–2024",
                   font=dict(color=FONT_CLR, size=14)),
        xaxis_title="Year", yaxis_title="Prevalence (%)",
        **_layout(h),
    )
    return fig


def fig_region_bar(reg, h=340):
    df = reg.sort_values("Prevalence")
    fig = px.bar(df, x="Prevalence", y="Code", orientation="h",
                 text="Prevalence", color="Prevalence",
                 color_continuous_scale=[[0,"#1a4a6e"],[.5,"#00a88c"],[1,"#e63946"]],
                 labels={"Prevalence":"Prevalence (%)","Code":"IDF Region"},
                 template="plotly_dark")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                      textfont_color=FONT_CLR)
    fig.update_layout(
        title=dict(text="Prevalence by IDF Region — 2024",
                   font=dict(color=FONT_CLR, size=14)),
        coloraxis_showscale=False,
        **_layout(h, l=60, r=50),
    )
    return fig


def fig_world_map(map_df, year=2024, h=460):
    df = map_df[map_df["Year"] == year].copy()
    df["Rank"] = df["Prevalence"].rank(ascending=False).astype(int)
    vmax = float(df["Prevalence"].quantile(0.97))
    vmin = float(df["Prevalence"].min())
    fig = px.choropleth(
        df, locations="Code", color="Prevalence",
        hover_name="Entity",
        hover_data={"Prevalence": True, "Rank": True, "Code": False},
        color_continuous_scale=[
            [0.0, "#0a2a4a"], [0.25, "#1a5c78"], [0.5, "#2a9d8f"],
            [0.75, "#f4a261"], [1.0,  "#e63946"],
        ],
        range_color=[vmin, vmax],
        labels={"Prevalence": "Prevalence (%)"},
    )
    fig.update_layout(
        paper_bgcolor="#0a1628",
        geo=dict(
            bgcolor="#0a1628", lakecolor="#0a1628",
            showframe=False, showcoastlines=True, coastlinecolor="#2a3a5c",
            landcolor="#162035", showocean=True, oceancolor="#0a1628",
            showcountries=True, countrycolor="#2a3a5c",
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Prevalence (%)", font=dict(color=FONT_CLR)),
            tickfont=dict(color=FONT_CLR),
            bgcolor="#1a2744", bordercolor="#2a3a5c", thickness=14, len=0.7,
        ),
        margin=dict(t=10, b=0, l=0, r=0),
        height=h,
    )
    return fig


def fig_age_gender(demo, h=340):
    fig = go.Figure([
        go.Bar(name="Male",   x=demo["AgeGroup"], y=demo["Male"],
               marker_color="#00d4aa", opacity=0.9),
        go.Bar(name="Female", x=demo["AgeGroup"], y=demo["Female"],
               marker_color="#f4a261", opacity=0.9),
    ])
    fig.update_layout(
        barmode="group",
        title=dict(text="Diabetes Prevalence by Age Group & Gender — 2024",
                   font=dict(color=FONT_CLR, size=14)),
        xaxis_title="Age Group", yaxis_title="Prevalence (%)",
        **_layout(h),
    )
    return fig


def fig_type_donut(h=320):
    fig = go.Figure(go.Pie(
        labels=["Type 2 (T2D)", "Type 1 (T1D)", "Gestational", "Other"],
        values=[92, 6, 3, 2], hole=0.52,
        marker=dict(colors=["#e63946","#00d4aa","#f4a261","#6a4c93"]),
        textinfo="label+percent",
        textfont=dict(size=12, color=FONT_CLR),
        hovertemplate="<b>%{label}</b><br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Global Distribution by Diabetes Type",
                   font=dict(color=FONT_CLR, size=14)),
        paper_bgcolor=PAPER_BG, height=h,
        margin=dict(t=50, b=20, l=10, r=10),
        legend=dict(bgcolor="#0a1628", font=dict(color=FONT_CLR)),
        annotations=[dict(text="Type", x=0.5, y=0.5,
                          font=dict(size=15, color="#fff"), showarrow=False)],
    )
    return fig


def fig_scatter(df, xf, yf, h=380):
    fig = px.scatter(df, x=xf, y=yf, color="Status",
                     color_discrete_map={"No Diabetes":"#00d4aa","Diabetes":"#e63946"},
                     opacity=0.6, template="plotly_dark",
                     hover_data=["Age","Outcome"],
                     labels={"Status": ""})
    fig.update_traces(marker_size=5)
    if xf == "Glucose":
        fig.add_vline(x=140, line_dash="dash", line_color="#f4a261",
                      annotation_text="Glucose = 140 mg/dL",
                      annotation_font_color="#f4a261", annotation_font_size=11)
    fig.update_layout(
        title=dict(text=f"{xf} vs {yf} — Diabetic vs Non-Diabetic",
                   font=dict(color=FONT_CLR, size=14)),
        **_layout(h),
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
                 color="Importance",
                 color_continuous_scale=[[0,"#1a4a6e"],[.5,"#00a88c"],[1,"#00d4aa"]],
                 text="Importance", template="plotly_dark",
                 labels={"Importance":"Importance","Feature":""})
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                      textfont_color=FONT_CLR)
    fig.update_layout(
        title=dict(text="RF Feature Importance — Diabetes Prediction",
                   font=dict(color=FONT_CLR, size=14)),
        yaxis_autorange="reversed",          # underscore avoids conflict with _layout()'s yaxis
        coloraxis_showscale=False,
        **_layout(h, l=140, r=60),
    )
    return fig


def fig_roc(md, h=340):
    fig = go.Figure()
    for name, prob, color in [
        ("Logistic Regression", md["lr_m"]["prob"], "#f4a261"),
        ("Random Forest",       md["rf_m"]["prob"], "#00d4aa"),
    ]:
        fpr, tpr, _ = roc_curve(md["yte"], prob)
        auc = roc_auc_score(md["yte"], prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC={auc:.3f})",
                                 line=dict(color=color, width=2.5)))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="#555", dash="dash", width=1))
    fig.update_layout(
        title=dict(text="ROC Curve — LR vs Random Forest",
                   font=dict(color=FONT_CLR, size=14)),
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        **_layout(h),
    )
    return fig


def fig_corr(df, h=400):
    feats = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
             "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    corr = df[feats].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=feats, y=feats,
        colorscale="RdBu_r", zmid=0,
        texttemplate="%{z:.2f}",        # use z directly — no separate text= array
        textfont=dict(size=10, color=FONT_CLR),
        colorbar=dict(
            tickfont=dict(color=FONT_CLR),
            title=dict(text="r", font=dict(color=FONT_CLR)),
        ),
    ))
    fig.update_layout(
        title=dict(text="Correlation Matrix — Clinical Features",
                   font=dict(color=FONT_CLR, size=14)),
        paper_bgcolor=PAPER_BG, font=dict(color=FONT_CLR),
        height=h, margin=dict(t=50, b=10, l=10, r=10),
        xaxis=dict(tickfont=dict(color=FONT_CLR)),
        yaxis=dict(tickfont=dict(color=FONT_CLR)),
    )
    return fig


def fig_confusion(md, h=300):
    cm     = confusion_matrix(md["yte"], md["rf_m"]["pred"])
    labels = ["No Diabetes", "Diabetes"]
    fig    = px.imshow(cm, x=labels, y=labels,
                       color_continuous_scale=[[0,"#1a2744"],[1,"#00d4aa"]],
                       text_auto=True,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       template="plotly_dark")
    # Fix: use marker_text font, not textfont magic underscore
    fig.update_traces(textfont=dict(size=22, color="#ffffff"))
    fig.update_layout(
        paper_bgcolor=PAPER_BG, height=h,
        margin=dict(t=30, b=30, l=10, r=10),
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(color=FONT_CLR)),
        yaxis=dict(tickfont=dict(color=FONT_CLR)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def page_overview(trend, regional):
    sec("🏠 Global Diabetes Overview",
        "Key statistics and burden summary — IDF Diabetes Atlas 2024 (11th Edition)")

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

    callout("The Middle East & North Africa (MENA) region carries the highest diabetes burden "
            "globally at 16.7% — nearly double Africa's rate.")

    st.markdown("<br>", unsafe_allow_html=True)
    sec("Understanding Diabetes Types")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='type-card'><h3>⚡ Type 1</h3>
        <p><strong style='color:#fff'>~6%</strong> of cases<br><br>
        Autoimmune — body produces no insulin. Diagnosed childhood/young adulthood.
        Requires lifelong insulin therapy.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='type-card'><h3>🔴 Type 2</h3>
        <p><strong style='color:#fff'>~92%</strong> of cases<br><br>
        Insulin resistance or insufficient production. Linked to obesity and
        sedentary lifestyle. Largely preventable.</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='type-card'><h3>🤰 Gestational</h3>
        <p><strong style='color:#fff'>~3–5%</strong> of pregnancies<br><br>
        Develops during pregnancy. Usually resolves post-delivery but raises
        future Type 2 risk for mother and child.</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_global_trend(trend, 310), use_container_width=True)
    with c2:
        st.plotly_chart(fig_region_bar(regional, 310), use_container_width=True)


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def page_dashboard(map_df, trend, regional, demo, pima):
    sec("📊 Analytics Dashboard",
        "All key visualizations at a glance")

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Adults with Diabetes", "537M",   delta="1 in 10 adults")
    k2.metric("Global Prevalence",    "10.5%",  delta="+4.1 pp since 2000")
    k3.metric("Deaths / Year",        "6.7M",   delta="1 every 5 seconds")
    k4.metric("Undiagnosed",          "240M",   delta="~1 in 2 unaware")
    k5.metric("MENA Rate",            "16.7%",  delta="Highest region")

    st.divider()

    # World map — full width
    st.markdown("#### 🌍 Global Diabetes Prevalence — 2024 (% of adults 20–79)")
    if not map_df.empty:
        st.plotly_chart(fig_world_map(map_df, year=2024, h=460),
                        use_container_width=True)
    else:
        st.warning("Map data unavailable — check your network connection.")

    st.divider()

    # Trend + Region
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_global_trend(trend, 340), use_container_width=True)
    with c2:
        st.plotly_chart(fig_region_bar(regional, 340), use_container_width=True)

    st.divider()

    # Age/gender + Diabetes type
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(fig_age_gender(demo, 340), use_container_width=True)
    with c4:
        st.plotly_chart(fig_type_donut(340), use_container_width=True)

    # Pima charts (only if dataset present)
    if pima is not None:
        st.divider()
        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(fig_scatter(pima, "Glucose", "BMI", 340),
                            use_container_width=True)
        with c6:
            md = get_trained_models()
            if md:
                st.plotly_chart(fig_feature_importance(md["imp"], 340),
                                use_container_width=True)

        st.divider()
        c7, c8 = st.columns(2)
        with c7:
            st.plotly_chart(fig_corr(pima, 380), use_container_width=True)
        with c8:
            md = get_trained_models()
            if md:
                st.plotly_chart(fig_roc(md, 380), use_container_width=True)
    else:
        st.divider()
        no_pima()


# ── GLOBAL MAP ────────────────────────────────────────────────────────────────
def page_global_map(map_df):
    sec("🌍 Global Geographic Distribution",
        "Diabetes prevalence (% adults 20–79) by country — World Bank / IDF Atlas")

    if map_df.empty:
        st.warning("Map data could not be loaded. Check your network connection.")
        return

    available = sorted(map_df["Year"].unique().tolist(), reverse=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        sel_yr = st.radio("Year", available, horizontal=True,
                          format_func=str)
    with c2:
        st.metric("Countries with data", len(map_df[map_df["Year"] == sel_yr]),
                  delta=f"Year {sel_yr}")

    st.plotly_chart(fig_world_map(map_df, year=sel_yr, h=500),
                    use_container_width=True)
    callout(f"In {sel_yr}, MENA and Pacific Island nations show the highest prevalence. "
            "Pakistan leads at 31.4% (2024). Hover any country for its exact rate and rank.")

    df_yr = map_df[map_df["Year"] == sel_yr].copy()
    df_yr["Rank"] = df_yr["Prevalence"].rank(ascending=False).astype(int)

    st.markdown("#### Top 20 Countries by Prevalence")
    top20 = df_yr.nlargest(20, "Prevalence").sort_values("Prevalence")
    fig_top = px.bar(top20, x="Prevalence", y="Entity", orientation="h",
                     color="Prevalence",
                     color_continuous_scale=[[0,"#1a4a6e"],[.5,"#f4a261"],[1,"#e63946"]],
                     text="Prevalence",
                     labels={"Prevalence":"Prevalence (%)","Entity":"Country"},
                     template="plotly_dark")
    fig_top.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          textfont_color=FONT_CLR)
    fig_top.update_layout(coloraxis_showscale=False,
                          **_layout(540, l=150, r=60))
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.plotly_chart(fig_top, use_container_width=True)
    with c2:
        tbl = (df_yr.nlargest(20, "Prevalence")
               .reset_index(drop=True)[["Rank","Entity","Prevalence"]]
               .rename(columns={"Entity":"Country","Prevalence":"Prevalence (%)"}))
        tbl["Prevalence (%)"] = tbl["Prevalence (%)"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=540)


# ── TRENDS ────────────────────────────────────────────────────────────────────
def page_trends(map_df, trend, regional):
    sec("📈 Trends Over Time",
        "Global trajectory and country comparisons — IDF Atlas 2000–2024")

    tab1, tab2, tab3 = st.tabs([
        "Global Trend",
        "Country Comparison (2011 vs 2024)",
        "Regional Breakdown",
    ])

    with tab1:
        st.plotly_chart(fig_global_trend(trend, 380), use_container_width=True)
        trend_cat = trend.assign(Year=trend["Year"].astype(str))  # categorical = no gaps
        fig_abs = px.bar(trend_cat, x="Year", y="Adults_M", text="Adults_M",
                         color="Adults_M",
                         color_continuous_scale=[[0,"#1a4a6e"],[1,"#00d4aa"]],
                         labels={"Adults_M":"Adults with Diabetes (Millions)", "Year":"Year"},
                         template="plotly_dark")
        fig_abs.update_traces(texttemplate="%{text}M", textposition="outside",
                              textfont_color=FONT_CLR)
        fig_abs.update_layout(
            title=dict(text="Number of Adults with Diabetes (Millions)",
                       font=dict(color=FONT_CLR, size=14)),
            coloraxis_showscale=False, **_layout(320))
        st.plotly_chart(fig_abs, use_container_width=True)
        callout("Global diabetes cases nearly quadrupled from 151M (2000) to 589M (2024), "
                "driven by population growth, ageing, and rising obesity rates.")

    with tab2:
        if map_df.empty:
            st.warning("Country data unavailable.")
        else:
            codes_both = (map_df.groupby("Code")["Year"].nunique()
                          .loc[lambda s: s == 2].index.tolist())
            df_both    = map_df[map_df["Code"].isin(codes_both)].copy()
            all_ctry   = sorted(df_both["Entity"].unique().tolist())
            defaults   = [c for c in ["Pakistan","Kuwait","Lebanon","Saudi Arabia",
                                       "United States","France","China","India"]
                          if c in all_ctry][:8]

            selected = st.multiselect("Select countries to compare",
                                      all_ctry, default=defaults)

            if not selected:
                st.info("Select at least one country above to see the comparison.")
            else:
                sub   = df_both[df_both["Entity"].isin(selected)].copy()
                pivot = (sub.pivot(index="Entity", columns="Year", values="Prevalence")
                         .reset_index())

                # Add Year_str column for colour grouping (avoids passing unnamed Series)
                sub["Year_str"] = sub["Year"].astype(str)

                order = (pivot.assign(chg=lambda d: d[2024]-d[2011])
                         .sort_values("chg", ascending=False)["Entity"].tolist())
                sub["Entity"] = pd.Categorical(
                    sub["Entity"], categories=order, ordered=True)
                sub = sub.sort_values("Entity")

                fig_cmp = px.bar(
                    sub, x="Entity", y="Prevalence",
                    color="Year_str", barmode="group",
                    color_discrete_map={"2011":"#457b9d","2024":"#e63946"},
                    text="Prevalence",
                    labels={"Prevalence":"Prevalence (%)","Entity":"Country",
                            "Year_str":"Year"},
                    template="plotly_dark")
                fig_cmp.update_traces(texttemplate="%{text:.1f}%",
                                      textposition="outside",
                                      textfont_color=FONT_CLR, textfont_size=10)
                fig_cmp.update_layout(
                    title=dict(text="Diabetes Prevalence: 2011 vs 2024",
                               font=dict(color=FONT_CLR, size=14)),
                    legend_title_text="Year",
                    **_layout(400, b=80))
                st.plotly_chart(fig_cmp, use_container_width=True)

                # Summary table
                if 2011 in pivot.columns and 2024 in pivot.columns:
                    pivot["Change (pp)"] = (pivot[2024] - pivot[2011]).round(1)
                    tbl = pivot[["Entity", 2011, 2024, "Change (pp)"]].copy()
                    tbl.columns = ["Country", "2011 (%)", "2024 (%)", "Change (pp)"]
                    tbl = tbl.sort_values("Change (pp)", ascending=False)
                    st.markdown("#### Change in Prevalence (2011 → 2024)")
                    st.dataframe(tbl, use_container_width=True, hide_index=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_region_bar(regional, 360), use_container_width=True)
        with c2:
            un = regional.sort_values("Undiagnosed", ascending=False)
            fig_u = px.bar(un, x="Undiagnosed", y="Code", orientation="h",
                           text="Undiagnosed", color="Undiagnosed",
                           color_continuous_scale=[[0,"#2a9d8f"],[1,"#e63946"]],
                           labels={"Undiagnosed":"Undiagnosed (%)","Code":"Region"},
                           template="plotly_dark")
            fig_u.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                                textfont_color=FONT_CLR)
            fig_u.update_layout(
                title=dict(text="Undiagnosed Diabetes by Region — 2024",
                           font=dict(color=FONT_CLR, size=14)),
                coloraxis_showscale=False, **_layout(360, l=60, r=50))
            st.plotly_chart(fig_u, use_container_width=True)
        callout("Africa has the highest undiagnosed rate (65%): nearly 2 in 3 diabetics "
                "there are unaware of their condition. North America has the best "
                "detection at only 20% undiagnosed.")


# ── DEMOGRAPHICS ──────────────────────────────────────────────────────────────
def page_demographics(demo, pima):
    sec("👥 Demographics Analysis",
        "Distribution by age group, gender, and diabetes type — IDF Atlas 2024")

    tab1, tab2, tab3, tab4 = st.tabs([
        "By Age Group", "By Gender", "Diabetes Type", "Pima Dataset Profile",
    ])

    with tab1:
        st.plotly_chart(fig_age_gender(demo, 420), use_container_width=True)
        callout("Prevalence spikes from ~2% at age 20 to ~25% by age 75. "
                "The steepest rise occurs between ages 40–60 — the key prevention window.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Pie(
                labels=["Male (11.3%)", "Female (10.9%)"],
                values=[11.3, 10.9], hole=0.55,
                marker=dict(colors=["#00d4aa","#f4a261"]),
                textinfo="label+percent",
                textfont=dict(size=14, color=FONT_CLR)))
            fig.update_layout(
                title=dict(text="Global Prevalence by Gender — 2024",
                           font=dict(color=FONT_CLR, size=14)),
                paper_bgcolor=PAPER_BG, height=320,
                margin=dict(t=50, b=20, l=10, r=10), showlegend=False,
                annotations=[dict(text="Gender", x=0.5, y=0.5,
                                  font=dict(size=16, color="#fff"), showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            abs_df = pd.DataFrame({"Gender":["Male","Female"],
                                   "Adults (M)":[278, 259]})
            fig2 = px.bar(abs_df, x="Gender", y="Adults (M)",
                          color="Gender",
                          color_discrete_map={"Male":"#00d4aa","Female":"#f4a261"},
                          text="Adults (M)", template="plotly_dark")
            fig2.update_traces(texttemplate="%{text}M", textposition="outside",
                               textfont_color=FONT_CLR)
            fig2.update_layout(
                title=dict(text="Adults with Diabetes by Gender (Millions)",
                           font=dict(color=FONT_CLR, size=14)),
                showlegend=False, **_layout(320, t=50))
            st.plotly_chart(fig2, use_container_width=True)
        callout("9.8M more men than women have diabetes globally. "
                "In MENA, female burden is disproportionately higher due to obesity trends.")

    with tab3:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.plotly_chart(fig_type_donut(380), use_container_width=True)
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            tbl = pd.DataFrame({
                "Type":           ["Type 2 (T2D)","Type 1 (T1D)","Gestational","Other (MODY…)"],
                "Share":          ["~92%","~6%","~3%","~1-2%"],
                "Typical Onset":  ["Adult 30+","Childhood","During pregnancy","Variable"],
                "Insulin Needed": ["Sometimes","Always","Sometimes","Variable"],
                "Preventable":    ["Largely yes","No","Partially","No"],
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            callout("Type 2 diabetes is the primary public health target: "
                    "92% of cases, largely preventable through lifestyle changes.")

    with tab4:
        if pima is None:
            no_pima()
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(pima, x="Age", color="Status",
                                   barmode="overlay", nbins=25, opacity=0.75,
                                   color_discrete_map={"No Diabetes":"#00d4aa",
                                                       "Diabetes":"#e63946"},
                                   labels={"Age":"Age (years)","Status":""},
                                   template="plotly_dark")
                fig.update_layout(
                    title=dict(text="Age Distribution by Diabetes Status",
                               font=dict(color=FONT_CLR, size=14)),
                    legend_title_text="",    # underscore avoids conflict with _layout()'s legend
                    **_layout(360))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                summary = pd.DataFrame({
                    "Metric": ["Total patients","With diabetes","Without diabetes",
                               "Mean age — diabetic","Mean age — non-diabetic",
                               "Mean glucose — diabetic","Mean BMI — diabetic"],
                    "Value": [
                        len(pima),
                        int(pima["Outcome"].sum()),
                        int((pima["Outcome"]==0).sum()),
                        f"{pima[pima['Outcome']==1]['Age'].mean():.1f} yrs",
                        f"{pima[pima['Outcome']==0]['Age'].mean():.1f} yrs",
                        f"{pima[pima['Outcome']==1]['Glucose'].mean():.0f} mg/dL",
                        f"{pima[pima['Outcome']==1]['BMI'].mean():.1f} kg/m²",
                    ],
                })
                st.markdown("**Dataset Profile**")
                st.dataframe(summary, use_container_width=True,
                             hide_index=True, height=290)


# ── RISK FACTORS ──────────────────────────────────────────────────────────────
def page_risk_factors(pima):
    sec("⚠️ Risk Factors Analysis",
        "Clinical risk indicators — Pima Indians Diabetes Dataset (N=768)")

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
                "followed by BMI (r = 0.29) and Age (r = 0.24). "
                "Blood pressure shows weak standalone predictive power.")

    with tab2:
        sel = st.selectbox("Feature", feats, format_func=lambda x: feat_lbl[x])
        c1, c2 = st.columns(2)
        with c1:
            fig_h = px.histogram(pima, x=sel, color="Status",
                                 barmode="overlay", nbins=30, opacity=0.75,
                                 color_discrete_map={"No Diabetes":"#00d4aa",
                                                     "Diabetes":"#e63946"},
                                 labels={"Status":""}, template="plotly_dark")
            fig_h.update_layout(
                title=dict(text=f"Distribution: {feat_lbl[sel]}",
                           font=dict(color=FONT_CLR, size=14)),
                legend_title_text="",        # underscore avoids conflict with _layout()'s legend
                **_layout(360))
            st.plotly_chart(fig_h, use_container_width=True)
        with c2:
            fig_b = px.box(pima, x="Status", y=sel, color="Status",
                           color_discrete_map={"No Diabetes":"#00d4aa",
                                               "Diabetes":"#e63946"},
                           template="plotly_dark", points="outliers",
                           labels={"Status":""})
            fig_b.update_layout(
                title=dict(text=f"Box Plot: {feat_lbl[sel]}",
                           font=dict(color=FONT_CLR, size=14)),
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


# ── PREDICTIVE ANALYTICS ─────────────────────────────────────────────────────
def page_predictive(pima):
    sec("🤖 Predictive Analytics — ML Risk Model",
        "Logistic Regression vs Random Forest trained on the Pima Indians Diabetes Dataset")

    if pima is None:
        no_pima()
        return

    md = get_trained_models()
    if md is None:
        st.error("Model training failed. Verify `data/pima_diabetes.csv` is valid.")
        return

    # Metrics
    st.markdown("#### Model Performance")
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

    # Confusion matrix
    st.markdown("#### Confusion Matrix — Random Forest")
    tn, fp, fn, tp = confusion_matrix(md["yte"], md["rf_m"]["pred"]).ravel()
    ca, cb = st.columns([1, 1.6])
    with ca:
        st.plotly_chart(fig_confusion(md, 300), use_container_width=True)
    with cb:
        st.markdown(f"""
        <div class='callout'>
        <strong>Breakdown:</strong><br><br>
        ✅ True Negatives (correctly healthy): <strong>{tn}</strong><br>
        ✅ True Positives (correctly diabetic): <strong>{tp}</strong><br>
        ❌ False Positives (false alarm): <strong>{fp}</strong><br>
        ❌ False Negatives (missed cases): <strong>{fn}</strong><br><br>
        The model correctly identifies
        <strong>{tp/(tp+fn):.0%}</strong> of actual diabetes cases (recall).
        </div>""", unsafe_allow_html=True)

    # Prediction form
    st.markdown("---")
    st.markdown("### 🧑‍⚕️ Patient Risk Assessment")
    callout("Enter a patient's clinical measurements to estimate diabetes risk probability.")
    st.markdown(
        "<p style='color:#f0d080;font-size:.82rem;'>"
        "⚠️ For educational purposes only — not a clinical diagnostic tool.</p>",
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
        submitted = st.form_submit_button("🔍 Assess Risk",
                                          use_container_width=True, type="primary")

    if submitted:
        inp     = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        rf_p    = float(md["rf"].predict_proba(inp)[0][1])
        lr_p    = float(md["lr"].predict_proba(md["scaler"].transform(inp))[0][1])
        avg_p   = (rf_p + lr_p) / 2
        color, icon, label = (
            ("#00d4aa", "✅", "LOW RISK")      if avg_p < 0.35 else
            ("#f4a261", "⚠️","MODERATE RISK") if avg_p < 0.65 else
            ("#e63946", "🚨","HIGH RISK")
        )
        st.markdown(f"""
        <div style='background:#1a2744;border:2px solid {color};border-radius:12px;
                    padding:24px;text-align:center;margin-top:14px;'>
          <div style='font-size:2.5rem;'>{icon}</div>
          <h2 style='color:{color};margin:8px 0;'>{label}</h2>
          <p style='color:#ccc;font-size:1.1rem;'>
            Estimated probability:
            <strong style='color:{color};font-size:1.4rem;'>{avg_p:.1%}</strong>
          </p>
          <p style='color:#8ab4c9;font-size:.85rem;'>
            Random Forest: {rf_p:.1%} &nbsp;|&nbsp; Logistic Regression: {lr_p:.1%}
          </p>
        </div>""", unsafe_allow_html=True)


# ── ABOUT ─────────────────────────────────────────────────────────────────────
def page_about():
    sec("ℹ️ About DiabetesIQ",
        "Data sources, methodology, and project information")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Data Sources")
        st.dataframe(pd.DataFrame({
            "Source":    ["World Bank API (SH.STA.DIAB.ZS)",
                          "IDF Atlas 11th Ed. 2024",
                          "Pima Indians Diabetes DB"],
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
- **Pima preprocessing**: Biological zero-values imputed with column median
- **ML pipeline**: 80/20 stratified split · StandardScaler for LR · RF on raw features
- **Risk score**: Ensemble average of LR and RF probability outputs
- **Limitation**: Pima dataset is female-only, single ethnic group — not universally generalisable
        """)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Project Info")
        st.markdown("""
| | |
|---|---|
| **Dashboard** | DiabetesIQ v2.0 |
| **Course** | MSBA382 — Healthcare Analytics |
| **School** | Suliman S. Olayan School of Business |
| **Data vintage** | IDF Atlas 2024 (11th Edition) |
        """)
    with c4:
        st.markdown("#### References")
        st.markdown("""
- IDF Diabetes Atlas 11th Edition (2024). [diabetesatlas.org](https://diabetesatlas.org)
- World Bank Open Data — Diabetes prevalence. [data.worldbank.org](https://data.worldbank.org/indicator/SH.STA.DIAB.ZS)
- Smith et al. (1988). ADAP Algorithm for Diabetes Onset. *SCAMC*.
- Scikit-learn: Pedregosa et al., JMLR 12 (2011) 2825–2830.
        """)
    st.markdown(
        "<p style='text-align:center;color:#8ab4c9;font-size:.82rem;margin-top:28px;'>"
        "Built with Streamlit · MSBA382 Healthcare Analytics · OSB 2026</p>",
        unsafe_allow_html=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    check_login()

    st.sidebar.markdown("""
    <div style='text-align:center;padding:14px 0 8px;'>
      <div style='font-size:2.2rem;'>🩺</div>
      <h2 style='color:#fff;margin:4px 0 2px;font-size:1.2rem;'>DiabetesIQ</h2>
      <p style='color:#8ab4c9;font-size:.75rem;margin:0;'>Global Diabetes Intelligence</p>
    </div>""", unsafe_allow_html=True)
    st.sidebar.divider()

    PAGES = [
        "🏠  Overview",
        "📊  Dashboard",
        "🌍  Global Map",
        "📈  Trends",
        "👥  Demographics",
        "⚠️  Risk Factors",
        "🤖  Predictive Analytics",
        "ℹ️  About",
    ]
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

    st.sidebar.divider()
    st.sidebar.markdown(
        "<p style='color:#8ab4c9;font-size:.72rem;text-align:center;'>"
        "Source: World Bank · IDF Atlas 2024<br>MSBA382 · OSB 2026</p>",
        unsafe_allow_html=True)
    if st.sidebar.button("🔓 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    # Load all data
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
