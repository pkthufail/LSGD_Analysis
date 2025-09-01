# pages/8_Other.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from lib.data import load_data, get_data_path, data_controls

# ---------------- Page Config ----------------
st.set_page_config(page_title="Other · LSGD Explorer", page_icon="🧠", layout="wide")
st.title("🧠 Other Insights")

# ---------------- Sidebar: Data controls ----------------
data_controls()
df = load_data(get_data_path()).copy()

# ---------------- Hygiene ----------------
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].str.strip()

if "Votes" in df.columns:
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)
if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")  # keep NaN if unknown

# ---------------- Small helpers ----------------
def render_styled_table(obj, fmt_numbers=None, fmt_perc=None):
    """Hide index, format numbers with commas and percentages to 2dp, responsive table."""
    fmt_numbers = fmt_numbers or []
    fmt_perc    = fmt_perc or []
    try:
        from pandas.io.formats.style import Styler  # pandas-version friendly
        styler = obj if isinstance(obj, Styler) else obj.style
    except Exception:
        styler = obj.style

    fmt_map = {**{c: "{:,.0f}" for c in fmt_numbers},
               **{c: "{:,.2f}%" for c in fmt_perc}}
    styler = styler.format(fmt_map)
    try:
        styler = styler.hide(axis="index")
    except Exception:
        styler = styler.hide_index()

    html = styler.to_html()
    st.markdown("""
      <style>
        .tbl-wrap { width: 100%; overflow-x: auto; }
        .tbl-wrap table { width: 100%; border-collapse: collapse; table-layout: auto; }
        .tbl-wrap th, .tbl-wrap td { padding: 6px 8px; }
        @media (max-width: 1200px) { .tbl-wrap th, .tbl-wrap td { font-size: 0.9rem; } }
      </style>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='tbl-wrap'>{html}</div>", unsafe_allow_html=True)

# ---------------- Filters ----------------
st.markdown("### Filters")
parties = sorted(df["Party"].dropna().unique().tolist())
default_party_ix = next((i for i, p in enumerate(parties) if str(p).upper().strip() == "IUML"), 0)
sel_party = st.selectbox("Party", parties, index=default_party_ix)

districts = ["Kerala"] + sorted(df["District"].dropna().unique().tolist(), key=lambda x: str(x))
sel_district = st.selectbox("District", districts, index=0)

# Filter by selection
if sel_district == "Kerala":
    df_party = df[df["Party"] == sel_party].copy()
else:
    df_party = df[(df["Party"] == sel_party) & (df["District"] == sel_district)].copy()

st.markdown("---")

# =====================================================
# 1) Under 40 vs Over 40 vs Unknown (table + pie)
# =====================================================
st.subheader(f"🧾 Candidates (Under 40 vs Over 40) — {sel_party}")

if df_party.empty or "Age" not in df_party.columns:
    st.info("No age data available for this selection.")
else:
    # NA-safe boolean logic for age groups
    age = pd.to_numeric(df_party["Age"], errors="coerce")
    df_party["AgeGroup40"] = np.where(
        age.notna() & (age < 40), "Under 40",
        np.where(age.notna() & (age >= 40), "Over 40", "Unknown")
    )

    age_summary = (
        df_party.groupby("AgeGroup40", dropna=False)
        .agg(
            Contested=("Candidate", "count"),
            Won=("Rank", lambda s: (pd.to_numeric(s, errors="coerce") == 1).sum())
        )
        .reset_index()
    )

    # Order rows: Under 40, Over 40, Unknown
    order_map = {"Under 40": 0, "Over 40": 1, "Unknown": 2}
    age_summary["__order"] = age_summary["AgeGroup40"].map(order_map).fillna(99)
    age_summary = age_summary.sort_values("__order").drop(columns="__order")

    age_summary["Win %"] = np.where(
        age_summary["Contested"] > 0,
        age_summary["Won"] / age_summary["Contested"] * 100,
        0.0
    )

    render_styled_table(age_summary, fmt_numbers=["Contested", "Won"], fmt_perc=["Win %"])

    # Pie (colors align to Under 40 / Over 40 / Unknown)
    fig_pie = go.Figure(data=[
        go.Pie(
            labels=age_summary["AgeGroup40"],
            values=age_summary["Contested"],
            hole=0.3,
            marker=dict(colors=["#AED6F1", "#F1948A", "#D5DBDB"]),
            textinfo="label+percent",
        )
    ])
    fig_pie.update_layout(title=f"🎂 Candidate Age Group — {sel_party}", height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

# =====================================================
# 2) Age-wise performance (grouped bars)
# =====================================================
def categorize_age(a):
    if pd.isna(a):
        return "Unknown"
    x = float(a)
    if x < 25:         return "< 25"
    if 25 <= x < 35:   return "25 – 35"
    if 35 <= x < 50:   return "35 – 50"
    if 50 <= x < 60:   return "50 – 60"
    return "60+"

if not df_party.empty:
    df_party["AgeCategory"] = df_party["Age"].apply(categorize_age)
    age_order = ["< 25", "25 – 35", "35 – 50", "50 – 60", "60+", "Unknown"]

    age_chart_df = (
        df_party.groupby("AgeCategory", dropna=False)
        .agg(
            Contested=("Candidate", "count"),
            Won=("Rank", lambda s: (pd.to_numeric(s, errors="coerce") == 1).sum())
        )
        .reset_index()
    )
    age_chart_df["AgeCategory"] = pd.Categorical(age_chart_df["AgeCategory"],
                                                 categories=age_order, ordered=True)
    age_chart_df = age_chart_df.sort_values("AgeCategory")

    st.subheader(f"📊 Age-Wise Performance — {sel_party}")
    age_chart_long = age_chart_df.melt(
        id_vars="AgeCategory",
        value_vars=["Contested", "Won"],
        var_name="Status",
        value_name="Count"
    )
    fig = px.bar(
        age_chart_long,
        x="AgeCategory",
        y="Count",
        color="Status",
        barmode="group",
        text="Count",
        labels={"AgeCategory": "Age Group"},
        color_discrete_map={"Contested": "#89CFF0", "Won": "#87BB62"}
    )
    fig.update_layout(
        xaxis_title="Age Group",
        yaxis_title="Number of Candidates",
        title="Contested vs Won by Age Category",
        legend_title="Status",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 3) Age distribution of candidates (+ Cumulative %)
# =====================================================
if not df_party.empty:
    age_order = ["< 25", "25 – 35", "35 – 50", "50 – 60", "60+", "Unknown"]
    age_dist = (
        df_party.groupby("AgeCategory", dropna=False)
        .agg(Contested=("Candidate", "count"))
        .reset_index()
    )
    # ensure all categories appear in order
    age_dist["AgeCategory"] = pd.Categorical(age_dist["AgeCategory"],
                                             categories=age_order, ordered=True)
    age_dist = age_dist.sort_values("AgeCategory")

    total_contested = int(age_dist["Contested"].sum())
    age_dist["% of Total Contested"] = np.where(
        total_contested > 0, age_dist["Contested"] / total_contested * 100, 0.0
    )
    # Cumulative (%)
    age_dist["Cumulative (%)"] = age_dist["% of Total Contested"].cumsum()

    st.subheader(f"📋 Age Distribution of Candidates — {sel_party}")
    render_styled_table(
        age_dist[["AgeCategory", "Contested", "% of Total Contested", "Cumulative (%)"]],
        fmt_numbers=["Contested"],
        fmt_perc=["% of Total Contested", "Cumulative (%)"]
    )

# =====================================================
# 4) Gender distribution (pie) + table
# =====================================================
if not df_party.empty and "Gender" in df_party.columns:
    st.subheader(f"👥 Gender-wise Performance — {sel_party}")

    # Pie
    gender_counts = df_party["Gender"].fillna("Unknown").value_counts().reset_index()
    gender_counts.columns = ["Gender", "Count"]
    fig_gender = go.Figure(data=[
        go.Pie(
            labels=gender_counts["Gender"],
            values=gender_counts["Count"],
            hole=0.3,
            marker=dict(colors=["#6FA8DC", "#F9CB9C", "#CCCCCC"]),  # M, F, Unknown
            textinfo="label+percent"
        )
    ])
    fig_gender.update_layout(title="Gender Distribution of Candidates", height=350)
    st.plotly_chart(fig_gender, use_container_width=True)

    # Table
    gender_table = (
        df_party.assign(RankN=pd.to_numeric(df_party.get("Rank"), errors="coerce"))
        .groupby(df_party["Gender"].fillna("Unknown"))
        .agg(
            Contested=("Candidate", "count"),
            Won=("RankN", lambda s: (s == 1).sum())
        )
        .reset_index()
        .rename(columns={"Gender": "Gender"})
    )
    gender_table["Win %"] = np.where(
        gender_table["Contested"] > 0,
        gender_table["Won"] / gender_table["Contested"] * 100,
        0.0
    )
    gender_table["Gender"] = gender_table["Gender"].replace({"M": "Male", "F": "Female"})
    render_styled_table(gender_table, fmt_numbers=["Contested", "Won"], fmt_perc=["Win %"])
