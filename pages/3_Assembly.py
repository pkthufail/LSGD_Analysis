# pages/3_Assembly.py
import streamlit as st
import pandas as pd
import numpy as np

from lib.data import load_data, get_data_path, data_controls
from lib.colors import FRONT_BG_COLORS, DEFAULT_BG_COLOR

# ---------------- Page Config ----------------
st.set_page_config(page_title="Assembly · LSGD Explorer", page_icon="🗳️", layout="wide")
st.title("🏛️ Assembly View")

# ---------------- Sidebar: Data controls ----------------
data_controls()
df = load_data(get_data_path()).copy()

# ---------------- Hygiene ----------------
if "Votes" in df.columns:
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].str.strip()

# Column presence
needed_any_assembly = ["Assembly", "ACName", "AssemblyName", "Constituency"]
assembly_col = next((c for c in needed_any_assembly if c in df.columns), None)
required = {"District", "Front", "LBType", "Tier", "Votes"}
missing = required - set(df.columns)
if missing or assembly_col is None:
    msg = []
    if missing:
        msg.append(f"Missing required columns: {', '.join(sorted(missing))}")
    if assembly_col is None:
        msg.append("No assembly column found (looked for: Assembly / ACName / AssemblyName / Constituency).")
    st.error(" | ".join(msg))
    st.stop()

# ---------------- Constants ----------------
FRONT_ORDER = ["UDF", "LDF", "NDA", "OTH"]
LBTYPE_ORDER = ["Grama", "Municipality", "Corporation", "Block", "District"]
SHOW_LBTYPES = ["Grama", "Municipality", "Corporation"]

# ---------------- Helpers (styling & rendering) ----------------
def _apply_number_formats(styler: pd.io.formats.style.Styler, df_display: pd.DataFrame, percent_cols: list[str]):
    fmt_map = {}
    for c in df_display.select_dtypes(include=["number"]).columns.tolist():
        fmt_map[c] = "{:,.2f}%" if c in percent_cols else "{:,.0f}"
    return styler.format(fmt_map)

def render_styled_table(styler: pd.io.formats.style.Styler):
    # hide index then render responsive HTML
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
          @media (max-width: 1200px) {
            .tbl-wrap th, .tbl-wrap td { font-size: 0.9rem; }
          }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='tbl-wrap'>{html}</div>", unsafe_allow_html=True)

# ---------------- Filters (on page) ----------------
st.markdown("### Filters")
districts = sorted([d for d in df["District"].dropna().unique().tolist()], key=lambda x: str(x))
default_dix = next((i for i, d in enumerate(districts) if str(d).strip().lower() == "malappuram"), 0)
sel_district = st.selectbox("District", districts, index=default_dix, key="assembly_district")

df_d = df[df["District"] == sel_district].copy()
assemblies = sorted([a for a in df_d[assembly_col].dropna().unique().tolist()], key=lambda x: str(x))
default_ax = next((i for i, a in enumerate(assemblies) if "malappuram" in str(a).strip().lower()), 0)
sel_assembly = st.selectbox("Assembly", assemblies, index=(default_ax if assemblies else 0), key="assembly_ac")

if df_d.empty or not assemblies:
    st.info("No rows for the selected district/assembly.")
    st.stop()

df_da = df_d[df_d[assembly_col] == sel_assembly].copy()

st.markdown("---")

# =====================================================
# 1) Front × LBType (Tier = Ward, Rank = 1) with Total
# =====================================================
st.subheader(f"🏆 Seats (Ward winners) by Front × LBType — {sel_assembly}")

if "Rank" in df_da.columns:
    df_winners = df_da[(df_da["Tier"].astype(str).str.title() == "Ward") & (df_da["Rank"] == 1)].copy()
else:
    df_winners = df_da[df_da["Tier"].astype(str).str.title() == "Ward"].copy()

if df_winners.empty:
    st.info("No Ward-tier winners found for this Assembly.")
else:
    df_winners["LBType"] = pd.Categorical(df_winners["LBType"], categories=LBTYPE_ORDER, ordered=True)
    xt = pd.crosstab(df_winners["Front"], df_winners["LBType"]).reindex(columns=SHOW_LBTYPES, fill_value=0)
    xt = xt.reindex(FRONT_ORDER, fill_value=0)
    xt["Total"] = xt.sum(axis=1)
    t1 = xt.reset_index()

    def _row_style_front(row: pd.Series):
        color = FRONT_BG_COLORS.get(row["Front"], "")
        return [f"background-color: {color}"] * len(row) if color else [""] * len(row)

    styled_t1 = _apply_number_formats(t1.style.apply(_row_style_front, axis=1), t1, percent_cols=[])
    render_styled_table(styled_t1)

# =====================================================
# 2) Votes by Front per Local Body (Ward-tier)
#     Rows = all LBs + summary rows [Total, Percentage, Rank]
#     Cols = LBName, LBType, UDF, LDF, NDA, OTH
#     Row color = front with max votes/% or best (lowest) rank
# =====================================================
st.subheader(f"🗳️ Votes by Front in Local Bodies — {sel_assembly}")

df_votes = df_da[
    (df_da["Tier"].astype(str).str.title() == "Ward")
    & df_da["Votes"].notna()
    & df_da["LBType"].isin(["Grama", "Municipality", "Corporation"])
].copy()

if df_votes.empty:
    st.info("No Ward-tier vote rows available for this Assembly.")
else:
    # Pivot LB × Front (sum of votes)
    pivot = (
        df_votes.groupby(["LBCode", "LBName", "LBType", "Front"], as_index=False)["Votes"].sum()
        .pivot(index=["LBCode", "LBName", "LBType"], columns="Front", values="Votes")
        .fillna(0)
        .reset_index()
    )

    # Ensure all Front columns exist
    for fr in FRONT_ORDER:
        if fr not in pivot.columns:
            pivot[fr] = 0

    # Keep needed cols and sort LBs
    lb_rows = pivot[["LBName", "LBType"] + FRONT_ORDER].sort_values(["LBType", "LBName"]).reset_index(drop=True)

    # --- Determine leader per LB row (for row coloring) ---
    def _lb_leader(row: pd.Series):
        vals = {fr: row.get(fr, 0) for fr in FRONT_ORDER}
        maxv = max(vals.values()) if vals else 0
        if maxv <= 0:
            return "NONE"
        winners = [fr for fr, v in vals.items() if v == maxv]
        return winners[0] if len(winners) == 1 else "TIE"

    lb_leaders = lb_rows.apply(_lb_leader, axis=1).tolist()

    # Assembly totals / percentages / ranks (per front)
    front_totals = {fr: int(lb_rows[fr].sum()) for fr in FRONT_ORDER}
    grand_total = sum(front_totals.values())
    front_percent = {fr: (front_totals[fr] / grand_total * 100) if grand_total > 0 else 0.0 for fr in FRONT_ORDER}
    ranks_series = pd.Series(front_totals).rank(ascending=False, method="min").astype(int)
    front_rank = {fr: int(ranks_series[fr]) for fr in FRONT_ORDER}

    # --- Leaders for summary rows ---
    def _winner_from_dict(d: dict, prefer_low: bool = False):
        if not d:
            return "NONE"
        if prefer_low:
            minv = min(d.values())
            winners = [fr for fr, v in d.items() if v == minv]
            return winners[0] if len(winners) == 1 else "TIE"
        maxv = max(d.values())
        if maxv <= 0:
            return "NONE"
        winners = [fr for fr, v in d.items() if v == maxv]
        return winners[0] if len(winners) == 1 else "TIE"

    leader_total = _winner_from_dict(front_totals, prefer_low=False)
    leader_pct   = _winner_from_dict(front_percent, prefer_low=False)
    leader_rank  = _winner_from_dict(front_rank, prefer_low=True)

    # --- Build display table (values as strings with formatting) ---
    lb_rows_fmt = lb_rows.copy()
    for fr in FRONT_ORDER:
        lb_rows_fmt[fr] = lb_rows_fmt[fr].map(lambda v: f"{int(v):,}")

    summary_rows = [
        {"LBName": "Total",      "LBType": "", **{fr: f"{front_totals[fr]:,}"     for fr in FRONT_ORDER}},
        {"LBName": "Percentage", "LBType": "", **{fr: f"{front_percent[fr]:,.2f}%" for fr in FRONT_ORDER}},
        {"LBName": "Rank",       "LBType": "", **{fr: f"{front_rank[fr]}"          for fr in FRONT_ORDER}},
    ]
    summary_df = pd.DataFrame(summary_rows)

    t2_display = pd.concat([lb_rows_fmt, summary_df], ignore_index=True)

    # Row-leader series aligned to display rows
    leaders_series = pd.Series(lb_leaders + [leader_total, leader_pct, leader_rank], index=t2_display.index)

    # --- Row coloring based on leader ---
    def _row_style_by_leader(row: pd.Series):
        leader = leaders_series.loc[row.name]
        color = FRONT_BG_COLORS.get(leader, DEFAULT_BG_COLOR)
        return [f"background-color: {color}"] * len(row)

    styled_t2 = t2_display.style.apply(_row_style_by_leader, axis=1)
    render_styled_table(styled_t2)

# =====================================================
# 3) Local Body Seats (like District last table, no search) — Ward-tier winners
# =====================================================
st.subheader(f"🏘️ Front-wise Seats Won in Local Body — {sel_assembly}")

mask_lb = (
    (df_da["Tier"].astype(str).str.title() == "Ward")
    & ((df_da["Rank"] == 1) if "Rank" in df_da.columns else True)
    & (df_da["LBType"].isin(SHOW_LBTYPES))
)
df_lb = df_da[mask_lb].copy()

if df_lb.empty:
    st.info("No Ward-tier winners for the selected Assembly and LB types.")
else:
    lb_summary = (
        df_lb.groupby(["LBCode", "LBName", "LBType", "Front"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for fr in FRONT_ORDER:
        if fr not in lb_summary.columns:
            lb_summary[fr] = 0

    def _leader_row(row):
        vals = {fr: row.get(fr, 0) for fr in FRONT_ORDER}
        maxv = max(vals.values()) if vals else 0
        leaders = [fr for fr, v in vals.items() if v == maxv and maxv > 0]
        return leaders[0] if len(leaders) == 1 else ("TIE" if maxv > 0 else "NONE")

    lb_summary["Leader"] = lb_summary.apply(_leader_row, axis=1)

    if "LBCode" in lb_summary.columns:
        lb_summary["Sl No"] = lb_summary["LBCode"].astype(str).str[0] + lb_summary["LBCode"].astype(str).str[-2:]
        col_order = ["Sl No", "LBName", "LBType", "UDF", "LDF", "NDA", "OTH", "Leader"]
    else:
        col_order = ["LBName", "LBType", "UDF", "LDF", "NDA", "OTH", "Leader"]

    lb_table = lb_summary[col_order].sort_values(["LBType", "LBName"]).reset_index(drop=True)

    def _row_style_lb(row: pd.Series):
        color = FRONT_BG_COLORS.get(row["Leader"], DEFAULT_BG_COLOR)
        return [f"background-color: {color}"] * len(row)

    styled_lb = lb_table.style.apply(_row_style_lb, axis=1).format({
        "UDF": "{:,.0f}", "LDF": "{:,.0f}", "NDA": "{:,.0f}", "OTH": "{:,.0f}"
    })
    render_styled_table(styled_lb)
