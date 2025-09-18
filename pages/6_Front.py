# pages/7_Front.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import html

from lib.data import load_data, get_data_path, data_controls
from lib.colors import FRONT_BG_COLORS, PARTY_BG_COLORS, DEFAULT_BG_COLOR, FRONT_COLORS
from lib.ui import render_color_legend

# ---- Safe Styler import (pandas-version friendly) ----
try:
    from pandas.io.formats.style import Styler
except Exception:
    from typing import Any as Styler  # typing fallback

# ---------------- Page Config ----------------
st.set_page_config(page_title="Front · LSGD Explorer", page_icon="🚩", layout="wide")
st.title("🚩 Front View")

# ---------------- Sidebar: Data controls ----------------
data_controls()
df = load_data(get_data_path()).copy()

# ---------------- Hygiene ----------------
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].str.strip()

if "Votes" in df.columns:
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)

# Normalize Tier for consistent matching
if "Tier" in df.columns:
    df["TierNorm"] = df["Tier"].astype(str).str.title()

LBTYPE_ORDER = ["Grama", "Municipality", "Corporation", "Block", "District"]
FRONT_ORDER  = ["UDF", "LDF", "NDA", "OTH"]
TIERS_ORDER  = ["Ward", "Block", "District"]

# ---------------- Helpers ----------------
def _fmt_sr(v):
    """Zero-padded 00.00% format for Strike Rate cells (robust to NaN/non-numeric)."""
    try:
        val = float(v)
    except Exception:
        return "—"
    return f"{val:05.2f}%"

def render_styled_table(obj, fmt_numbers=None, fmt_perc=None):
    fmt_numbers = fmt_numbers or []
    fmt_perc    = fmt_perc or []
    styler = obj if isinstance(obj, Styler) else obj.style
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

def color_rows_uniform_front(df_display: pd.DataFrame, front: str):
    def _all_rows_front(_row: pd.Series):
        color = FRONT_BG_COLORS.get(front, DEFAULT_BG_COLOR)
        return [f"background-color: {color}"] * len(_row)
    return df_display.style.apply(_all_rows_front, axis=1)

def table_votes_share_front(scope_df: pd.DataFrame, front: str) -> pd.DataFrame:
    """Tier summary: Front Votes, Total Votes, Share % within the scope_df."""
    tiers = TIERS_ORDER
    total_by = (scope_df[scope_df["TierNorm"].isin(tiers)]
                .groupby("TierNorm", as_index=False)["Votes"].sum()
                .rename(columns={"Votes": "Total Votes"}))
    front_by = (scope_df[(scope_df["TierNorm"].isin(tiers)) & (scope_df["Front"] == front)]
                .groupby("TierNorm", as_index=False)["Votes"].sum()
                .rename(columns={"Votes": "Front Votes"}))
    out = pd.merge(pd.DataFrame({"TierNorm": tiers}), total_by, on="TierNorm", how="left").fillna({"Total Votes": 0})
    out = pd.merge(out, front_by, on="TierNorm", how="left").fillna({"Front Votes": 0})
    out["Share (%)"] = np.where(out["Total Votes"] > 0, out["Front Votes"] / out["Total Votes"] * 100, 0.0)
    out["Tier"] = pd.Categorical(out["TierNorm"], categories=tiers, ordered=True)
    out = out.sort_values("Tier").drop(columns=["TierNorm"]).rename(columns={"Tier":"Tier"})
    return out[["Tier","Front Votes","Total Votes","Share (%)"]]

def _rank_to_label(v):
    if pd.isna(v):
        return v
    try:
        n = int(float(v))
    except Exception:
        return v
    if n == 1:
        return "Won"
    suffix = "th"
    if not 11 <= (n % 100) <= 13:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def table_lbtype_performance_front(scope_df: pd.DataFrame, front: str, include_block_district: bool = False) -> pd.DataFrame:
    """
    LBType x Ranks (by Tier):
      - include_block_district=False: Ward-tier only (Grama, Municipality, Corporation)
      - include_block_district=True: add Block & District rows
    Returns: LBType | Won | 2 | 3 | ... | Contested | Strike Rate (%)
    """
    if "Rank" not in scope_df.columns or "TierNorm" not in scope_df.columns:
        return pd.DataFrame()

    tiers = ["Ward", "Block", "District"] if include_block_district else ["Ward"]
    d = scope_df[(scope_df["TierNorm"].isin(tiers)) & (scope_df["Front"] == front)].copy()
    if d.empty:
        return pd.DataFrame(columns=["LBType", "Won", "Contested", "Strike Rate (%)"])

    base_rows = ["Grama", "Municipality", "Corporation"]
    if include_block_district:
        base_rows += ["Block", "District"]

    d["LBType"] = pd.Categorical(d["LBType"], categories=base_rows, ordered=True)
    d["Rank"] = pd.to_numeric(d["Rank"], errors="coerce")
    d = d.dropna(subset=["Rank"])
    if d.empty:
        return pd.DataFrame(columns=["LBType", "Won", "Contested", "Strike Rate (%)"])
    d["Rank"] = d["Rank"].astype(int)

    xt = pd.crosstab(d["LBType"], d["Rank"]).fillna(0).astype(int)
    xt = xt.reindex(base_rows, fill_value=0)

    numeric_cols = [
        c for c in xt.columns
        if isinstance(c, (int, float, np.integer, np.floating)) and not pd.isna(c)
    ]
    rank_numeric = sorted({int(c) for c in numeric_cols})
    rename_map = {c: _rank_to_label(c) for c in numeric_cols}
    xt = xt.rename(columns=rename_map)

    rank_labels = [_rank_to_label(n) for n in rank_numeric]
    present_cols = [c for c in rank_labels if c in xt.columns]
    xt["Contested"] = xt[present_cols].sum(axis=1) if present_cols else 0
    xt["Strike Rate (%)"] = np.where(xt["Contested"] > 0, xt.get("Won", 0) / xt["Contested"] * 100, 0.0)

    col_order: list[str] = []
    if "Won" in xt.columns:
        col_order.append("Won")
    col_order.extend([c for c in rank_labels if c != "Won" and c in xt.columns])
    col_order += ["Contested", "Strike Rate (%)"]
    xt = xt[col_order].reset_index()
    return xt


def style_rows_by_party(df_display: pd.DataFrame) -> Styler:
    def _row_style(row: pd.Series):
        party = str(row.get("Party", "")).strip()
        color = PARTY_BG_COLORS.get(party, DEFAULT_BG_COLOR)
        return [f"background-color: {color}"] * len(row)
    return df_display.style.apply(_row_style, axis=1)


def table_party_lbtype_wins(scope_df: pd.DataFrame, front: str) -> pd.DataFrame:
    required = {"Front", "LBType", "Party"}
    if not required.issubset(scope_df.columns):
        return pd.DataFrame()

    d = scope_df.copy()
    if "TierNorm" not in d.columns and "Tier" in d.columns:
        d["TierNorm"] = d["Tier"].astype(str).str.title()

    d = d[(d["Front"] == front) & (d["TierNorm"].astype(str).str.title() == "Ward")]
    if "Rank" in d.columns:
        ranks = pd.to_numeric(d["Rank"], errors="coerce")
        d = d[ranks == 1]
    if d.empty:
        return pd.DataFrame()

    d["Party"] = d["Party"].fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")
    d["LBType"] = pd.Categorical(d["LBType"], categories=LBTYPE_ORDER, ordered=True)
    pivot = pd.crosstab(d["Party"], d["LBType"]).reindex(columns=LBTYPE_ORDER, fill_value=0)
    pivot = pivot.loc[:, pivot.sum(axis=0) > 0]
    if pivot.empty:
        return pd.DataFrame()

    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False).reset_index().rename(columns={"Party": "Party"})
    return pivot


def table_party_rank_summary(scope_df: pd.DataFrame, front: str) -> pd.DataFrame:
    required = {"Front", "Party", "Rank"}
    if not required.issubset(scope_df.columns):
        return pd.DataFrame()

    d = scope_df.copy()
    if "TierNorm" not in d.columns and "Tier" in d.columns:
        d["TierNorm"] = d["Tier"].astype(str).str.title()

    d = d[(d["Front"] == front) & (d["TierNorm"].astype(str).str.title() == "Ward")]
    if d.empty:
        return pd.DataFrame()

    d["Party"] = d["Party"].fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")
    d["Rank"] = pd.to_numeric(d["Rank"], errors="coerce")
    d = d.dropna(subset=["Rank"])
    if d.empty:
        return pd.DataFrame()
    d["Rank"] = d["Rank"].astype(int)

    cross = pd.crosstab(d["Party"], d["Rank"]).astype(int)
    numeric_cols = [c for c in cross.columns if isinstance(c, (int, np.integer))]
    rank_labels = [_rank_to_label(n) for n in sorted(numeric_cols)]
    rename_map = {c: _rank_to_label(c) for c in numeric_cols}
    cross = cross.rename(columns=rename_map)

    ordered_cols = []
    if "Won" in cross.columns:
        ordered_cols.append("Won")
    ordered_cols.extend([c for c in rank_labels if c != "Won" and c in cross.columns])
    cross["Contested"] = cross[ordered_cols].sum(axis=1) if ordered_cols else 0
    cross["Strike Rate (%)"] = np.where(cross["Contested"] > 0, cross.get("Won", 0) / cross["Contested"] * 100, 0.0)

    ordered_cols += ["Contested", "Strike Rate (%)"]
    result = cross[ordered_cols].reset_index().rename(columns={"index": "Party"})
    sort_cols = [c for c in ["Won", "Contested"] if c in result.columns]
    if sort_cols:
        result = result.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
    return result.reset_index(drop=True)

# ---------- Strength helpers ----------
_STRENGTH_ORDER = [
    "-500 or less", "-200 to -499", "-100 to -199", "-50 to -99", "-1 to -49",
    "0", "1-49", "50-99", "100-199", "200-499", "500+"
]


STRENGTH_COLOR_MAP = {
    "-500 or less": "#f8d7da",
    "-200 to -499": "#f9d6dc",
    "-100 to -199": "#fbd2d9",
    "-50 to -99": "#fde2e4",
    "-1 to -49": "#fff1e6",
    "0": "#f8f9fa",
    "1-49": "#e8f6ef",
    "50-99": "#d4f3e4",
    "100-199": "#bcefd4",
    "200-499": "#a3e8c4",
    "500+": "#8bdcb3",
}

WON_NAME_COLOR = "#2e7d32"
NOT_WON_NAME_COLOR = "#c62828"


def blend_hex(color_from: str, color_to: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, float(ratio)))
    def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    c1 = _hex_to_rgb(color_from)
    c2 = _hex_to_rgb(color_to)
    blended = tuple(int(c1[i] + (c2[i] - c1[i]) * ratio) for i in range(3))
    return _rgb_to_hex(blended)


def style_strength_table(df_display: pd.DataFrame, band_col: str = "Strength Band") -> Styler:
    def _row_style(row: pd.Series):
        band = str(row.get(band_col, ""))
        color = STRENGTH_COLOR_MAP.get(band, DEFAULT_BG_COLOR)
        return [f"background-color: {color}"] * len(row)
    return df_display.style.apply(_row_style, axis=1)


def format_vote_names(names: list[str], color: str) -> str:
    if not names:
        return "—"
    return ", ".join(
        f"<span style='color:{color};font-weight:600'>" + html.escape(name) + "</span>"
        for name in names
    )


def style_lead_table(df_display: pd.DataFrame, value_col: str, positive: bool = True) -> Styler:
    values = pd.to_numeric(df_display[value_col], errors='coerce')
    max_val = values.replace([np.inf, -np.inf], np.nan).abs().max()
    if not pd.notna(max_val) or max_val <= 0:
        return df_display.style
    target = '#2e7d32' if positive else '#c62828'
    def _row_style(row: pd.Series):
        value = pd.to_numeric(row.get(value_col), errors='coerce')
        if not pd.notna(value):
            return [f"background-color: {DEFAULT_BG_COLOR}"] * len(row)
        ratio = min(abs(float(value)) / max_val, 1.0)
        color = blend_hex('#ffffff', target, ratio)
        return [f"background-color: {color}"] * len(row)
    return df_display.style.apply(_row_style, axis=1)




def style_share_margin_table(df_display: pd.DataFrame, share_col: str, threshold: float = 50.0, stronger: bool = True) -> Styler:
    values = pd.to_numeric(df_display.get(share_col), errors="coerce")
    if values.dropna().empty:
        return df_display.style
    deltas = (values - threshold) if stronger else (threshold - values)
    deltas = deltas.where(deltas > 0, 0)
    max_val = deltas.replace([np.inf, -np.inf], np.nan).max()
    if not pd.notna(max_val) or max_val <= 0:
        return df_display.style
    target = "#2e7d32" if stronger else "#c62828"

    def _row_style(row: pd.Series):
        value = pd.to_numeric(row.get(share_col), errors="coerce")
        if not pd.notna(value):
            return [f"background-color: {DEFAULT_BG_COLOR}"] * len(row)
        delta = (value - threshold) if stronger else (threshold - value)
        if delta <= 0:
            return [f"background-color: {DEFAULT_BG_COLOR}"] * len(row)
        ratio = min(float(delta) / max_val, 1.0)
        color = blend_hex("#ffffff", target, ratio)
        return [f"background-color: {color}"] * len(row)

    return df_display.style.apply(_row_style, axis=1)


def _lead_to_strength(lead: float | int | None) -> str | None:
    if pd.isna(lead):
        return None
    try:
        x = float(lead)
    except Exception:
        return None
    if x <= -500:           return "-500 or less"
    if -500 < x <= -200:    return "-200 to -499"
    if -200 < x <= -100:    return "-100 to -199"
    if -100 < x <= -50:     return "-50 to -99"
    if -50  < x <= -1:      return "-1 to -49"
    if x == 0:              return "0"
    if 0   < x <= 49:       return "1-49"
    if 50  <= x <= 99:      return "50-99"
    if 100 <= x <= 199:     return "100-199"
    if 200 <= x <= 499:     return "200-499"
    if x >= 500:            return "500+"
    return None

def _build_strength_chart_front(scope_df: pd.DataFrame, sel_front: str):
    d = scope_df.copy()
    d = d[(d["TierNorm"] == "Ward") & (d["Front"] == sel_front)]
    if d.empty:
        return None

    if "Strength" in d.columns and d["Strength"].notna().any():
        s = d["Strength"].astype(str)
    elif "Lead" in d.columns:
        s = d["Lead"].apply(_lead_to_strength)
    else:
        return None

    strength_summary = (
        pd.DataFrame({"Strength": s})
        .dropna()
        .groupby("Strength", observed=True, as_index=False)
        .size().rename(columns={"size": "Wards"})
    )
    if strength_summary.empty:
        return None

    strength_summary["Strength"] = pd.Categorical(strength_summary["Strength"], categories=_STRENGTH_ORDER, ordered=True)
    strength_summary = strength_summary.sort_values("Strength")

    mirror_df = strength_summary.copy()
    mirror_df["Display_Wards"] = mirror_df.apply(
        lambda row: -row["Wards"] if str(row["Strength"]).startswith("-") else row["Wards"], axis=1
    )
    mirror_df["Status"] = mirror_df["Strength"].apply(
        lambda x: "Lost" if str(x).startswith("-") else "Won"
    )

    fig = px.bar(
        mirror_df,
        x="Display_Wards",
        y="Strength",
        orientation="h",
        text="Wards",
        color="Status",
        color_discrete_map={"Won": "#6c80ac", "Lost": "#cc807c"},
        title=""
    )
    fig.update_layout(
        xaxis_title="Number of Wards",
        yaxis_title="Strength Category",
        height=560,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
        legend=dict(title="Status", orientation="h", x=1.0, y=0, xanchor="right", yanchor="bottom")
    )
    return fig

def _vote_bin_order(labels: list[str]) -> list[str]:
    def key(lbl: str):
        if lbl is None: return (999999, lbl)
        s = str(lbl)
        if s.startswith("<"): return (-1, s)
        if s.endswith("+"):
            m = re.match(r"(\d+)", s)
            return (int(m.group(1)) if m else 999998, s)
        m = re.match(r"(\d+)", s)
        return (int(m.group(1)) if m else 999999, s)
    return sorted(labels, key=key)

def _build_vote_bin_chart_front(scope_df: pd.DataFrame, sel_front: str):
    d = scope_df.copy()
    d = d[(d["TierNorm"] == "Ward") & (d["Front"] == sel_front)]
    if d.empty or "VoteBin" not in d.columns:
        return None

    d["Status"] = np.where(d.get("Rank", 0).astype("Int64") == 1, "Won", "Not won")
    agg = (d.groupby(["VoteBin", "Status"], as_index=False)
             .size().rename(columns={"size": "Wards"}))

    bins = _vote_bin_order(agg["VoteBin"].astype(str).unique().tolist())
    agg["VoteBin"] = pd.Categorical(agg["VoteBin"].astype(str), categories=bins, ordered=True)
    agg = agg.sort_values("VoteBin")

    fig = px.bar(
        agg, x="VoteBin", y="Wards", color="Status",
        barmode="stack", text="Wards",
        color_discrete_map={"Won": "#6c80ac", "Not won": "#cc807c"},
        title=""
    )
    fig.update_layout(
        xaxis_title="VoteBin",
        yaxis_title="Number of Wards",
        height=420,
        legend=dict(title="Status", orientation="h", x=1.0, y=1.02, xanchor="right", yanchor="bottom")
    )
    return fig

# --------- Winner/Runner join keys + opponent table (FRONT) ---------
def _ward_join_keys(df_: pd.DataFrame) -> list[str]:
    if "WardCode" in df_.columns:
        return ["WardCode"]
    if "WardNo" in df_.columns:
        cols = [c for c in ["District", "LBName", "WardNo"] if c in df_.columns]
        if cols:
            return cols
    cols = [c for c in ["District", "LBName", "WardName"] if c in df_.columns]
    return cols or ["WardName"]

def table_opponent_front(scope_df: pd.DataFrame, sel_front: str) -> pd.DataFrame:
    """
    Front-based opponent breakdown:
      Opponent Front | Runner-up (when Selected Won) | Winners (when Selected Second)
    Ward-tier only.
    """
    if "Rank" not in scope_df.columns or "Front" not in scope_df.columns:
        return pd.DataFrame(columns=["Front", "Runner-up (when Selected Won)", "Winners (when Selected Second)"])

    d = scope_df.copy()
    d["TierNorm"] = d["Tier"].astype(str).str.title() if "TierNorm" not in d.columns else d["TierNorm"]
    d["Rank"] = pd.to_numeric(d["Rank"], errors="coerce").astype("Int64")
    d = d[d["TierNorm"] == "Ward"]
    if d.empty:
        return pd.DataFrame(columns=["Front", "Runner-up (when Selected Won)", "Winners (when Selected Second)"])

    keys = _ward_join_keys(d)
    winners = d[d["Rank"] == 1][keys + ["Front"]].rename(columns={"Front": "WinnerFront"})
    runners = d[d["Rank"] == 2][keys + ["Front"]].rename(columns={"Front": "RunnerFront"})

    # Selected front wins -> count runner-up fronts
    wins_sel = winners[winners["WinnerFront"] == sel_front]
    ru_vs_selwin = (wins_sel.merge(runners, on=keys, how="left")
                    .groupby("RunnerFront", dropna=True).size().rename("Runner-up (when Selected Won)"))
    ru_vs_selwin.index = ru_vs_selwin.index.fillna("UNKNOWN")

    # Selected front second -> count winner fronts
    sec_sel = runners[runners["RunnerFront"] == sel_front]
    win_vs_selsec = (sec_sel.merge(winners, on=keys, how="left")
                     .groupby("WinnerFront", dropna=True).size().rename("Winners (when Selected Second)"))
    win_vs_selsec.index = win_vs_selsec.index.fillna("UNKNOWN")

    all_fronts = sorted(set(ru_vs_selwin.index.tolist()) | set(win_vs_selsec.index.tolist()), key=lambda x: str(x))
    out = pd.DataFrame({"Front": all_fronts})
    out = out.merge(ru_vs_selwin.reset_index().rename(columns={"RunnerFront": "Front"}), on="Front", how="left")
    out = out.merge(win_vs_selsec.reset_index().rename(columns={"WinnerFront": "Front"}), on="Front", how="left")
    out[["Runner-up (when Selected Won)", "Winners (when Selected Second)"]] = \
        out[["Runner-up (when Selected Won)", "Winners (when Selected Second)"]].fillna(0).astype(int)

    out["Total"] = out["Runner-up (when Selected Won)"] + out["Winners (when Selected Second)"]
    out = out.sort_values(["Total", "Front"], ascending=[False, True]).drop(columns=["Total"]).reset_index(drop=True)
    return out

# ---------------- Filters (Front) ----------------
st.markdown("### Filters")
sel_front = st.selectbox("Front", FRONT_ORDER, index=0)  # default UDF
st.markdown("---")

# ---------------- Tabs ----------------
tab_d, tab_a, tab_l = st.tabs(["🏙️ District", "🏛️ Assembly", "🏘️ Local Body"])

# ---------- District Tab ----------
with tab_d:
    st.markdown("#### Scope")
    districts = ["All Kerala"] + sorted(df["District"].dropna().unique().tolist(), key=lambda x: str(x))
    sel_district = st.selectbox("District", districts, index=0, key="front_tab_district")  # default All Kerala

    if sel_district == "All Kerala":
        scoped = df.copy()
        scope_label = "**All Kerala**"
    else:
        scoped = df[df["District"] == sel_district].copy()
        scope_label = f"**{sel_district}**"

    # TABLE 1: Votes & Share
    st.subheader(f"🧮 {sel_front} — Votes & Vote Share by Tier ({scope_label})")
    t_votes = table_votes_share_front(scoped, sel_front)
    styled_votes = color_rows_uniform_front(t_votes, sel_front)
    render_styled_table(styled_votes, fmt_numbers=["Front Votes","Total Votes"], fmt_perc=["Share (%)"])

    # TABLE 2: Seats & Ranks by LBType (Ward/Block/District)
    st.subheader(f"🏆 {sel_front} — Seats & Ranks by LBType (Ward/Block/District) ({scope_label})")
    t_perf = table_lbtype_performance_front(scoped, sel_front, include_block_district=True)
    if t_perf.empty:
        st.info("No Rank data in this scope.")
    else:
        fmt_nums = [c for c in t_perf.columns if c not in ["LBType", "Strike Rate (%)"]]
        styled_perf = (
            t_perf.style
            .apply(lambda r: [f"background-color: {FRONT_BG_COLORS.get(sel_front, DEFAULT_BG_COLOR)}"] * len(r), axis=1)
            .format({**{c: "{:,.0f}" for c in fmt_nums}, "Strike Rate (%)": _fmt_sr})
        )
        render_styled_table(styled_perf)
        # Legend for selected Front color
        render_color_legend({sel_front: FRONT_BG_COLORS.get(sel_front, DEFAULT_BG_COLOR)}, title="Row color")

    st.subheader(f"{sel_front} - Party-wise Ward Wins by LBType ({scope_label})")
    t_party_wins = table_party_lbtype_wins(scoped, sel_front)
    if t_party_wins.empty:
        st.info("No ward winners available for this scope.")
    else:
        num_cols = [c for c in t_party_wins.columns if c != "Party"]
        styled_party = style_rows_by_party(t_party_wins)
        render_styled_table(styled_party, fmt_numbers=num_cols)
        legend_map = {p: PARTY_BG_COLORS.get(p, DEFAULT_BG_COLOR) for p in t_party_wins["Party"].astype(str).tolist()}
        if legend_map:
            render_color_legend(legend_map, title="Party colors")


    # OPPONENT BREAKDOWN by FRONT — TABLE
    st.subheader(f"🤝 Opponent Breakdown — {sel_front} ({scope_label})")
    t_opp = table_opponent_front(scoped, sel_front)
    if t_opp.empty:
        st.info("No Ward-tier winner/runner data available for this scope.")
    else:
        def _row_front_color(row: pd.Series):
            fkey = str(row.get("Front","")).strip()
            color = FRONT_BG_COLORS.get(fkey, DEFAULT_BG_COLOR)
            return [f"background-color: {color}"] * len(row)

        styled_opp = (
            t_opp.style
            .apply(_row_front_color, axis=1)
            .format({
                "Runner-up (when Selected Won)": "{:,.0f}",
                "Winners (when Selected Second)": "{:,.0f}"
            })
        )
        render_styled_table(styled_opp)

    # STRENGTH (mirror) CHART
    st.subheader(f"📶 {sel_front} — Number of Strong and Weak Wards ({scope_label})")
    fig_strength = _build_strength_chart_front(scoped, sel_front)
    if fig_strength is None:
        st.info("No Strength/Lead data available for this scope.")
    else:
        st.plotly_chart(fig_strength, use_container_width=True)

    # VOTEBIN STACKED BAR
    st.subheader(f"📊 {sel_front} — VoteBin vs Wards (Won + Not won) ({scope_label})")
    fig_vote = _build_vote_bin_chart_front(scoped, sel_front)
    if fig_vote is None:
        st.info("No VoteBin data available for this scope.")
    else:
        st.plotly_chart(fig_vote, use_container_width=True)

# ---------- Assembly Tab ----------
with tab_a:
    st.markdown("#### Scope")
    asm_cols = ["Assembly", "ACName", "AssemblyName", "Constituency"]
    asm_col  = next((c for c in asm_cols if c in df.columns), None)
    if not asm_col:
        st.info("No assembly column found.")
    else:
        districts = sorted(df["District"].dropna().unique().tolist(), key=lambda x: str(x))
        default_dix = next((i for i, d in enumerate(districts) if str(d).strip().lower() == "malappuram"), 0)
        c1, c2 = st.columns([1, 1])
        with c1:
            sel_district_a = st.selectbox("District", districts, index=default_dix, key="front_tab_a_district")
        df_d = df[df["District"] == sel_district_a]
        assemblies = sorted(df_d[asm_col].dropna().unique().tolist(), key=lambda x: str(x))
        default_ax = next((i for i, a in enumerate(assemblies) if "malappuram" in str(a).strip().lower()), 0)
        with c2:
            sel_assembly = st.selectbox("Assembly", assemblies, index=(default_ax if assemblies else 0), key="front_tab_a_assembly")

        scoped = df[(df["District"] == sel_district_a) & (df[asm_col] == sel_assembly)].copy()
        scope_label = f"**{sel_assembly}**"

        # TABLE 1: Votes & Share in this Assembly (Ward-tier)
        st.subheader(f"🧮 {sel_front} — Votes & Share in Assembly (Ward-tier) ({scope_label})")
        asm_ward = scoped[scoped["TierNorm"] == "Ward"].copy()
        total_votes = int(asm_ward["Votes"].sum()) if "Votes" in asm_ward.columns else 0
        front_votes = int(asm_ward.loc[asm_ward["Front"] == sel_front, "Votes"].sum()) if "Votes" in asm_ward.columns else 0
        share = (front_votes / total_votes * 100) if total_votes > 0 else 0.0

        t_votes_asm = pd.DataFrame({
            "Front Votes": [front_votes],
            "Total Votes": [total_votes],
            "Share (%)":   [share],
        })
        styled_votes_asm = color_rows_uniform_front(t_votes_asm, sel_front)
        render_styled_table(styled_votes_asm, fmt_numbers=["Front Votes", "Total Votes"], fmt_perc=["Share (%)"])

        # TABLE 2: LBName × Ranks (Won, 2, 3, ...), Contested, Strike Rate — TOTAL row
        st.subheader(f"🏆 {sel_front} — Performance by Local Body (Ward-tier) ({scope_label})")
        df_p = asm_ward[asm_ward["Front"] == sel_front].copy()
        if df_p.empty or "Rank" not in df_p.columns or "LBName" not in df_p.columns:
            st.info("No Ward-tier Rank data for this front in the selected assembly.")
        else:
            df_p["Rank"] = pd.to_numeric(df_p["Rank"], errors="coerce").astype("Int64")
            ct = pd.crosstab(df_p["LBName"], df_p["Rank"]).fillna(0).astype(int)
            # Rename numeric rank columns to labels: 1->Won, 2->2nd, 3->3rd, ...
            all_ranks = sorted([int(c) for c in ct.columns if isinstance(c, (int, float, np.integer, np.floating))])
            ct = ct.rename(columns={c: _rank_to_label(c) for c in ct.columns})
            ordered_cols = (["Won"] if "Won" in ct.columns else []) + [
                _rank_to_label(r) for r in all_ranks if r != 1 and _rank_to_label(r) in ct.columns
            ]
            ct["Contested"] = ct[ordered_cols].sum(axis=1) if ordered_cols else 0
            ct["Strike Rate (%)"] = np.where(ct["Contested"] > 0, ct.get("Won", 0) / ct["Contested"] * 100, 0.0)
            t2 = ct[ordered_cols + ["Contested", "Strike Rate (%)"]].reset_index()

            # TOTAL row
            total_row = {col: 0 for col in t2.columns if col not in ["LBName", "Strike Rate (%)"]}
            for col in total_row.keys():
                total_row[col] = int(t2[col].sum())
            total_won = total_row.get("Won", 0)
            total_cont = total_row.get("Contested", 0)
            total_row["LBName"] = "Total"
            total_row["Strike Rate (%)"] = (total_won / total_cont * 100) if total_cont > 0 else 0.0

            t2 = pd.concat([t2.sort_values("LBName"), pd.DataFrame([total_row])], ignore_index=True)

            fmt_nums = [c for c in t2.columns if c not in ["LBName", "Strike Rate (%)"]]
            styled_t2 = (
                t2.style
                .apply(lambda r: [f"background-color: {FRONT_BG_COLORS.get(sel_front, DEFAULT_BG_COLOR)}"] * len(r), axis=1)
                .format({**{c: "{:,.0f}" for c in fmt_nums}, "Strike Rate (%)": _fmt_sr})
            )
            render_styled_table(styled_t2)
            render_color_legend({sel_front: FRONT_BG_COLORS.get(sel_front, DEFAULT_BG_COLOR)}, title="Row color")

        st.subheader(f"{sel_front} - Party-wise Ward Performance ({scope_label})")
        t_party_summary = table_party_rank_summary(scoped, sel_front)
        if t_party_summary.empty:
            st.info("No ward-tier party performance data for this scope.")
        else:
            num_cols = [c for c in t_party_summary.columns if c not in ["Party", "Strike Rate (%)"]]
            styled_party = style_rows_by_party(t_party_summary)
            render_styled_table(styled_party, fmt_numbers=num_cols, fmt_perc=["Strike Rate (%)"])
            legend_map = {p: PARTY_BG_COLORS.get(p, DEFAULT_BG_COLOR) for p in t_party_summary["Party"].astype(str).tolist()}
            if legend_map:
                render_color_legend(legend_map, title="Party colors")


        # OPPONENT BREAKDOWN by FRONT — TABLE
        st.subheader(f"🤝 {sel_front} — Opponent Breakdown ({scope_label})")
        t_opp_a = table_opponent_front(scoped, sel_front)
        if t_opp_a is None or t_opp_a.empty:
            st.info("No Ward-tier winner/runner data available for this scope.")
        else:
            def _row_front_color_a(row: pd.Series):
                fkey = str(row.get("Front","")).strip()
                color = FRONT_BG_COLORS.get(fkey, DEFAULT_BG_COLOR)
                return [f"background-color: {color}"] * len(row)

            styled_opp_a = (
                t_opp_a.style
                .apply(_row_front_color_a, axis=1)
                .format({
                    "Runner-up (when Selected Won)": "{:,.0f}",
                    "Winners (when Selected Second)": "{:,.0f}"
                })
            )
            render_styled_table(styled_opp_a)

        # STRENGTH (mirror) CHART
        st.subheader(f"📶 {sel_front} — Number of Strong and Weak Wards ({scope_label})")
        fig_strength_a = _build_strength_chart_front(scoped, sel_front)
        if fig_strength_a is None:
            st.info("No Strength/Lead data available for this scope.")
        else:
            st.plotly_chart(fig_strength_a, use_container_width=True)

        # VOTEBIN STACKED BAR
        st.subheader(f"📊 {sel_front} — VoteBin vs Wards (Won + Not won) ({scope_label})")
        fig_vote_a = _build_vote_bin_chart_front(scoped, sel_front)
        if fig_vote_a is None:
            st.info("No VoteBin data available for this scope.")
        else:
            st.plotly_chart(fig_vote_a, use_container_width=True)

        if asm_ward.empty or "Votes" not in asm_ward.columns:
            st.info("No ward-level vote data available for this assembly.")
        else:
            front_contested = asm_ward[asm_ward["Front"] == sel_front].copy()
            if front_contested.empty:
                st.info(f"{sel_front} did not contest any wards in this assembly.")
            else:
                key_cols = _ward_join_keys(asm_ward) or ["WardName"]
                total_by_ward = asm_ward.groupby(key_cols)["Votes"].sum().rename("Total Votes")
                front_by_ward = front_contested.groupby(key_cols)["Votes"].sum().rename("Front Votes")
                ward_summary = front_by_ward.to_frame().join(total_by_ward, how="left")
                ward_summary = ward_summary[ward_summary["Total Votes"] > 0]
                if ward_summary.empty:
                    st.info("No ward-level totals available to compute vote shares.")
                else:
                    summary = ward_summary.reset_index()
                    meta_candidates = [c for c in ["LBName", "WardName", "WardNo", "WardCode"] if c in asm_ward.columns]
                    extra_cols = [c for c in meta_candidates if c not in summary.columns]
                    if extra_cols:
                        meta_df = asm_ward[[*key_cols, *extra_cols]].drop_duplicates(key_cols)
                        summary = summary.merge(meta_df, on=key_cols, how="left")
                    summary["Front Votes"] = summary["Front Votes"].astype(int)
                    summary["Total Votes"] = summary["Total Votes"].astype(int)
                    summary["Share (%)"] = np.where(
                        summary["Total Votes"] > 0,
                        summary["Front Votes"] / summary["Total Votes"] * 100,
                        0.0,
                    )
                    column_order = []
                    for col in ["LBName", "WardName", "WardNo", "WardCode"]:
                        if col in summary.columns:
                            column_order.append(col)
                    column_order += ["Front Votes", "Total Votes", "Share (%)"]
                    column_order = list(dict.fromkeys(column_order))
                    strongest = summary[summary["Share (%)"] > 50].sort_values(
                        ["Share (%)", "Front Votes"], ascending=[False, False]
                    ).head(20)
                    weakest = summary[summary["Share (%)"] < 50].sort_values(
                        ["Share (%)", "Front Votes"], ascending=[True, False]
                    ).head(20)

                    def _render_ward_list(frame: pd.DataFrame, title: str, empty_message: str, stronger: bool) -> None:
                        st.subheader(title)
                        if frame.empty:
                            st.info(empty_message)
                            return
                        table = frame[column_order].copy()
                        for col in ["Front Votes", "Total Votes"]:
                            if col in table.columns:
                                table[col] = table[col].astype(int)
                        fmt_nums = [c for c in ["Front Votes", "Total Votes"] if c in table.columns]
                        if "Share (%)" in table.columns:
                            styled_table = style_share_margin_table(table, "Share (%)", stronger=stronger)
                        else:
                            styled_table = table.style
                        render_styled_table(styled_table, fmt_numbers=fmt_nums, fmt_perc=["Share (%)"])

                    _render_ward_list(
                        strongest,
                        f"{sel_front} - Strongest Wards ({scope_label})",
                        f"No wards where {sel_front} crossed 50% vote share in this assembly.",
                        True,
                    )
                    _render_ward_list(
                        weakest,
                        f"{sel_front} - Weakest Wards ({scope_label})",
                        f"No wards where {sel_front} fell below 50% vote share in this assembly.",
                        False,
                    )


# ---------- Local Body Tab ----------
with tab_l:
    st.markdown("#### Scope")
    districts = sorted(df["District"].dropna().unique().tolist(), key=lambda x: str(x))
    default_dix = next((i for i, d in enumerate(districts) if str(d).strip().lower() == "malappuram"), 0)
    c1, c2 = st.columns([1, 1])
    with c1:
        sel_district_l = st.selectbox("District", districts, index=default_dix, key="front_tab_l_district")
    df_d = df[df["District"] == sel_district_l]
    lbnames = sorted(df_d["LBName"].dropna().unique().tolist(), key=lambda x: str(x))
    with c2:
        sel_lb = st.selectbox("Local Body", lbnames, index=0 if lbnames else 0, key="front_tab_l_lb")

    scoped = df[(df["District"] == sel_district_l) & (df["LBName"] == sel_lb)].copy()
    scope_label = f"**{sel_lb}**"

    # Restrict to Ward-tier
    lb_ward = scoped[scoped["TierNorm"] == "Ward"].copy()
    lb_front = lb_ward[lb_ward["Front"] == sel_front].copy()

    # ===== SUMMARY =====
    st.subheader(f"🧮 Summary — {sel_front} in {scope_label}")
    if lb_ward.empty:
        st.info("No Ward-tier data for this local body.")
    else:
        total_votes = int(lb_ward["Votes"].sum()) if "Votes" in lb_ward.columns else 0
        front_votes = int(lb_front["Votes"].sum()) if "Votes" in lb_front.columns else 0
        share = (front_votes / total_votes * 100) if total_votes > 0 else 0.0
        contested = len(lb_front)
        won = int((lb_front.get("Rank", pd.Series(dtype="Int64")).astype("Int64") == 1).sum()) if "Rank" in lb_front.columns else 0

        front_hex = FRONT_COLORS.get(sel_front, "#222")
        st.markdown(
            f"""
            <div>
              <span style="font-weight:700;color:{front_hex}">{sel_front}</span>
              secured <span style="font-weight:700">{front_votes:,}</span> votes
              (<span style="font-weight:700">{share:.2f}%</span>) out of
              <span style="font-weight:700">{total_votes:,}</span> total votes.
            </div>
            <div>
              Won <span style="font-weight:700">{won:,}</span> seats out of
              <span style="font-weight:700">{contested:,}</span> contested.
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.subheader(f"{sel_front} - Party-wise Ward Performance ({scope_label})")
    t_party_summary_lb = table_party_rank_summary(scoped, sel_front)
    if t_party_summary_lb.empty:
        st.info("No ward-tier party performance data for this local body.")
    else:
        num_cols = [c for c in t_party_summary_lb.columns if c not in ["Party", "Strike Rate (%)"]]
        styled_party_lb = style_rows_by_party(t_party_summary_lb)
        render_styled_table(styled_party_lb, fmt_numbers=num_cols, fmt_perc=["Strike Rate (%)"])
        legend_map = {p: PARTY_BG_COLORS.get(p, DEFAULT_BG_COLOR) for p in t_party_summary_lb["Party"].astype(str).tolist()}
        if legend_map:
            render_color_legend(legend_map, title="Party colors")



    # ===== STRENGTH ANALYSIS (Lead/Trail split, show all categories present) =====
    st.subheader("📶 Strength Analysis")
    if lb_front.empty:
        st.info("No rows for the selected front in this local body.")
    else:
        if "Strength" in lb_front.columns and lb_front["Strength"].notna().any():
            s_series = lb_front["Strength"].astype(str)
        elif "Lead" in lb_front.columns:
            s_series = lb_front["Lead"].apply(_lead_to_strength)
        else:
            s_series = pd.Series(dtype="object")

        if s_series.dropna().empty or "WardName" not in lb_front.columns:
            st.info("No Strength/Lead or WardName data available.")
        else:
            s_df = pd.DataFrame({"Strength": s_series, "WardName": lb_front["WardName"]}).dropna(subset=["Strength"])
            s_df["Strength"] = pd.Categorical(s_df["Strength"], categories=_STRENGTH_ORDER, ordered=True)
            agg = (
                s_df.groupby("Strength", observed=True, as_index=False)
                .agg(
                    Wards=("WardName", "count"),
                    Names=("WardName", lambda x: ", ".join(sorted(map(str, x.unique()))))
                )
                .sort_values("Strength")
            )

            is_trail = agg["Strength"].astype(str).str.startswith("-")
            is_zero = agg["Strength"].astype(str).eq("0")
            lead_tbl = agg[~is_trail & ~is_zero]
            trail_tbl = agg[is_trail]

            def _render_strength_table(frame: pd.DataFrame, title: str) -> None:
                st.markdown(f"**{title}**")
                if frame.empty:
                    st.caption(f"No {title.lower()} categories.")
                    return
                table = frame.copy()
                table["Wards"] = table["Wards"].astype(int)
                table = table.rename(columns={"Strength": "Strength Band"})
                styled = style_strength_table(table)
                render_styled_table(styled, fmt_numbers=["Wards"])

            _render_strength_table(lead_tbl, "Lead")
            _render_strength_table(trail_tbl, "Trail")

    # ===== VOTEBIN LIST (color ward names by win/loss) =====
    st.subheader("🧊 VoteBin Summary (Won/Not won names colour-coded)")
    if lb_front.empty or "VoteBin" not in lb_front.columns or "WardName" not in lb_front.columns:
        st.info("VoteBin or WardName not available for the selected front.")
    else:
        tmp = lb_front.copy()
        if "Rank" in tmp.columns:
            rank_series = tmp["Rank"].astype("Int64")
        else:
            rank_series = pd.Series(pd.NA, index=tmp.index, dtype="Int64")
        tmp["Status"] = np.where(rank_series.fillna(0) == 1, "Won", "Not won")
        bins = _vote_bin_order(tmp["VoteBin"].astype(str).unique().tolist())
        tmp["VoteBinStr"] = pd.Categorical(tmp["VoteBin"].astype(str), categories=bins, ordered=True)
        group_cols = ["WardName"]
        if "Rank" in tmp.columns:
            group_cols.append("Rank")

        def _summarize_vote_bin(g: pd.DataFrame) -> dict:
            ranks = g["Rank"].astype("Int64") if "Rank" in g.columns else pd.Series(pd.NA, index=g.index, dtype="Int64")
            won = []
            not_won = []
            for nm, rk in zip(g["WardName"], ranks):
                name = str(nm)
                if pd.notna(rk) and rk == 1:
                    won.append(name)
                else:
                    not_won.append(name)
            return {"count": int(len(g)), "won": sorted(won), "not": sorted(not_won)}

        grp = (
            tmp.groupby("VoteBinStr", observed=True)[group_cols]
            .apply(_summarize_vote_bin)
            .reset_index(name="data")
        )

        if grp.empty:
            st.info("No VoteBin data available for this front.")
        else:
            table = grp.copy()
            table["Count"] = table["data"].apply(lambda d: int(d["count"]))
            table["Won Names"] = table["data"].apply(lambda d: format_vote_names(d["won"], WON_NAME_COLOR))
            table["Not Won Names"] = table["data"].apply(lambda d: format_vote_names(d["not"], NOT_WON_NAME_COLOR))
            table = table.rename(columns={"VoteBinStr": "VoteBin"})
            table = table[["VoteBin", "Count", "Won Names", "Not Won Names"]]
            styled = color_rows_uniform_front(table, sel_front)
            styled = styled.format({"Won Names": lambda x: x, "Not Won Names": lambda x: x}, escape=False)
            render_styled_table(styled, fmt_numbers=["Count"])
    # ===== OPPONENT BREAKDOWN by FRONT =====
    st.subheader(f"🤝 Opponent Breakdown — {sel_front} ({scope_label})")
    t_opp_l = table_opponent_front(lb_ward, sel_front)
    if t_opp_l.empty:
        st.info("No Ward-tier winner/runner data available for this local body.")
    else:
        def _row_front_color_l(row: pd.Series):
            fkey = str(row.get("Front","")).strip()
            color = FRONT_BG_COLORS.get(fkey, DEFAULT_BG_COLOR)
            return [f"background-color: {color}"] * len(row)

        styled_opp_l = (
            t_opp_l.style
            .apply(_row_front_color_l, axis=1)
            .format({
                "Runner-up (when Selected Won)": "{:,.0f}",
                "Winners (when Selected Second)": "{:,.0f}"
            })
        )
        render_styled_table(styled_opp_l)

    # ===== WINNING CANDIDATES (details) =====
    st.subheader(f"🏅 Winning Candidates - {sel_front}")
    if lb_ward.empty or "Rank" not in lb_ward.columns:
        st.info("No Rank data available.")
    else:
        keys = _ward_join_keys(lb_ward)
        winners = lb_ward[(lb_ward["Front"] == sel_front) & (lb_ward["Rank"].astype("Int64") == 1)].copy()
        if winners.empty:
            st.info("No winning wards for the selected front here.")
        else:
            totals = (lb_ward.groupby(keys, as_index=False)["Votes"].sum().rename(columns={"Votes": "TotalVotes"}))
            runners = lb_ward[lb_ward["Rank"].astype("Int64") == 2][keys + ["Front", "Party", "Candidate", "Votes"]]
            runners = runners.rename(columns={
                "Front": "Trailing Front", "Party": "Trailing Party",
                "Candidate": "Trailing Candidate", "Votes": "RunnerVotes"
            })
            w = winners.merge(totals, on=keys, how="left").merge(runners, on=keys, how="left")
            w["Vote share (%)"] = np.where(w["TotalVotes"] > 0, w["Votes"] / w["TotalVotes"] * 100, 0.0)
            if "Lead" in w.columns:
                w["Lead"] = w["Lead"].fillna(w["Votes"] - w.get("RunnerVotes", 0))
            else:
                w["Lead"] = w["Votes"] - w.get("RunnerVotes", 0)

            winners_tbl = pd.DataFrame({
                "Ward name": w.get("WardName", pd.Series(index=w.index, dtype=str)),
                "Candidate Name": w.get("Candidate", pd.Series(index=w.index, dtype=str)),
                "Votes": w.get("Votes", pd.Series(index=w.index)),
                "Vote share (%)": w.get("Vote share (%)", pd.Series(index=w.index)),
                "Lead": w.get("Lead", pd.Series(index=w.index)),
                "Trailing Front": w.get("Trailing Front", pd.Series(index=w.index, dtype=str)).fillna("-"),
                "Trailing Party": w.get("Trailing Party", pd.Series(index=w.index, dtype=str)).fillna("-"),
                "Trailing Candidate": w.get("Trailing Candidate", pd.Series(index=w.index, dtype=str)).fillna("-"),
            }).sort_values("Ward name").reset_index(drop=True)

            styled_win = style_lead_table(winners_tbl, "Lead", positive=True).format({
                "Votes": "{:,.0f}",
                "Lead": "{:,.0f}",
                "Vote share (%)": "{:,.2f}%"
            })
            render_styled_table(styled_win)

    # ===== LOSING CANDIDATES (details) =====
    st.subheader(f"📉 Losing Candidates - {sel_front}")
    if lb_ward.empty or "Rank" not in lb_ward.columns:
        st.info("No Rank data available.")
    else:
        keys = _ward_join_keys(lb_ward)
        losers = lb_ward[(lb_ward["Front"] == sel_front) & (lb_ward["Rank"].astype("Int64") != 1)].copy()
        if losers.empty:
            st.info("No losing wards for the selected front here.")
        else:
            totals = (lb_ward.groupby(keys, as_index=False)["Votes"].sum().rename(columns={"Votes": "TotalVotes"}))
            winners_any = lb_ward[lb_ward["Rank"].astype("Int64") == 1][keys + ["Front", "Party", "Candidate", "Votes"]]
            winners_any = winners_any.rename(columns={
                "Front": "Winning Front", "Party": "Winning Party",
                "Candidate": "Winning Candidate", "Votes": "WinnerVotes"
            })
            L = losers.merge(totals, on=keys, how="left").merge(winners_any, on=keys, how="left")
            L["Vote share (%)"] = np.where(L["TotalVotes"] > 0, L["Votes"] / L["TotalVotes"] * 100, 0.0)
            L["Trail"] = L.get("WinnerVotes", 0) - L.get("Votes", 0)

            losers_tbl = pd.DataFrame({
                "Ward name": L.get("WardName", pd.Series(index=L.index, dtype=str)),
                "Candidate Name": L.get("Candidate", pd.Series(index=L.index, dtype=str)),
                "Votes": L.get("Votes", pd.Series(index=L.index)),
                "Vote share (%)": L.get("Vote share (%)", pd.Series(index=L.index)),
                "Trail": L.get("Trail", pd.Series(index=L.index)),
                "Winning Front": L.get("Winning Front", pd.Series(index=L.index, dtype=str)).fillna("-"),
                "Winning Party": L.get("Winning Party", pd.Series(index=L.index, dtype=str)).fillna("-"),
                "Winning Candidate": L.get("Winning Candidate", pd.Series(index=L.index, dtype=str)).fillna("-"),
            }).sort_values("Ward name").reset_index(drop=True)

            styled_lose = style_lead_table(losers_tbl, "Trail", positive=False).format({
                "Votes": "{:,.0f}",
                "Trail": "{:,.0f}",
                "Vote share (%)": "{:,.2f}%"
            })
            render_styled_table(styled_lose)
