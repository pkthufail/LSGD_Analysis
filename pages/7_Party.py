# pages/6_Party.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import html

from lib.data import load_data, get_data_path, data_controls, load_wards_2025, lb_ward_count_lookup
from lib.colors import FRONT_BG_COLORS, PARTY_BG_COLORS, DEFAULT_BG_COLOR, FRONT_COLORS
from lib.ui import render_color_legend

# ---- Safe Styler import (pandas-version friendly) ----
try:
    from pandas.io.formats.style import Styler
except Exception:
    from typing import Any as Styler  # typing fallback

# ---------------- Page Config ----------------
st.set_page_config(page_title="Party · LSGD Explorer", page_icon="🏴", layout="wide")
st.title("🏴 Party View")

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

wards_2025_df = load_wards_2025()
LB_WARD_COUNTS = lb_ward_count_lookup(df, wards_2025_df)

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

def color_rows_uniform_party(df_display: pd.DataFrame, party: str):
    def _all_rows_party(_row: pd.Series):
        color = PARTY_BG_COLORS.get(party, DEFAULT_BG_COLOR)
        return [f"background-color: {color}"] * len(_row)
    return df_display.style.apply(_all_rows_party, axis=1)

def build_party_lb_performance(scope_df: pd.DataFrame, sel_party: str) -> pd.DataFrame:
    """
    For the given scope (already filtered to District/Assembly if needed):
      - Use Ward-tier rows only
      - For each LBName: Party Votes, Share (% of total LB votes), Contested, Won
      - Returns sorted by Share (%) desc then Party Votes desc.
    """
    if scope_df.empty:
        return pd.DataFrame(columns=["LBName", "Party Votes", "Share (%)", "Contested", "Won"])

    d = scope_df.copy()
    if "TierNorm" in d.columns:
        d = d[d["TierNorm"] == "Ward"].copy()
    else:
        d = d[d["Tier"].astype(str).str.title() == "Ward"].copy()
    if d.empty:
        return pd.DataFrame(columns=["LBName", "Party Votes", "Share (%)", "Contested", "Won"])

    # Totals per LB
    total_lb = d.groupby("LBName", as_index=False)["Votes"].sum().rename(columns={"Votes": "Total Votes"})
    # Party aggregates
    dp = d[d["Party"] == sel_party].copy()
    party_votes = dp.groupby("LBName", as_index=False)["Votes"].sum().rename(columns={"Votes": "Party Votes"})
    contested = dp.groupby("LBName", as_index=False).size().rename(columns={"size": "Contested"})
    if "Rank" in dp.columns:
        won = (dp[pd.to_numeric(dp["Rank"], errors="coerce").astype("Int64") == 1]
               .groupby("LBName", as_index=False)
               .size().rename(columns={"size": "Won"}))
    else:
        won = pd.DataFrame({"LBName": [], "Won": []})

    out = pd.merge(total_lb, party_votes, on="LBName", how="left")
    out = pd.merge(out, contested, on="LBName", how="left")
    out = pd.merge(out, won, on="LBName", how="left")
    for c in ["Party Votes", "Contested", "Won"]:
        if c in out.columns:
            out[c] = out[c].fillna(0).astype(int)
        else:
            out[c] = 0
    out["Share (%)"] = np.where(out["Total Votes"] > 0, out["Party Votes"] / out["Total Votes"] * 100, 0.0)
    out = out.drop(columns=["Total Votes"], errors="ignore")
    out = out.sort_values(["Share (%)", "Party Votes", "LBName"], ascending=[False, False, True])
    return out.reset_index(drop=True)

def party_options_for_front(front: str) -> list[str]:
    pref = {"UDF": ["IUML", "INC"], "LDF": ["CPI(M)", "CPI"], "NDA": ["BJP"], "OTH": ["IND", "SDPI", "WPI"]}
    parties_data = (df[df["Front"] == front]["Party"].dropna().sort_values().unique().tolist()
                    if {"Front","Party"}.issubset(df.columns) else [])
    base = pref.get(front, [])
    seen, ordered = set(), []
    for p in base:
        if not parties_data or p in parties_data:
            if p not in seen:
                ordered.append(p); seen.add(p)
    for p in parties_data:
        if p not in seen:
            ordered.append(p); seen.add(p)
    return ordered or base or parties_data

def table_votes_share(scope_df: pd.DataFrame, party: str) -> pd.DataFrame:
    """Tier summary (Ward/Block/District): Party Votes, Total Votes, Share % within the scope_df."""
    tiers = TIERS_ORDER
    total_by = (scope_df[scope_df["TierNorm"].isin(tiers)]
                .groupby("TierNorm", as_index=False)["Votes"].sum()
                .rename(columns={"Votes": "Total Votes"}))
    party_by = (scope_df[(scope_df["TierNorm"].isin(tiers)) & (scope_df["Party"] == party)]
                .groupby("TierNorm", as_index=False)["Votes"].sum()
                .rename(columns={"Votes": "Party Votes"}))
    out = pd.merge(pd.DataFrame({"TierNorm": tiers}), total_by, on="TierNorm", how="left").fillna({"Total Votes": 0})
    out = pd.merge(out, party_by, on="TierNorm", how="left").fillna({"Party Votes": 0})
    out["Share (%)"] = np.where(out["Total Votes"] > 0, out["Party Votes"] / out["Total Votes"] * 100, 0.0)
    out["Tier"] = pd.Categorical(out["TierNorm"], categories=tiers, ordered=True)
    out = out.sort_values("Tier").drop(columns=["TierNorm"]).rename(columns={"Tier":"Tier"})
    return out[["Tier","Party Votes","Total Votes","Share (%)"]]

def table_lbtype_performance(scope_df: pd.DataFrame, party: str, include_block_district: bool = False) -> pd.DataFrame:
    """
    LBType × Ranks (by Tier):
      - If include_block_district=False: Ward-tier only (Grama, Municipality, Corporation).
      - If include_block_district=True: Ward + Block + District tiers, so rows include Block & District too.
    Returns columns: LBType | Won | 2 | 3 | ... | Contested | Strike Rate (%)
    """
    if "Rank" not in scope_df.columns or "TierNorm" not in scope_df.columns:
        return pd.DataFrame()

    tiers = ["Ward", "Block", "District"] if include_block_district else ["Ward"]
    d = scope_df[(scope_df["TierNorm"].isin(tiers)) & (scope_df["Party"] == party)].copy()
    if d.empty:
        return pd.DataFrame(columns=["LBType", "Won", "Contested", "Strike Rate (%)"])

    base_rows = ["Grama", "Municipality", "Corporation"]
    if include_block_district:
        base_rows += ["Block", "District"]

    d["LBType"] = pd.Categorical(d["LBType"], categories=base_rows, ordered=True)
    xt = pd.crosstab(d["LBType"], d["Rank"]).fillna(0).astype(int)
    xt = xt.reindex(base_rows, fill_value=0)

    # Rename numeric rank columns to labels
    def _rank_to_label(v):
        try:
            n = int(v)
        except Exception:
            return v
        if n == 1:
            return "Won"
        suf = "th"
        if not 11 <= (n % 100) <= 13:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    rank_numeric = sorted([int(c) for c in xt.columns if isinstance(c, (int, float, np.integer, np.floating))])
    xt = xt.rename(columns={c: _rank_to_label(c) for c in xt.columns})

    rank_label_cols = [_rank_to_label(n) for n in rank_numeric]
    present_cols = [c for c in rank_label_cols if c in xt.columns]
    xt["Contested"] = xt[present_cols].sum(axis=1) if present_cols else 0
    xt["Strike Rate (%)"] = np.where(xt["Contested"] > 0, xt.get("Won", 0) / xt["Contested"] * 100, 0.0)

    col_order = (["Won"] if "Won" in xt.columns else []) + [c for c in rank_label_cols if c != "Won" and c in xt.columns] + ["Contested", "Strike Rate (%)"]
    xt = xt[col_order].reset_index()
    return xt

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

def _build_strength_chart(scope_df: pd.DataFrame, sel_party: str, title_suffix: str = ""):
    # Use precomputed Strength if available; otherwise derive from Lead.
    d = scope_df.copy()
    d = d[(d["TierNorm"] == "Ward") & (d["Party"] == sel_party)]
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
        .groupby("Strength", as_index=False)
        .size().rename(columns={"size": "Wards"})
    )
    if strength_summary.empty:
        return None

    strength_summary["Strength"] = pd.Categorical(
        strength_summary["Strength"],
        categories=_STRENGTH_ORDER,
        ordered=True
    )
    strength_summary = strength_summary.sort_values("Strength")

    mirror_df = strength_summary.copy()
    mirror_df["Display_Wards"] = mirror_df.apply(
        lambda row: -row["Wards"] if str(row["Strength"]).startswith("-") else row["Wards"],
        axis=1
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

def _build_vote_bin_chart(scope_df: pd.DataFrame, sel_party: str, title_suffix: str = ""):
    d = scope_df.copy()
    d = d[(d["TierNorm"] == "Ward") & (d["Party"] == sel_party)]
    if d.empty:
        return None
    if "VoteBin" not in d.columns:
        return None

    d["Status"] = np.where(d.get("Rank", 0).astype("Int64") == 1, "Won", "Not won")
    agg = (d.groupby(["VoteBin", "Status"], as_index=False)
             .size().rename(columns={"size": "Wards"}))

    # set category order for X axis
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

# --------- Winner/Runner join keys + opponent table (shared) ---------
def _ward_join_keys(df_: pd.DataFrame) -> list[str]:
    """Best-effort ward identity for joining winners↔runners."""
    if "WardCode" in df_.columns:
        return ["WardCode"]
    if "WardNo" in df_.columns:
        cols = [c for c in ["District", "LBName", "WardNo"] if c in df_.columns]
        if cols:
            return cols
    cols = [c for c in ["District", "LBName", "WardName"] if c in df_.columns]
    return cols or ["WardName"]

def table_opponent_breakdown(scope_df: pd.DataFrame, sel_party: str) -> pd.DataFrame:
    """
    Returns a table with:
      Party | Runner-up (when Selected Won) | Winners (when Selected Second)
    Only Ward-tier rows in the given scope are used.
    """
    if "Rank" not in scope_df.columns:
        return pd.DataFrame(columns=["Party", "Runner-up (when Selected Won)", "Winners (when Selected Second)"])

    d = scope_df.copy()
    d["TierNorm"] = d["Tier"].astype(str).str.title() if "TierNorm" not in d.columns else d["TierNorm"]
    d["Rank"] = pd.to_numeric(d["Rank"], errors="coerce").astype("Int64")
    d = d[d["TierNorm"] == "Ward"]
    if d.empty:
        return pd.DataFrame(columns=["Party", "Runner-up (when Selected Won)", "Winners (when Selected Second)"])

    keys = _ward_join_keys(d)
    winners = d[d["Rank"] == 1][keys + ["Party"]].rename(columns={"Party": "WinnerParty"})
    runners = d[d["Rank"] == 2][keys + ["Party"]].rename(columns={"Party": "RunnerParty"})

    # Ensure consistent dtype for merge keys
    if "WinnerParty" in winners.columns:
        winners["WinnerParty"] = winners["WinnerParty"].astype(str)
    if "RunnerParty" in runners.columns:
        runners["RunnerParty"] = runners["RunnerParty"].astype(str)

    # (A) Where selected party WON → count Runner-up parties
    wins_sel = winners[winners["WinnerParty"] == sel_party]
    ru_vs_selwin = (wins_sel.merge(runners, on=keys, how="left")
                    .groupby("RunnerParty", dropna=True).size().rename("Runner-up (when Selected Won)"))
    # Normalize index as string for safe merges later
    ru_vs_selwin.index = ru_vs_selwin.index.fillna("UNKNOWN").astype(str)

    # (B) Where selected party was SECOND → count Winner parties
    sec_sel = runners[runners["RunnerParty"] == sel_party]
    win_vs_selsec = (sec_sel.merge(winners, on=keys, how="left")
                      .groupby("WinnerParty", dropna=True).size().rename("Winners (when Selected Second)"))
    win_vs_selsec.index = win_vs_selsec.index.fillna("UNKNOWN").astype(str)

    # Combine
    all_parties = sorted(set(ru_vs_selwin.index.tolist()) | set(win_vs_selsec.index.tolist()), key=lambda x: str(x))
    out = pd.DataFrame({"Party": all_parties})
    out["Party"] = out["Party"].astype(str)
    out = out.merge(ru_vs_selwin.reset_index().rename(columns={"RunnerParty": "Party"}).assign(Party=lambda df_: df_["Party"].astype(str)), on="Party", how="left")
    out = out.merge(win_vs_selsec.reset_index().rename(columns={"WinnerParty": "Party"}).assign(Party=lambda df_: df_["Party"].astype(str)), on="Party", how="left")
    out[["Runner-up (when Selected Won)", "Winners (when Selected Second)"]] = \
        out[["Runner-up (when Selected Won)", "Winners (when Selected Second)"]].fillna(0).astype(int)

    # Sort by overall impact then party
    out["Total"] = out["Runner-up (when Selected Won)"] + out["Winners (when Selected Second)"]
    out = out.sort_values(["Total", "Party"], ascending=[False, True]).drop(columns=["Total"]).reset_index(drop=True)
    return out


# ---------------- Filters (Front / Party) ----------------
st.markdown("### Filters")
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    sel_front = st.selectbox("Front", FRONT_ORDER, index=0)  # default: UDF
with c2:
    party_choices = party_options_for_front(sel_front)
    default_ix = next((i for i, p in enumerate(party_choices) if p.upper().strip() == "IUML"), 0)
    sel_party = st.selectbox("Party", party_choices, index=min(default_ix, max(len(party_choices)-1, 0)))

st.markdown("---")

# ---------------- Tabs ----------------
tab_d, tab_a, tab_l = st.tabs(["🏙️ District", "🏛️ Assembly", "🏘️ Local Body"])

# ---------- District Tab ----------
with tab_d:
    st.markdown("#### Scope")
    districts = ["All Kerala"] + sorted(df["District"].dropna().unique().tolist(), key=lambda x: str(x))
    sel_district = st.selectbox("District", districts, index=0, key="party_tab_district")  # default All Kerala

    if sel_district == "All Kerala":
        scoped = df.copy()
        scope_label = "**All Kerala**"
    else:
        scoped = df[df["District"] == sel_district].copy()
        scope_label = f"**{sel_district}**"

    # TABLE 1
    st.subheader(f"🧮 {sel_party} — Votes & Vote Share by Tier ({scope_label})")
    t_votes = table_votes_share(scoped, sel_party)
    styled_votes = color_rows_uniform_party(t_votes, sel_party)
    render_styled_table(styled_votes, fmt_numbers=["Party Votes","Total Votes"], fmt_perc=["Share (%)"])

    # TABLE 2 (Ward + Block + District)
    st.subheader(f"🏆 {sel_party} — Seats & Ranks by LBType (Ward/Block/District) ({scope_label})")
    t_perf = table_lbtype_performance(scoped, sel_party, include_block_district=True)
    if t_perf.empty:
        st.info("No Rank data in this scope.")
    else:
        fmt_nums = [c for c in t_perf.columns if c not in ["LBType", "Strike Rate (%)"]]
        styled_perf = (
            t_perf.style
            .apply(lambda r: [f"background-color: {PARTY_BG_COLORS.get(sel_party, DEFAULT_BG_COLOR)}"] * len(r), axis=1)
            .format({**{c: "{:,.0f}" for c in fmt_nums}, "Strike Rate (%)": _fmt_sr})
        )
        render_styled_table(styled_perf)

    # OPPONENT BREAKDOWN (District scope) — TABLE
    st.subheader(f"🤝 Opponent Breakdown — {sel_party} ({scope_label})")
    t_opp = table_opponent_breakdown(scoped, sel_party)
    if t_opp.empty:
        st.info("No Ward-tier winner/runner data available for this scope.")
    else:
        def _row_party_color(row: pd.Series):
            party_key = {"CPM": "CPI(M)"}.get(str(row.get("Party","")).strip(), str(row.get("Party","")).strip())
            color = PARTY_BG_COLORS.get(party_key, DEFAULT_BG_COLOR)
            return [f"background-color: {color}"] * len(row)

        styled_opp = (
            t_opp.style
            .apply(_row_party_color, axis=1)
            .format({
                "Runner-up (when Selected Won)": "{:,.0f}",
                "Winners (when Selected Second)": "{:,.0f}"
            })
        )
        render_styled_table(styled_opp)

    # STRENGTH (mirror) CHART
    st.subheader(f"📶 {sel_party} — Number of Strong and Weak Wards ({scope_label})")
    fig_strength = _build_strength_chart(scoped, sel_party)
    if fig_strength is None:
        st.info("No Strength/Lead data available for this scope.")
    else:
        st.plotly_chart(fig_strength, use_container_width=True)

    # VOTEBIN STACKED BAR (Won vs Not won)
    st.subheader(f"📊 {sel_party} — VoteBin vs Wards (Won + Not won) ({scope_label})")
    fig_vote = _build_vote_bin_chart(scoped, sel_party)
    if fig_vote is None:
        st.info("No VoteBin data available for this scope.")
    else:
        st.plotly_chart(fig_vote, use_container_width=True)

    # Local Body performance (Ward-tier) table should be last in Assembly tab
    st.subheader(f"Local Body performance (Ward-tier) — {sel_party} ({scope_label})")
    t_lb_perf_a = build_party_lb_performance(scoped, sel_party)
    if t_lb_perf_a.empty:
        st.info("No Ward-tier rows available to compute LB performance.")
    else:
        styled_lb_a = color_rows_uniform_party(t_lb_perf_a, sel_party)
        render_styled_table(styled_lb_a, fmt_numbers=["Party Votes", "Contested", "Won"], fmt_perc=["Share (%)"])

    # Local Body performance table should appear last; hide for All Kerala
    if sel_district != "All Kerala":
        st.subheader(f"Local Body performance (Ward-tier) — {sel_party} ({scope_label})")
        t_lb_perf_d = build_party_lb_performance(scoped, sel_party)
        if t_lb_perf_d.empty:
            st.info("No Ward-tier rows available to compute LB performance.")
        else:
            styled_lb_d = color_rows_uniform_party(t_lb_perf_d, sel_party)
            render_styled_table(styled_lb_d, fmt_numbers=["Party Votes", "Contested", "Won"], fmt_perc=["Share (%)"])

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
            sel_district_a = st.selectbox("District", districts, index=default_dix, key="party_tab_a_district")
        df_d = df[df["District"] == sel_district_a]
        assemblies = sorted(df_d[asm_col].dropna().unique().tolist(), key=lambda x: str(x))
        default_ax = next((i for i, a in enumerate(assemblies) if "malappuram" in str(a).strip().lower()), 0)
        with c2:
            sel_assembly = st.selectbox("Assembly", assemblies, index=(default_ax if assemblies else 0), key="party_tab_a_assembly")

        scoped = df[(df["District"] == sel_district_a) & (df[asm_col] == sel_assembly)].copy()
        scope_label = f"**{sel_assembly}**"

        # =====================================================
        # TABLE 1: Party votes & share in this Assembly (Tier = Ward)
        # =====================================================
        st.subheader(f"🧮 {sel_party} — Votes & Share in Assembly (Ward-tier) ({scope_label})")

        asm_ward = scoped[scoped["TierNorm"] == "Ward"].copy()
        total_votes = int(asm_ward["Votes"].sum()) if "Votes" in asm_ward.columns else 0
        party_votes = int(asm_ward.loc[asm_ward["Party"] == sel_party, "Votes"].sum()) if "Votes" in asm_ward.columns else 0
        share = (party_votes / total_votes * 100) if total_votes > 0 else 0.0

        t_votes_asm = pd.DataFrame({
            "Party Votes": [party_votes],
            "Total Votes": [total_votes],
            "Share (%)":   [share],
        })

        styled_votes_asm = color_rows_uniform_party(t_votes_asm, sel_party)
        render_styled_table(styled_votes_asm, fmt_numbers=["Party Votes", "Total Votes"], fmt_perc=["Share (%)"])
        render_color_legend({sel_party: PARTY_BG_COLORS.get(sel_party, DEFAULT_BG_COLOR)}, title="Row color")

        # =====================================================
        # TABLE 2: LBName × Ranks (Won, 2, 3, ...), Contested, Strike Rate — with TOTAL row
        # =====================================================
        st.subheader(f"🏆 {sel_party} — Performance by Local Body (Ward-tier) ({scope_label})")

        df_p = asm_ward[asm_ward["Party"] == sel_party].copy()
        if df_p.empty or "Rank" not in df_p.columns or "LBName" not in df_p.columns:
            st.info("No Ward-tier Rank data for this party in the selected assembly.")
        else:
            df_p["Rank"] = pd.to_numeric(df_p["Rank"], errors="coerce").astype("Int64")

            # Crosstab LBName × Rank
            ct = pd.crosstab(df_p["LBName"], df_p["Rank"]).fillna(0).astype(int)

            # Determine all numeric rank columns present
            # Rename numeric rank columns to human-friendly labels: 1->Won, 2->2nd, 3->3rd, ...
            def _rank_to_label(v):
                try:
                    n = int(v)
                except Exception:
                    return v
                if n == 1:
                    return "Won"
                suf = "th"
                if not 11 <= (n % 100) <= 13:
                    suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
                return f"{n}{suf}"
            all_ranks = sorted([int(c) for c in ct.columns if isinstance(c, (int, float, np.integer, np.floating))])
            ct = ct.rename(columns={c: _rank_to_label(c) for c in ct.columns})

            # Rename Rank 1 → Won
            # already renamed above

            # Build ordered columns: Won, 2..max, then Contested, Strike Rate
            other_ranks = [_rank_to_label(r) for r in all_ranks if r != 1 and _rank_to_label(r) in ct.columns]
            ordered_cols = (["Won"] if "Won" in ct.columns else []) + other_ranks

            # Per-LB totals & strike rate
            ct["Contested"] = ct[ordered_cols].sum(axis=1) if ordered_cols else 0
            if "Won" in ct.columns:
                ct["Strike Rate (%)"] = np.where(ct["Contested"] > 0, ct["Won"] / ct["Contested"] * 100, 0.0)
            else:
                ct["Strike Rate (%)"] = 0.0

            # Reorder and reset index
            t2 = ct[ordered_cols + ["Contested", "Strike Rate (%)"]].reset_index()

            # TOTAL row
            total_vals = {col: (t2[col].sum() if col not in ["LBName", "Strike Rate (%)"] else None) for col in t2.columns}
            total_won = total_vals.get("Won", 0) if total_vals.get("Won", 0) is not None else 0
            total_cont = total_vals.get("Contested", 0) if total_vals.get("Contested", 0) is not None else 0
            total_sr = (total_won / total_cont * 100) if total_cont > 0 else 0.0

            total_row = {col: 0 for col in t2.columns if col not in ["LBName", "Strike Rate (%)"]}
            for col in total_row.keys():
                total_row[col] = int(t2[col].sum())
            total_row["LBName"] = "Total"
            total_row["Strike Rate (%)"] = total_sr

            t2 = pd.concat([t2.sort_values("LBName"), pd.DataFrame([total_row])], ignore_index=True)

            # Style & render
            fmt_nums = [c for c in t2.columns if c not in ["LBName", "Strike Rate (%)"]]
            styled_t2 = (
                t2.style
                .apply(lambda r: [f"background-color: {PARTY_BG_COLORS.get(sel_party, DEFAULT_BG_COLOR)}"] * len(r), axis=1)
                .format({**{c: "{:,.0f}" for c in fmt_nums}, "Strike Rate (%)": _fmt_sr})
            )
            render_styled_table(styled_t2)
            render_color_legend({sel_party: PARTY_BG_COLORS.get(sel_party, DEFAULT_BG_COLOR)}, title="Row color")

        # ---------- OPPONENT BREAKDOWN — TABLE (no chart) ----------
        st.subheader(f"🤝 {sel_party} — Opponent Breakdown ({scope_label})")
        t_opp_a = table_opponent_breakdown(scoped, sel_party)
        if t_opp_a.empty:
            st.info("No Ward-tier winner/runner data available for this scope.")
        else:
            def _row_party_color_a(row: pd.Series):
                party_key = {"CPM": "CPI(M)"}.get(str(row.get("Party","")).strip(), str(row.get("Party","")).strip())
                color = PARTY_BG_COLORS.get(party_key, DEFAULT_BG_COLOR)
                return [f"background-color: {color}"] * len(row)

            styled_opp_a = (
                t_opp_a.style
                .apply(_row_party_color_a, axis=1)
                .format({
                    "Runner-up (when Selected Won)": "{:,.0f}",
                    "Winners (when Selected Second)": "{:,.0f}"
                })
            )
            render_styled_table(styled_opp_a)

        # ---------- STRENGTH (mirror) CHART ----------
        st.subheader(f"📶 {sel_party} — Number of Strong and Weak Wards ({scope_label})")
        fig_strength_a = _build_strength_chart(scoped, sel_party)
        if fig_strength_a is None:
            st.info("No Strength/Lead data available for this scope.")
        else:
            st.plotly_chart(fig_strength_a, use_container_width=True)

        # ---------- VOTEBIN STACKED BAR ----------
        st.subheader(f"📊 {sel_party} — VoteBin vs Wards (Won + Not won) ({scope_label})")
        fig_vote_a = _build_vote_bin_chart(scoped, sel_party)
        if fig_vote_a is None:
            st.info("No VoteBin data available for this scope.")
        else:
            st.plotly_chart(fig_vote_a, use_container_width=True)

        if asm_ward.empty or "Votes" not in asm_ward.columns:
            st.info("No ward-level vote data available for this assembly.")
        else:
            party_contested = asm_ward[asm_ward["Party"] == sel_party].copy()
            if party_contested.empty:
                st.info(f"{sel_party} did not contest any wards in this assembly.")
            else:
                key_cols = _ward_join_keys(asm_ward) or ["WardName"]
                total_by_ward = asm_ward.groupby(key_cols)["Votes"].sum().rename("Total Votes")
                party_by_ward = party_contested.groupby(key_cols)["Votes"].sum().rename("Party Votes")
                ward_summary = party_by_ward.to_frame().join(total_by_ward, how="left")
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
                    summary["Party Votes"] = summary["Party Votes"].astype(int)
                    summary["Total Votes"] = summary["Total Votes"].astype(int)
                    summary["Share (%)"] = np.where(summary["Total Votes"] > 0, summary["Party Votes"] / summary["Total Votes"] * 100, 0.0)
                    column_order = []
                    for col in ["LBName", "WardName", "WardNo", "WardCode"]:
                        if col in summary.columns:
                            column_order.append(col)
                    column_order += ["Party Votes", "Total Votes", "Share (%)"]
                    column_order = list(dict.fromkeys(column_order))
                    strongest = summary[summary["Share (%)"] > 50].sort_values(["Share (%)", "Party Votes"], ascending=[False, False]).head(20)
                    weakest = summary[summary["Share (%)"] < 50].sort_values(["Share (%)", "Party Votes"], ascending=[True, False]).head(20)

                    def _render_ward_list(frame: pd.DataFrame, title: str, empty_message: str, stronger: bool) -> None:
                        st.subheader(title)
                        if frame.empty:
                            st.info(empty_message)
                            return
                        table = frame[column_order].copy()
                        for col in ["Party Votes", "Total Votes"]:
                            if col in table.columns:
                                table[col] = table[col].astype(int)
                        fmt_nums = [c for c in ["Party Votes", "Total Votes"] if c in table.columns]
                        if "Share (%)" in table.columns:
                            styled_table = style_share_margin_table(table, "Share (%)", stronger=stronger)
                        else:
                            styled_table = table.style
                        render_styled_table(styled_table, fmt_numbers=fmt_nums, fmt_perc=["Share (%)"])

                    _render_ward_list(
                        strongest,
                        f"{sel_party} - Strongest Wards ({scope_label})",
                        f"No wards where {sel_party} crossed 50% vote share in this assembly.",
                        True,
                    )
                    _render_ward_list(
                        weakest,
                        f"{sel_party} - Weakest Wards ({scope_label})",
                        f"No wards where {sel_party} fell below 50% vote share in this assembly.",
                        False,
                    )


# ---------- Local Body Tab ----------
with tab_l:
    st.markdown("#### Scope")
    districts = sorted(df["District"].dropna().unique().tolist(), key=lambda x: str(x))
    default_dix = next((i for i, d in enumerate(districts) if str(d).strip().lower() == "malappuram"), 0)
    c1, c2 = st.columns([1, 1])
    with c1:
        sel_district_l = st.selectbox("District", districts, index=default_dix, key="party_tab_l_district")
    df_d = df[df["District"] == sel_district_l]
    lbnames = sorted(df_d["LBName"].dropna().unique().tolist(), key=lambda x: str(x))
    with c2:
        sel_lb = st.selectbox("Local Body", lbnames, index=0 if lbnames else 0, key="party_tab_l_lb")

    scoped = df[(df["District"] == sel_district_l) & (df["LBName"] == sel_lb)].copy()
    scope_label = f"**{sel_lb}**"

    # Restrict to Ward-tier for most computations
    lb_ward = scoped[scoped["TierNorm"] == "Ward"].copy()
    lb_party = lb_ward[lb_ward["Party"] == sel_party].copy()

    # =============== SUMMARY ===============
    st.subheader(f"🧮 Summary — {sel_party} in {scope_label}")
    lb_code_series = lb_ward.get("LBCode", pd.Series(dtype=str))
    if lb_code_series is not None and not lb_code_series.dropna().empty:
        lb_code_val = lb_code_series.dropna().astype(str).unique()[0]
        counts_info = LB_WARD_COUNTS.get(lb_code_val)
        if counts_info:
            st.markdown(f"**Ward counts:** 2020 - {counts_info['wards_2020']:,} | 2025 - {counts_info['wards_2025']:,} | New wards - {counts_info['new_wards']:,}")
    if lb_ward.empty:
        st.info("No Ward-tier data for this local body.")
    else:
        total_votes = int(lb_ward["Votes"].sum()) if "Votes" in lb_ward.columns else 0
        party_votes = int(lb_party["Votes"].sum()) if "Votes" in lb_party.columns else 0
        share = (party_votes / total_votes * 100) if total_votes > 0 else 0.0
        contested = len(lb_party)
        won = int((lb_party.get("Rank", pd.Series(dtype="Int64")).astype("Int64") == 1).sum()) if "Rank" in lb_party.columns else 0

        party_hex = PARTY_BG_COLORS.get(sel_party, "#e9ecef")
        st.markdown(
            f"""
            <div>
              <span style="font-weight:700;color:{party_hex}">{sel_party}</span>
              secured <span style="font-weight:700">{party_votes:,}</span> votes
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



    # =============== STRENGTH ANALYSIS (Lead / Trail) ===============
    st.subheader("📶 Strength Analysis")

    if lb_party.empty:
        st.info("No rows for the selected party in this local body.")
    else:
        if "Strength" in lb_party.columns and lb_party["Strength"].notna().any():
            s_series = lb_party["Strength"].astype(str)
        elif "Lead" in lb_party.columns:
            s_series = lb_party["Lead"].apply(_lead_to_strength)
        else:
            s_series = pd.Series(dtype="object")

        if s_series.dropna().empty or "WardName" not in lb_party.columns:
            st.info("No Strength/Lead or WardName data available.")
        else:
            s_df = pd.DataFrame({"Strength": s_series, "WardName": lb_party["WardName"]}).dropna(subset=["Strength"])
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
            lead_tbl = agg[~is_trail & ~is_zero & (agg["Wards"] >= 1)]
            trail_tbl = agg[is_trail & (agg["Wards"] >= 1)]

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

    # =============== VOTEBIN LIST (with colored ward names) ===============
    st.subheader("🧊 VoteBin Summary (Won/Not won names colour-coded)")
    if lb_party.empty or "VoteBin" not in lb_party.columns or "WardName" not in lb_party.columns:
        st.info("VoteBin or WardName not available for the selected party.")
    else:
        tmp = lb_party.copy()
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
            st.info("No VoteBin data available for this party.")
        else:
            table = grp.copy()
            table["Count"] = table["data"].apply(lambda d: int(d["count"]))
            table["Won Names"] = table["data"].apply(lambda d: format_vote_names(d["won"], WON_NAME_COLOR))
            table["Not Won Names"] = table["data"].apply(lambda d: format_vote_names(d["not"], NOT_WON_NAME_COLOR))
            table = table.rename(columns={"VoteBinStr": "VoteBin"})
            table = table[["VoteBin", "Count", "Won Names", "Not Won Names"]]
            styled = color_rows_uniform_party(table, sel_party)
            styled = styled.format({"Won Names": lambda x: x, "Not Won Names": lambda x: x}, escape=False)
            render_styled_table(styled, fmt_numbers=["Count"])
    # =============== OPPONENT BREAKDOWN (same table format) ===============
    st.subheader(f"🤝 Opponent Breakdown — {sel_party} ({scope_label})")
    t_opp_l = table_opponent_breakdown(lb_ward, sel_party)
    if t_opp_l.empty:
        st.info("No Ward-tier winner/runner data available for this local body.")
    else:
        def _row_party_color_l(row: pd.Series):
            party_key = {"CPM": "CPI(M)"}.get(str(row.get("Party","")).strip(), str(row.get("Party","")).strip())
            color = PARTY_BG_COLORS.get(party_key, DEFAULT_BG_COLOR)
            return [f"background-color: {color}"] * len(row)

        styled_opp_l = (
            t_opp_l.style
            .apply(_row_party_color_l, axis=1)
            .format({
                "Runner-up (when Selected Won)": "{:,.0f}",
                "Winners (when Selected Second)": "{:,.0f}"
            })
        )
        render_styled_table(styled_opp_l)

    # =============== WINNING CANDIDATES (details) ===============
    st.subheader(f"🏅 Winning Candidates - {sel_party}")
    if lb_ward.empty or "Rank" not in lb_ward.columns:
        st.info("No Rank data available.")
    else:
        keys = _ward_join_keys(lb_ward)
        winners = lb_ward[(lb_ward["Party"] == sel_party) & (lb_ward["Rank"].astype("Int64") == 1)].copy()
        if winners.empty:
            st.info("No winning wards for the selected party here.")
        else:
            # Ward totals for vote share
            totals = (lb_ward.groupby(keys, as_index=False)["Votes"].sum()
                              .rename(columns={"Votes": "TotalVotes"}))
            # Runner-up join
            runners = lb_ward[lb_ward["Rank"].astype("Int64") == 2][keys + ["Party", "Candidate", "Votes"]]
            runners = runners.rename(columns={"Party": "Trailing Party", "Candidate": "Trailing Candidate", "Votes": "RunnerVotes"})

            w = winners.merge(totals, on=keys, how="left").merge(runners, on=keys, how="left")
            w["Vote share (%)"] = np.where(w["TotalVotes"] > 0, w["Votes"] / w["TotalVotes"] * 100, 0.0)
            # Lead: use provided or compute
            if "Lead" in w.columns:
                # If missing, compute from votes
                w["Lead"] = w["Lead"].fillna(w["Votes"] - w.get("RunnerVotes", 0))
            else:
                w["Lead"] = w["Votes"] - w.get("RunnerVotes", 0)

            winners_tbl = pd.DataFrame({
                "Ward name": w.get("WardName", pd.Series(index=w.index, dtype=str)),
                "Candidate Name": w.get("Candidate", pd.Series(index=w.index, dtype=str)),
                "Votes": w.get("Votes", pd.Series(index=w.index)),
                "Vote share (%)": w.get("Vote share (%)", pd.Series(index=w.index)),
                "Lead": w.get("Lead", pd.Series(index=w.index)),
                "Trailing Party": w.get("Trailing Party", pd.Series(index=w.index, dtype=str)).fillna("-"),
                "Trailing Candidate": w.get("Trailing Candidate", pd.Series(index=w.index, dtype=str)).fillna("-"),
            }).sort_values("Ward name").reset_index(drop=True)

            styled_win = style_lead_table(winners_tbl, "Lead", positive=True).format({
                "Votes": "{:,.0f}",
                "Lead": "{:,.0f}",
                "Vote share (%)": "{:,.2f}%"
            })
            render_styled_table(styled_win)

    # =============== LOSING CANDIDATES (details) ===============
    st.subheader(f"📉 Losing Candidates - {sel_party}")
    if lb_ward.empty or "Rank" not in lb_ward.columns:
        st.info("No Rank data available.")
    else:
        keys = _ward_join_keys(lb_ward)
        losers = lb_ward[(lb_ward["Party"] == sel_party) & (lb_ward["Rank"].astype("Int64") != 1)].copy()
        if losers.empty:
            st.info("No losing wards for the selected party here.")
        else:
            totals = (lb_ward.groupby(keys, as_index=False)["Votes"].sum()
                              .rename(columns={"Votes": "TotalVotes"}))
            winners_any = lb_ward[lb_ward["Rank"].astype("Int64") == 1][keys + ["Party", "Candidate", "Votes"]]
            winners_any = winners_any.rename(columns={"Party": "Winning Party", "Candidate": "Winning Candidate", "Votes": "WinnerVotes"})

            L = losers.merge(totals, on=keys, how="left").merge(winners_any, on=keys, how="left")
            L["Vote share (%)"] = np.where(L["TotalVotes"] > 0, L["Votes"] / L["TotalVotes"] * 100, 0.0)
            L["Trail"] = L.get("WinnerVotes", 0) - L.get("Votes", 0)

            losers_tbl = pd.DataFrame({
                "Ward name": L.get("WardName", pd.Series(index=L.index, dtype=str)),
                "Candidate Name": L.get("Candidate", pd.Series(index=L.index, dtype=str)),
                "Votes": L.get("Votes", pd.Series(index=L.index)),
                "Vote share (%)": L.get("Vote share (%)", pd.Series(index=L.index)),
                "Trail": L.get("Trail", pd.Series(index=L.index)),
                "Winning Party": L.get("Winning Party", pd.Series(index=L.index, dtype=str)).fillna("-"),
                "Winning Candidate": L.get("Winning Candidate", pd.Series(index=L.index, dtype=str)).fillna("-"),
            }).sort_values("Ward name").reset_index(drop=True)

            styled_lose = style_lead_table(losers_tbl, "Trail", positive=False).format({
                "Votes": "{:,.0f}",
                "Trail": "{:,.0f}",
                "Vote share (%)": "{:,.2f}%"
            })
            render_styled_table(styled_lose)

