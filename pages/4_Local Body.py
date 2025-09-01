# pages/4_Local_Body.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from lib.data import load_data, get_data_path, data_controls
from lib.colors import FRONT_BG_COLORS, DEFAULT_BG_COLOR, FRONT_COLORS, PARTY_BG_COLORS

# ---------------- Page Config ----------------
st.set_page_config(page_title="Local Body · LSGD Explorer", page_icon="🏘️", layout="wide")
st.title("🏘️ Local Body (Panchayath / Municipality / Corporation)")

# ---------------- Sidebar: Data controls ----------------
data_controls()
df = load_data(get_data_path()).copy()

# ---------------- Hygiene ----------------
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].str.strip()
if "Votes" in df.columns:
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# Scope to Ward tier only
df = df[df["Tier"].astype(str).str.title() == "Ward"].copy()

# ---------------- Helpers ----------------
def _apply_number_formats(styler: pd.io.formats.style.Styler, df_display: pd.DataFrame, percent_cols: list[str]):
    fmt_map = {}
    for col in df_display.select_dtypes(include=["number"]).columns.tolist():
        fmt_map[col] = "{:,.2f}%" if col in percent_cols else "{:,.0f}"
    return styler.format(fmt_map)

def style_rows_by_palette(df_display: pd.DataFrame, key_col: str, palette: dict, default_color: str = DEFAULT_BG_COLOR):
    def _row_style(row: pd.Series):
        color = palette.get(row.get(key_col, None), "")
        return [f"background-color: {color}"] * len(row) if color else [""] * len(row)
    styler = df_display.style.apply(_row_style, axis=1)
    # (Hide index here too, but render_styled_table will also enforce hiding)
    try:
        styler = styler.hide(axis="index")
    except Exception:
        styler = styler.hide_index()
    return styler

def render_styled_table(styler: pd.io.formats.style.Styler):
    # ALWAYS hide the index before rendering to HTML (fixes stray 0..N index)
    try:
        styler = styler.hide(axis="index")
    except Exception:
        styler = styler.hide_index()
    html = styler.to_html()
    st.markdown(
        """
        <style>
          .tbl-wrap { width: 100%; overflow-x: auto; }
          .tbl-wrap table { width: 100%; border-collapse: collapse; table-layout: auto; }
          .tbl-wrap th, .tbl-wrap td { padding: 6px 8px; }
          @media (max-width: 1200px) { .tbl-wrap th, .tbl-wrap td { font-size: 0.9rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='tbl-wrap'>{html}</div>", unsafe_allow_html=True)

# ---------------- Filters (on page) ----------------
st.markdown("### Filters")
districts = sorted(df["District"].dropna().unique().tolist(), key=lambda x: str(x))
default_dix = next((i for i, d in enumerate(districts) if str(d).lower() == "malappuram"), 0)
sel_district = st.selectbox("District", districts, index=default_dix, key="lb_district")

df_d = df[df["District"] == sel_district].copy()
lbnames = sorted(df_d["LBName"].dropna().unique().tolist(), key=lambda x: str(x))
default_lbx = next((i for i, n in enumerate(lbnames) if "malappuram" in str(n).lower()), 0) if lbnames else 0
sel_lb = st.selectbox("Local Body", lbnames, index=(default_lbx if lbnames else 0), key="lb_name")

if df_d.empty or not lbnames:
    st.info("No Ward-tier rows found for this district/local body.")
    st.stop()

df_lb_all = df_d[df_d["LBName"] == sel_lb].copy()

# Winners (Rank==1 if available)
df_winners = df_lb_all[df_lb_all["Rank"] == 1].copy() if "Rank" in df_lb_all.columns else df_lb_all.copy()

# ===== Summary lines (colored) =====
st.markdown("### Summary")
FRONT_ORDER = ["UDF", "LDF", "NDA", "OTH"]

party_specific = {
    "INC":  "#1F77B4", "IUML": "#2E8B57", "CPI(M)": "#D62728", "CPM": "#D62728",
    "CPI":  "#EF5350", "BJP":  "#FF7F0E", "IND": "#7F7F7F", "SDPI": "#2CA02C", "WPI":  "#17BECF",
}
front_palettes = {
    "UDF": ["#0B5394", "#6FA8DC", "#9FC5E8", "#A7C7E7"],
    "LDF": ["#C62828", "#E57373", "#F4A6A6", "#FF8A80"],
    "NDA": ["#FF8C00", "#F7B731", "#FFD199", "#CC7722"],
    "OTH": ["#757575", "#9E9E9E", "#BDBDBD", "#E0E0E0"],
}

# Build party colors present in winners
final_party_color = {}
for fr in FRONT_ORDER:
    parties_fr = df_winners.loc[df_winners["Front"] == fr, "Party"].dropna().unique().tolist()
    palette = front_palettes.get(fr, ["#666666"])
    k = 0
    for p in parties_fr:
        if p in final_party_color:
            continue
        final_party_color[p] = party_specific.get(p, palette[k % len(palette)])
        if p not in party_specific:
            k += 1

# One colored line per front
for fr in FRONT_ORDER:
    df_f = df_winners[df_winners["Front"] == fr]
    total = len(df_f)
    if total == 0:
        continue
    party_counts = df_f.groupby("Party").size().sort_values(ascending=False).to_dict()
    front_span = f"<span style='color:{FRONT_COLORS.get(fr, '#333')}; font-weight:800'>{fr} ({total:,})</span>"
    parts = []
    for p, cnt in party_counts.items():
        pcol = final_party_color.get(p, "#444")
        parts.append(f"<span style='color:{pcol}; font-weight:600'>{p}</span> <span style='opacity:.85'>({cnt:,})</span>")
    st.markdown(front_span + ": " + ", ".join(parts), unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# 1) Seats Won by Front (winners only)
# =====================================================
st.subheader(f"🏆 Seats Won by Front — {sel_lb}, {sel_district}")
front_summary = (
    df_winners.groupby("Front", as_index=False)
    .size().rename(columns={"size": "Seats Won"})
    .sort_values("Seats Won", ascending=False)
)
front_summary = pd.DataFrame({"Front": FRONT_ORDER}).merge(front_summary, on="Front", how="left").fillna({"Seats Won": 0})
styled_front = style_rows_by_palette(front_summary, key_col="Front", palette=FRONT_BG_COLORS)
styled_front = _apply_number_formats(styled_front, front_summary, percent_cols=[])
render_styled_table(styled_front)

# =====================================================
# 2) Visualization — Winning Party Composition by Front
# =====================================================
st.subheader(f"📊 Winning Party Composition by Front — {sel_lb}")

if df_winners.empty:
    st.info("No winners available to visualize.")
else:
    df_agg = (
        df_winners.groupby(["Front", "Party"], as_index=False)
        .size().rename(columns={"size": "Seats Won"})
    )
    base = alt.Chart(df_agg).mark_bar().encode(
        y=alt.Y("Front:N", sort=FRONT_ORDER, title="Front"),
        x=alt.X("Seats Won:Q", stack="zero", title="Winning Seats"),
        color=alt.Color(
            "Party:N",
            scale=alt.Scale(domain=list(final_party_color.keys()), range=list(final_party_color.values())),
            legend=alt.Legend(title="Party", orient="bottom", columns=4, labelLimit=220),
        ),
        tooltip=[
            alt.Tooltip("Front:N", title="Front"),
            alt.Tooltip("Party:N", title="Party"),
            alt.Tooltip("Seats Won:Q", title="Seats Won", format="d"),
        ],
    )
    labels = base.mark_text(color="white", dx=-8, align="center").encode(
        text=alt.Text("Seats Won:Q", format=".0f")
    )
    st.altair_chart((base + labels).properties(height=220), use_container_width=True)

# =====================================================
# 2B) Party Performance in this Local Body (by Rank)
#      Columns: Party | Won | 2 | 3 | ... | Contested
# =====================================================
st.subheader(f"🏅 Party Performance — {sel_lb}, {sel_district} (by Rank)")

df_party = df_lb_all.copy()

if "Party" not in df_party.columns or "Rank" not in df_party.columns:
    st.info("`Party` or `Rank` column not found.")
else:
    # Ensure numeric Rank
    df_party["Rank"] = pd.to_numeric(df_party["Rank"], errors="coerce").astype("Int64")

    # Party × Rank counts
    rank_xt = (
        pd.crosstab(df_party["Party"], df_party["Rank"])  # columns are ranks 1,2,3,...
        .fillna(0)
        .astype(int)
        .sort_index(axis=1)  # ascending ranks
    )

    # Rename Rank=1 to 'Won'
    if 1 in rank_xt.columns:
        rank_xt = rank_xt.rename(columns={1: "Won"})

    # Total contested = sum across all rank columns
    rank_xt["Contested"] = rank_xt.sum(axis=1)

    # Column order: Won, then ranks 2..N, then Contested
    other_ranks = [c for c in rank_xt.columns if isinstance(c, (int, np.integer)) and c != 1]
    ordered_cols = (["Won"] if "Won" in rank_xt.columns else []) + other_ranks + ["Contested"]
    perf_table = rank_xt[ordered_cols].reset_index().rename(columns={"Party": "Party"})

    # Sort rows by Won desc then Contested desc (if Won exists)
    sort_cols = [c for c in ["Won", "Contested"] if c in perf_table.columns]
    if sort_cols:
        perf_table = perf_table.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    # Row coloring by party
    def _row_style_party_rank(row: pd.Series):
        party = str(row.get("Party", "")).strip()
        alias = {"CPM": "CPI(M)"}  # normalize
        party_key = alias.get(party, party)
        color = PARTY_BG_COLORS.get(party_key, DEFAULT_BG_COLOR)
        return [f"background-color: {color}"] * len(row)

    # Format all numeric columns with commas
    num_cols = [c for c in perf_table.columns if c != "Party"]
    styled_perf = perf_table.style.apply(_row_style_party_rank, axis=1).format({c: "{:,.0f}" for c in num_cols})
    render_styled_table(styled_perf)


# =====================================================
# 3) Ward & Candidate Details (winner + runner-up) — color by winning party
# =====================================================
st.subheader(f"📋 Ward-wise Results — {sel_lb}, {sel_district}")

winners = df_lb_all[df_lb_all["Rank"] == 1].copy() if "Rank" in df_lb_all.columns else df_lb_all.copy()
trailers = df_lb_all[df_lb_all["Rank"] == 2].copy() if "Rank" in df_lb_all.columns else pd.DataFrame(columns=df_lb_all.columns)

join_key = "WardCode" if "WardCode" in df_lb_all.columns else ("WardNo" if "WardNo" in df_lb_all.columns else None)

if join_key is None:
    merged = pd.merge(winners, trailers, on=["WardName"], how="left", suffixes=("_win", "_trail"))
else:
    merged = pd.merge(winners, trailers, on=[join_key], how="left", suffixes=("_win", "_trail"))

def _last2(val):
    s = str(val) if pd.notna(val) else ""
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[-2:] if digits else ""

final_cols = {
    "Ward Name": merged.get("WardName_win", merged.get("WardName", pd.Series(index=merged.index, dtype=str))),
    "Won": merged.get("Candidate_win", merged.get("Candidate", pd.Series(index=merged.index, dtype=str))),
    "Won Party": merged.get("Party_win", merged.get("Party", pd.Series(index=merged.index, dtype=str))),
    "Lead": pd.to_numeric(merged.get("Lead_win", merged.get("Lead", pd.Series(index=merged.index))), errors="coerce"),
}
trail_party = merged.get("Party_trail", pd.Series(index=merged.index, dtype=str)).fillna("")
trail_cand  = merged.get("Candidate_trail", pd.Series(index=merged.index, dtype=str)).fillna("")
final_cols["Trail"] = np.where(trail_party.eq(""), "-", trail_party + " (" + trail_cand + ")")

# Serial column
if join_key is not None:
    sl_series = merged[join_key].map(_last2).replace("", np.nan)
    if sl_series.isna().all():
        sl_series = pd.Series(range(1, len(merged) + 1), index=merged.index).astype(int).astype(str).str.zfill(2)
else:
    sl_series = pd.Series(range(1, len(merged) + 1), index=merged.index).astype(int).astype(str).str.zfill(2)

final_table = pd.DataFrame({"Sl. No": sl_series, **final_cols})

# Strict sort by numeric Sl. No, then by Ward Name
final_table["SlNoSort"] = final_table["Sl. No"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
final_table = (
    final_table.sort_values(["SlNoSort", "Ward Name"])
               .drop(columns=["SlNoSort"])
               .reset_index(drop=True)
)

# Row coloring by winning party
def _row_style_party(row: pd.Series):
    party = str(row.get("Won Party", "")).strip()
    alias = {"CPM": "CPI(M)"}  # normalize
    party_key = alias.get(party, party)
    color = PARTY_BG_COLORS.get(party_key, DEFAULT_BG_COLOR)
    return [f"background-color: {color}"] * len(row)

styled_final = final_table.style.apply(_row_style_party, axis=1).format({"Lead": "{:,.0f}"}, na_rep="–")
render_styled_table(styled_final)  # <- this function now ALWAYS hides the index
