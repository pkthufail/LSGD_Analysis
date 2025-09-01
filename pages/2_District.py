# pages/2_District.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


from lib.data import load_data, get_data_path, data_controls
from lib.colors import FRONT_BG_COLORS, PARTY_BG_COLORS, DEFAULT_BG_COLOR, FRONT_COLORS

# ---------------- Page Config ----------------
st.set_page_config(page_title="District · LSGD Explorer", page_icon="🗳️", layout="wide")
st.title("🏙️ District View")

# ---------------- Sidebar: Data controls ----------------
data_controls()
df = load_data(get_data_path()).copy()

# ---------------- Basic hygiene ----------------
if "Votes" in df.columns:
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].str.strip()

required = {"District", "Front", "Party", "LBType", "Tier"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

# ---------------- Helpers (styling & rendering) ----------------
LBTYPE_ORDER = ["Grama", "Municipality", "Corporation", "Block", "District"]
FRONT_ORDER = ["UDF", "LDF", "NDA", "OTH"]

def _apply_number_formats(styler: pd.io.formats.style.Styler, df_display: pd.DataFrame, percent_cols: list[str]):
    fmt_map = {}
    num_cols = df_display.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        fmt_map[c] = "{:,.2f}%" if c in percent_cols else "{:,.0f}"
    return styler.format(fmt_map)

def style_rows_by_palette(
    df_display: pd.DataFrame,
    key_col: str,
    palette: dict,
    default_color: str = DEFAULT_BG_COLOR,
    percent_cols: list[str] | None = None,
):
    percent_cols = percent_cols or []
    def _row_style(row: pd.Series):
        key = row.get(key_col, None)
        color = palette.get(key, "")
        return [f"background-color: {color}"] * len(row) if color else [""] * len(row)
    styler = df_display.style.apply(_row_style, axis=1)
    styler = _apply_number_formats(styler, df_display, percent_cols)
    return styler

def render_styled_table(styler: pd.io.formats.style.Styler):
    # hide index (handles pandas versions)
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

# ---------------- UI: District + Front selectors ----------------
districts = sorted([d for d in df["District"].dropna().unique().tolist()], key=lambda x: str(x))
default_dix = next((i for i, d in enumerate(districts) if str(d).strip().lower() == "malappuram"), 0)

st.subheader("Filters")
c1, c2 = st.columns([1.2, 0.8], gap="large")
with c1:
    sel_district = st.selectbox("Select District", districts, index=default_dix)
with c2:
    sel_front = st.selectbox("Select Front", FRONT_ORDER, index=0)  # default UDF

# ---------------- Filtered frames ----------------
df_d = df[df["District"] == sel_district].copy()
df_d_winners = df_d[df_d["Rank"] == 1] if "Rank" in df.columns else df_d.copy()

# Consistent LBType order for winners table
if "LBType" in df_d_winners.columns:
    df_d_winners["LBType"] = pd.Categorical(df_d_winners["LBType"], categories=LBTYPE_ORDER, ordered=True)

st.markdown("---")

# =====================================================
# 1) Seats won by Front × LBType (district, winners only)
# =====================================================
st.subheader(f"🏅 Seats Won by Front in {sel_district}")
if {"Front", "LBType"}.issubset(df_d_winners.columns):
    seat_xt = pd.crosstab(df_d_winners["Front"], df_d_winners["LBType"]).reindex(columns=LBTYPE_ORDER, fill_value=0)
    seat_xt = seat_xt.reindex(FRONT_ORDER, fill_value=0)
    seat_xt["Total"] = seat_xt.sum(axis=1)
    table_front = seat_xt.reset_index()

    styled = style_rows_by_palette(table_front, key_col="Front", palette=FRONT_BG_COLORS)
    render_styled_table(styled)
else:
    st.info("Need columns `Front` and `LBType` for this table.")

# =====================================================
# 2) Rank × Party (Tier = 'Ward') for selected District & Front
#    Show Won (Rank=1), Total Contested, Hit Rate (%)
# =====================================================
st.subheader(f"🟦 Party Performance in {sel_front} — Ward Tier (Rank × Party) in {sel_district}")

if "Rank" not in df.columns:
    st.info("`Rank` column not found; cannot compute Rank × Party table.")
else:
    # Normalize Tier and filter to Ward + selected front
    df_w = df_d.copy()
    df_w["TierNorm"] = df_w["Tier"].astype(str).str.title()
    df_w = df_w[(df_w["TierNorm"] == "Ward") & (df_w["Front"] == sel_front)].copy()

    if df_w.empty:
        st.info("No rows for Tier = 'Ward' in this district/front.")
    else:
        df_w["Rank"] = pd.to_numeric(df_w["Rank"], errors="coerce").astype("Int64")

        # Crosstab Party × Rank -> counts per rank
        rank_xt = pd.crosstab(df_w["Party"], df_w["Rank"]).fillna(0).astype(int)

        # Identify numeric rank columns and sort
        rank_cols = [c for c in rank_xt.columns if isinstance(c, (int, np.integer))]
        rank_cols_sorted = sorted(rank_cols)

        # Rename Rank 1 to 'Won' if present
        if 1 in rank_xt.columns:
            rank_xt = rank_xt.rename(columns={1: "Won"})

        # Contested = sum across rank columns (incl. 'Won' if present)
        contested_cols = (["Won"] if "Won" in rank_xt.columns else []) + [c for c in rank_cols_sorted if c != 1]
        contested_cols = [c for c in contested_cols if c in rank_xt.columns]
        rank_xt["Contested"] = rank_xt[contested_cols].sum(axis=1)

        # Hit Rate (%)
        if "Won" in rank_xt.columns:
            rank_xt["Hit Rate (%)"] = np.where(rank_xt["Contested"] > 0, (rank_xt["Won"] / rank_xt["Contested"] * 100), 0.0)
        else:
            rank_xt["Hit Rate (%)"] = 0.0

        # Final column order: Won, other ranks ascending, Contested, Hit Rate
        other_ranks = [c for c in rank_cols_sorted if c != 1 and c in rank_xt.columns]
        ordered_cols = (["Won"] if "Won" in rank_xt.columns else []) + other_ranks + ["Contested", "Hit Rate (%)"]
        ordered_cols = [c for c in ordered_cols if c in rank_xt.columns]

        # Sort rows by Won desc then Contested desc (if present)
        sort_cols = [c for c in ["Won", "Contested"] if c in rank_xt.columns]
        if sort_cols:
            rank_xt = rank_xt.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

        table_rank_party = rank_xt.reset_index().rename(columns={"index": "Party"})
        table_rank_party = table_rank_party[["Party"] + ordered_cols]

        # Style & render
        styled_rp = style_rows_by_palette(
            table_rank_party,
            key_col="Party",
            palette=PARTY_BG_COLORS,
            default_color=DEFAULT_BG_COLOR,
            percent_cols=["Hit Rate (%)"],
        )
        render_styled_table(styled_rp)

# =====================================================
# 3) Prep Local Body winners (Ward tier) once
#    -> used by BOTH 'Leaders count' and 'Final LB list'
# =====================================================
mask = (
    (df_d["Tier"].astype(str).str.title() == "Ward") &
    (df_d["Rank"] == 1 if "Rank" in df_d.columns else True) &
    (df_d["LBType"].isin(["Grama", "Municipality", "Corporation"]))
)
df_lb = df_d[mask].copy()

# Build lb_summary if there is data
lb_summary = pd.DataFrame()
if not df_lb.empty:
    lb_summary = (
        df_lb.groupby(["LBCode", "LBName", "LBType", "Front"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # ensure all front columns exist
    for fr in FRONT_ORDER:
        if fr not in lb_summary.columns:
            lb_summary[fr] = 0

    # derive leader front (ties → 'TIE', zero → 'NONE')
    def _leader_row(row):
        counts = {fr: row.get(fr, 0) for fr in FRONT_ORDER}
        maxv = max(counts.values()) if counts else 0
        leaders = [fr for fr, v in counts.items() if v == maxv and maxv > 0]
        return leaders[0] if len(leaders) == 1 else ("TIE" if maxv > 0 else "NONE")

    lb_summary["Leader"] = lb_summary.apply(_leader_row, axis=1)

# =====================================================
# 4) NEW: Leaders count per LBType (Fronts × Panchayath/Municipality/Corporation)
#     (PLACE THIS BEFORE THE FINAL LB LIST)
# =====================================================
st.subheader("🏁 Local Bodies Led by Front")

if lb_summary.empty:
    st.info("No Ward-tier winners for the selected district and LB types.")
else:
    leaders_only = lb_summary[lb_summary["Leader"].isin(FRONT_ORDER)].copy()
    if leaders_only.empty:
        st.info("No local bodies with a single leading front in the selected district.")
    else:
        lead_xt = pd.crosstab(leaders_only["Leader"], leaders_only["LBType"])
        lb_cols = ["Grama", "Municipality", "Corporation"]
        lead_xt = (
            lead_xt.reindex(index=FRONT_ORDER, fill_value=0)
                   .reindex(columns=lb_cols, fill_value=0)
        )
        lead_table = lead_xt.reset_index().rename(columns={"Leader": "Front"})

        styled_leaders = style_rows_by_palette(
            lead_table,
            key_col="Front",
            palette=FRONT_BG_COLORS,
            default_color=DEFAULT_BG_COLOR,
            percent_cols=[],  # counts only
        )
        render_styled_table(styled_leaders)
        
# --- Bar chart: local bodies led by front (labels; no gaps for zero fronts; hide empty 'Corporation') ---
lead_long = lead_table.melt(
    id_vars="Front",
    value_vars=["Grama", "Municipality", "Corporation"],
    var_name="LBType",
    value_name="Count",
)

# keep only LB types that have any count > 0 (drops 'Corporation' if absent)
present_types = [t for t in ["Grama", "Municipality", "Corporation"]
                 if lead_long.loc[lead_long["LBType"] == t, "Count"].sum() > 0]
lead_long = lead_long[lead_long["LBType"].isin(present_types)]

# drop zero rows so no slot is reserved for zero-count fronts
lead_long = lead_long[lead_long["Count"] > 0]

# fronts that actually appear (prevents empty-trace gaps)
present_fronts = [f for f in FRONT_ORDER if (lead_long.loc[lead_long["Front"] == f, "Count"].sum() > 0)]

if not lead_long.empty and present_types and present_fronts:
    lead_long["Label"] = lead_long["Count"].map(lambda x: f"{x:,}")

    fig = px.bar(
        lead_long,
        x="LBType",
        y="Count",
        color="Front",
        text="Label",
        barmode="group",
        category_orders={"LBType": present_types, "Front": present_fronts},
        color_discrete_map=FRONT_COLORS,
        title=f"Local Bodies Led by Front — {sel_district}",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Count",
        legend=dict(orientation="h", x=1.0, y=1.02, xanchor="right", yanchor="bottom"),
        bargap=0.2,
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No local bodies led data to display.")


# =====================================================
# 5) FINAL: Local Body table (front-wise seats per LB), row colored by leading front
#     + Search box to filter by LBName / LBType / Leader
# =====================================================
st.subheader(f"🏘️ Front-wise Seats Won in Local Body — {sel_district}")

if lb_summary.empty:
    st.info("No Ward-tier winners for the selected district and LB types.")
else:
    # --- Search box (filters by LBName / LBType / Leader) ---
    search_q = st.text_input(
        "Search local bodies",
        value="",
        placeholder="Type LB name (e.g., 'Kottakkal'), or LB type (Grama/Municipality/Corporation), or Leader (UDF/LDF/NDA/OTH)"
    ).strip()

    df_show = lb_summary.copy()

    if search_q:
        mask = (
            df_show["LBName"].astype(str).str.contains(search_q, case=False, na=False)
            | df_show["LBType"].astype(str).str.contains(search_q, case=False, na=False)
            | df_show["Leader"].astype(str).str.contains(search_q, case=False, na=False)
        )
        df_show = df_show[mask]

    # Sl No from LBCode (first letter + last 2 digits)
    df_show["Sl No"] = df_show["LBCode"].astype(str).str[0] + df_show["LBCode"].astype(str).str[-2:]

    # Reorder & sort
    columns_order = ["Sl No", "LBName", "LBType", "UDF", "LDF", "NDA", "OTH", "Leader"]
    lb_table = df_show[columns_order].sort_values(["LBType", "LBName"]).reset_index(drop=True)

    # Info on how many shown
    st.caption(f"Showing {len(lb_table):,} of {len(lb_summary):,} local bodies")

    if lb_table.empty:
        st.warning("No matches for your search.")
    else:
        # Row color by Leader front (TIE/NONE -> neutral)
        def _row_style(row: pd.Series):
            key = row.get("Leader")
            color = FRONT_BG_COLORS.get(key, DEFAULT_BG_COLOR)
            return [f"background-color: {color}"] * len(row)

        styled = lb_table.style.apply(_row_style, axis=1).format({
            "UDF": "{:,.0f}", "LDF": "{:,.0f}", "NDA": "{:,.0f}", "OTH": "{:,.0f}"
        })
        render_styled_table(styled)
