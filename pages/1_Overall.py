# pages/1_Overall.py
import streamlit as st
import pandas as pd
import numpy as np

from lib.data import load_data, get_data_path, data_controls
from lib.colors import FRONT_BG_COLORS, PARTY_BG_COLORS, DEFAULT_BG_COLOR

# Safe Styler import for type hints across pandas versions
try:
    from pandas.io.formats.style import Styler
except Exception:  # pandas too old / lazy import differences
    from typing import Any as Styler  # fallback for typing only


st.set_page_config(page_title="Overall · LSGD Explorer", page_icon="🗳️", layout="wide")
st.title("🗺️ Kerala Local Body Election 2020 - Dashboard")
st.subheader("📊 Overall Summary")

# ---------- Sidebar: data controls only ----------
data_controls()
df = load_data(get_data_path()).copy()

required_cols = {"Front", "LBType", "Candidate", "Votes"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns: {', '.join(sorted(missing))}")
    st.stop()

if "Votes" in df.columns:
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)

# ---------- PAGE filter: District (default = All Kerala) ----------
st.markdown("### Filters")
if "District" in df.columns:
    districts = ["All Kerala"] + sorted(df["District"].dropna().unique().tolist())
    sel_district = st.selectbox("District", districts, index=0, key="overall_district")
else:
    st.info("District column not found; showing All Kerala.")
    sel_district = "All Kerala"

def _apply_district_filter(frame: pd.DataFrame) -> pd.DataFrame:
    if sel_district == "All Kerala" or "District" not in frame.columns:
        return frame
    return frame[frame["District"] == sel_district]

# Winners-only for seat counts if available, after district filter
df_filtered = _apply_district_filter(df)
df_winners = df_filtered[df_filtered["Rank"] == 1].copy() if "Rank" in df.columns else df_filtered.copy()

LBTYPE_ORDER = ["Grama", "Municipality", "Corporation", "Block", "District"]
FRONT_ORDER = ["UDF", "LDF", "NDA", "OTH"]

if "LBType" in df_winners.columns:
    df_winners["LBType"] = pd.Categorical(df_winners["LBType"], categories=LBTYPE_ORDER, ordered=True)

# ---------- helpers ----------
def _hide_index(styler: Styler):
    try:
        return styler.hide(axis="index")
    except Exception:
        return styler.hide_index()

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
    total_label: str | None = None,
    percent_cols: list[str] | None = None,
):
    percent_cols = percent_cols or []
    def _row_style(row: pd.Series):
        key = row.get(key_col, None)
        if total_label is not None and key == total_label:
            color = default_color
        else:
            color = palette.get(key, "")
        return [f"background-color: {color}"] * len(row) if color else [""] * len(row)

    styler = df_display.style.apply(_row_style, axis=1)
    styler = _apply_number_formats(styler, df_display, percent_cols)
    return _hide_index(styler)

def render_styled_table(styler: pd.io.formats.style.Styler):
    # Render as HTML so Streamlit doesn't re-add an index column; make responsive
    html = styler.to_html()
    st.markdown(
        """
        <style>
          .tbl-wrap { width: 100%; overflow-x: auto; }
          .tbl-wrap table { width: 100%; border-collapse: collapse; table-layout: auto; }
          .tbl-wrap th, .tbl-wrap td { padding: 6px 8px; }
          @media (max-width: 1200px) {
            .tbl-wrap th, .tbl-wrap td { font-size: 0.9rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='tbl-wrap'>{html}</div>", unsafe_allow_html=True)

# ---------- tabs ----------
tab1, tab2 = st.tabs(["🏆 Seats Won", "📉 Vote Share"])

# =====================================================
# TAB 1: Seats Won  (summary + collapsible per-front party tables)
# =====================================================
with tab1:
    loc_label = sel_district
    st.subheader(f"🏅 Number of Seats Won by Front — {loc_label}")

    if {"Front", "LBType"}.issubset(df_winners.columns):
        # Summary: Front × LBType + Total
        seat_xt = pd.crosstab(df_winners["Front"], df_winners["LBType"]).reindex(columns=LBTYPE_ORDER, fill_value=0)
        seat_xt = seat_xt.reindex(FRONT_ORDER, fill_value=0)
        seat_xt["Total"] = seat_xt.sum(axis=1)

        total_row = pd.DataFrame([seat_xt.sum()], index=pd.Index(["Total"], name="Front"))
        front_table = pd.concat([seat_xt, total_row], axis=0).reset_index()

        styled_front = style_rows_by_palette(
            front_table,
            key_col="Front",
            palette=FRONT_BG_COLORS,
            default_color=DEFAULT_BG_COLOR,
            total_label="Total",
            percent_cols=[],
        )
        render_styled_table(styled_front)
    else:
        st.info("Need columns `Front` and `LBType` for this table.")

    # Collapsible sections: party-wise seats for each front (like vote share tab)
    if not df_winners.empty:
        for front in FRONT_ORDER:
            with st.expander(f"{front} – Party-wise Seats ({loc_label})", expanded=False):
                df_f = df_winners[df_winners["Front"] == front].copy()
                if df_f.empty or "Party" not in df_f.columns:
                    st.info("No rows for this front.")
                    continue

                party_xt = pd.crosstab(df_f["Party"], df_f["LBType"]).reindex(columns=LBTYPE_ORDER, fill_value=0)
                party_xt["Total"] = party_xt.sum(axis=1)
                party_xt = party_xt.sort_values("Total", ascending=False).reset_index().rename(columns={"index": "Party"})

                styled_party = style_rows_by_palette(
                    party_xt,
                    key_col="Party",
                    palette=PARTY_BG_COLORS,
                    default_color=DEFAULT_BG_COLOR,
                    percent_cols=[],
                )
                render_styled_table(styled_party)

# =====================================================
# TAB 2: Vote Share (unchanged)
# =====================================================
with tab2:
    loc_label = sel_district
    st.subheader(f"🗳️ Total Vote Share by Front — {loc_label}")

    # Filtered frame for votes (respect district selection)
    df_votes = _apply_district_filter(df)
    if "Tier" in df_votes.columns:
        df_votes = df_votes[df_votes["Tier"] == "Ward"].copy()
    df_votes = df_votes[df_votes["Votes"].notna()].copy()

    if df_votes.empty:
        st.info("No vote rows available for calculation.")
    else:
        total_votes = df_votes["Votes"].sum()

        front_votes = (
            df_votes.groupby("Front", as_index=False)
            .agg(Votes=("Votes", "sum"))
            .sort_values("Votes", ascending=False)
        )
        front_votes = (
            pd.DataFrame({"Front": FRONT_ORDER})
            .merge(front_votes, on="Front", how="left")
            .fillna({"Votes": 0})
        )
        front_votes["% Share"] = np.where(
            total_votes > 0,
            (front_votes["Votes"] / total_votes * 100),
            0.0,
        )

        styled_front_votes = style_rows_by_palette(
            front_votes,
            key_col="Front",
            palette=FRONT_BG_COLORS,
            default_color=DEFAULT_BG_COLOR,
            percent_cols=["% Share"],
        )
        render_styled_table(styled_front_votes)

        # Collapsible: party-wise vote share per front
        for front in FRONT_ORDER:
            with st.expander(f"{front} – Party-wise Vote Share ({loc_label})", expanded=False):
                df_f = df_votes[df_votes["Front"] == front].copy()
                if df_f.empty:
                    st.info("No rows for this front.")
                    continue

                front_total = df_f["Votes"].sum()
                party_votes = (
                    df_f.groupby("Party", as_index=False)
                    .agg(Votes=("Votes", "sum"))
                    .sort_values("Votes", ascending=False)
                )
                party_votes["% Front Share"] = np.where(
                    front_total > 0, (party_votes["Votes"] / front_total * 100), 0.0,
                )
                party_votes["% Total Share"] = np.where(
                    total_votes > 0, (party_votes["Votes"] / total_votes * 100), 0.0,
                )

                styled_party_votes = style_rows_by_palette(
                    party_votes,
                    key_col="Party",
                    palette=PARTY_BG_COLORS,
                    default_color=DEFAULT_BG_COLOR,
                    percent_cols=["% Front Share", "% Total Share"],
                )
                render_styled_table(styled_party_votes)
