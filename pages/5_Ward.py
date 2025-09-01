# pages/5_Ward.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from lib.data import load_data, get_data_path, data_controls
from lib.colors import PARTY_BG_COLORS, FRONT_COLORS, DEFAULT_BG_COLOR

# ---------------- Page Config ----------------
st.set_page_config(page_title="Ward · LSGD Explorer", page_icon="🗳️", layout="wide")
st.title("🗳️ Ward View")

# ---------------- Sidebar: Data controls ----------------
data_controls()
df = load_data(get_data_path()).copy()

# ---------------- Hygiene ----------------
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].str.strip()

if "Votes" in df.columns:
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0)

# Scope to Ward tier only (ward-level results)
df = df[df["Tier"].astype(str).str.title() == "Ward"].copy()

# ---------------- Small helpers ----------------
def render_styled_table(styler_or_df, number_cols=None, percent_cols=None):
    number_cols = number_cols or []
    percent_cols = percent_cols or []

    # Accept either a DataFrame or an existing Styler
    if isinstance(styler_or_df, pd.io.formats.style.Styler):
        styler = styler_or_df
    else:
        styler = styler_or_df.style

    # Number / percent formatting (safe even if some cols missing)
    fmt_map = {**{c: "{:,.0f}" for c in number_cols},
               **{c: "{:,.2f}%" for c in percent_cols}}
    styler = styler.format(fmt_map)

    # Always hide index
    try:
        styler = styler.hide(axis="index")
    except Exception:
        styler = styler.hide_index()

    # Responsive HTML wrapper
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


def party_color(party: str) -> str:
    # Normalize common alias
    alias = {"CPM": "CPI(M)"}
    key = alias.get(str(party).strip(), str(party).strip())
    return PARTY_BG_COLORS.get(key, "#CCCCCC")

# ---------------- Filters (on page) ----------------
st.markdown("### Filters")

# District (default Malappuram)
districts = sorted(df["District"].dropna().unique().tolist(), key=lambda x: str(x))
default_dix = next((i for i, d in enumerate(districts) if str(d).strip().lower() == "malappuram"), 0)
sel_district = st.selectbox("District", districts, index=default_dix, key="ward_district")

# Local Body list (by district)
df_d = df[df["District"] == sel_district].copy()
lbnames = sorted(df_d["LBName"].dropna().unique().tolist(), key=lambda x: str(x))
sel_lb = st.selectbox("Local Body", lbnames, index=0 if lbnames else 0, key="ward_lb")

if df_d.empty or not lbnames:
    st.info("No ward rows found for the selected District.")
    st.stop()

# Ward list (by LB) with search + dropdown
df_lb = df_d[df_d["LBName"] == sel_lb].copy()
all_wards = sorted(df_lb["WardName"].dropna().unique().tolist(), key=lambda x: str(x))

c1, c2 = st.columns([1.2, 1.2], gap="large")
with c1:
    ward_query = st.text_input("Search Ward", value="", placeholder="Type to filter wards…").strip()
filtered_wards = [w for w in all_wards if ward_query.lower() in str(w).lower()] if ward_query else all_wards
with c2:
    sel_ward = st.selectbox("Ward", filtered_wards, index=0 if filtered_wards else 0, key="ward_name")

if not filtered_wards:
    st.warning("No wards match your search.")
    st.stop()

# ---------------- Data for selected Ward ----------------
df_w = df_lb[df_lb["WardName"] == sel_ward].copy()
if df_w.empty:
    st.info("No rows for the selected ward.")
    st.stop()

# Aggregate by Candidate/Party/Front to ensure one row per candidate
group_cols = [c for c in ["Candidate", "Party", "Front"] if c in df_w.columns]
ward_votes = df_w.groupby(group_cols, as_index=False)["Votes"].sum()

# Total votes & percentage
total_votes = ward_votes["Votes"].sum()
ward_votes["%"] = np.where(total_votes > 0, ward_votes["Votes"] / total_votes * 100, 0.0)

# Sort by votes desc to identify winner and runner-up
ward_votes = ward_votes.sort_values("Votes", ascending=False).reset_index(drop=True)

# Winner / margin
winner = ward_votes.iloc[0] if not ward_votes.empty else None
second_votes = ward_votes.iloc[1]["Votes"] if len(ward_votes) > 1 else 0
lead = (winner["Votes"] - second_votes) if winner is not None else 0

# ---------------- Headline: Winning candidate line ----------------
if winner is not None:
    w_name = winner.get("Candidate", "—")
    w_party = winner.get("Party", "—")
    w_front = winner.get("Front", "—")
    # Colored name tokens
    party_hex = party_color(w_party)
    front_hex = FRONT_COLORS.get(str(w_front), "#333")
    line = (
        f"<span style='font-weight:700'>{w_name}</span> "
        f"from <span style='font-weight:700; color:{party_hex}'>{w_party}</span> "
        f"(<span style='font-weight:700; color:{front_hex}'>{w_front}</span>) "
        f"by <span style='font-weight:700'>{int(lead):,}</span> votes."
    )
    st.markdown(line, unsafe_allow_html=True)
else:
    st.markdown("No winner identified for this ward.")

st.markdown("---")

# ---------------- Table: Name · Party · Front · Votes · % ----------------
st.subheader(f"📋 Candidates — {sel_ward}, {sel_lb}")

# Select & rename for display
display_cols = {}
display_cols["Name"] = ward_votes.get("Candidate", pd.Series(index=ward_votes.index, dtype=str))
display_cols["Party"] = ward_votes.get("Party", pd.Series(index=ward_votes.index, dtype=str))
display_cols["Front"] = ward_votes.get("Front", pd.Series(index=ward_votes.index, dtype=str))
display_cols["Votes"] = ward_votes["Votes"]
display_cols["%"] = ward_votes["%"]

tbl = pd.DataFrame(display_cols)

# Optional: color rows by party for readability
def _row_style_party(row: pd.Series):
    color = party_color(row.get("Party", ""))
    return [f"background-color: {color}"] * len(row) if color else [""] * len(row)

styled_tbl = tbl.style.apply(_row_style_party, axis=1)
render_styled_table(styled_tbl, number_cols=["Votes"], percent_cols=["%"])

# ---------------- Pie Chart: Votes by Party (party colors) ----------------
st.subheader("🟢 Vote Share (Pie)")

# Combine any duplicate party rows first
pie_df = (
    ward_votes.groupby("Party", as_index=False)["Votes"].sum()
              .sort_values("Votes", ascending=False)
)
# Guard: nothing to plot
if pie_df["Votes"].sum() <= 0:
    st.info("No vote data to display.")
else:
    # Build color domain/range that exactly matches the parties shown
    parties_present = pie_df["Party"].astype(str).fillna("Unknown").tolist()
    color_range = [party_color(p) for p in parties_present]

    pie = (
        alt.Chart(pie_df)
        # Compute assembly total ONCE and attach as a new field 'total'
        .transform_joinaggregate(total="sum(Votes)")
        # Compute percentage safely using the joined 'total'
        .transform_calculate(Percent="datum.Votes / datum.total * 100")
        .mark_arc(outerRadius=130)
        .encode(
            theta=alt.Theta("Votes:Q"),
            color=alt.Color(
                "Party:N",
                scale=alt.Scale(domain=parties_present, range=color_range),
                legend=alt.Legend(title="Party")
            ),
            tooltip=[
                alt.Tooltip("Party:N"),
                alt.Tooltip("Votes:Q", format=",d"),
                alt.Tooltip("Percent:Q", format=".2f")
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(pie, use_container_width=True)

