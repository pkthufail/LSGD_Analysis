import pandas as pd
import streamlit as st

from lib.data import load_wards_2025

st.set_page_config(page_title="Delimitation Overview", page_icon="", layout="wide")
st.title("Delimitation Overview")

wards_df = load_wards_2025().copy()
if wards_df.empty:
    st.warning("Wards_2025.csv could not be loaded or is empty.")
    st.stop()

# Ensure clean, typed columns for calculations
wards_df["District"] = wards_df.get("District", "").astype(str).str.strip()
wards_df["Type"] = wards_df.get("Type", "").astype(str).str.strip()
wards_df["LBName"] = wards_df.get("LBName", "").astype(str).str.strip()
wards_df["WardCode"] = wards_df.get("WardCode", "").astype(str).str.strip()
wards_df["WardName"] = wards_df.get("WardName", "").astype(str).str.strip()
wards_df["TotalVoters"] = pd.to_numeric(wards_df.get("TotalVoters", 0), errors="coerce").fillna(0).astype(int)

# Filters
st.markdown("### Filters")
district_options = sorted([d for d in wards_df["District"].dropna().unique().tolist() if d])
if not district_options:
    st.info("No districts available in Wards_2025.csv.")
    st.stop()

sel_district = st.selectbox("District", district_options)
filtered_df = wards_df[wards_df["District"] == sel_district].copy()

if filtered_df.empty:
    st.info("No wards found for the selected district.")
    st.stop()

type_options = sorted([t for t in filtered_df["Type"].dropna().unique().tolist() if t])
if not type_options:
    st.info("No ward types found for the selected district.")
    st.stop()

sel_type = st.selectbox("Type", type_options)
filtered_df = filtered_df[filtered_df["Type"] == sel_type].copy()

if filtered_df.empty:
    st.info("No wards found for the selected type.")
    st.stop()

lb_options = sorted([lb for lb in filtered_df["LBName"].dropna().unique().tolist() if lb])
if not lb_options:
    st.info("No local body names found for the chosen filters.")
    st.stop()

sel_lb = st.selectbox("Local Body", lb_options)
lb_df = filtered_df[filtered_df["LBName"] == sel_lb].copy()

if lb_df.empty:
    st.info("No wards found for the selected local body.")
    st.stop()

lb_total = int(lb_df["TotalVoters"].sum())
ward_count = int(len(lb_df))
max_total = int(lb_df["TotalVoters"].max()) if ward_count else 0
avg_total = (lb_total / ward_count) if ward_count else 0

serial = lb_df["WardCode"].str[-3:].str.zfill(3)
if lb_total:
    percent = lb_df["TotalVoters"].div(lb_total).fillna(0) * 100
else:
    percent = pd.Series(0.0, index=lb_df.index)
percent = percent.round(2)

range_from_top = (max_total - lb_df["TotalVoters"]).astype(int)
deviation = (lb_df["TotalVoters"] - avg_total).round().astype(int)

result_df = pd.DataFrame({
    "SLNo.": serial,
    "WardName": lb_df["WardName"],
    "Total Voters": lb_df["TotalVoters"],
    "% of Voters": percent,
    "Range": range_from_top,
    "Deviation": deviation,
})

result_df = result_df.sort_values(by="Total Voters", ascending=False).reset_index(drop=True)

st.markdown("### Ward Voter Distribution")
st.caption(
    f"Total voters in {sel_lb}: {lb_total:,}. Average per ward: {avg_total:,.0f} across {ward_count} wards."
)

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True,
)
