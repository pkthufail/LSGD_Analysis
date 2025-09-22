import numpy as np
import pandas as pd
import streamlit as st

from lib.data import load_wards_2025

st.set_page_config(page_title="Delimitation Overview", page_icon="", layout="wide")
st.title("Delimitation Overview")


def _gini(values: pd.Series) -> float:
    """Return the Gini coefficient for a positive series."""
    array = values.dropna().to_numpy(dtype=float)
    if array.size == 0:
        return 0.0
    array = np.sort(array)
    total = array.sum()
    if total <= 0:
        return 0.0
    index = np.arange(1, array.size + 1)
    gini = np.sum((2 * index - array.size - 1) * array) / (array.size * total)
    return float(abs(gini))


def _format_dev_list(df: pd.DataFrame) -> str:
    if df.empty:
        return "None"
    formatted = [f"{row['WardName']} ({row['Rel. Deviation (%)']:+.1f}%)" for _, row in df.iterrows()]
    return ", ".join(formatted)


def _summarize_lb(group: pd.DataFrame) -> dict[str, object]:
    ward_count = int(len(group))
    total = int(group["TotalVoters"].sum())
    avg_total = (total / ward_count) if ward_count else 0.0
    max_total = int(group["TotalVoters"].max()) if ward_count else 0
    min_total = int(group["TotalVoters"].min()) if ward_count else 0
    ratio = (max_total / min_total) if min_total else 0.0
    gini = _gini(group["TotalVoters"]) if ward_count else 0.0
    std_total = float(group["TotalVoters"].std(ddof=0)) if ward_count else 0.0
    cv = (std_total / avg_total) if avg_total else 0.0

    if avg_total:
        rel = (group["TotalVoters"] - avg_total) / avg_total * 100
        above = int((rel >= 25).sum())
        below = int((rel <= -25).sum())
    else:
        above = below = 0

    return {
        "Total Voters": total,
        "Number of Wards": ward_count,
        "Average Voters": int(round(avg_total)) if ward_count else 0,
        "Max/Min Ratio": round(ratio, 2),
        "Gini": round(gini, 3),
        "CV": round(cv, 3),
        "Wards >= +25%": above,
        "Wards <= -25%": below,
    }


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

district_options = sorted([d for d in wards_df["District"].dropna().unique().tolist() if d])
if not district_options:
    st.info("No districts available in Wards_2025.csv.")
    st.stop()

local_tab, district_tab = st.tabs(["Local Body", "District"])


def _render_local_body_tab() -> None:
    with local_tab:
        st.markdown("### Local Body View")
        st.markdown("#### Filters")

        sel_district = st.selectbox("District", district_options, key="local_district")
        filtered_df = wards_df[wards_df["District"] == sel_district].copy()
        if filtered_df.empty:
            st.info("No wards found for the selected district.")
            return

        type_options = sorted([t for t in filtered_df["Type"].dropna().unique().tolist() if t])
        if not type_options:
            st.info("No ward types found for the selected district.")
            return

        sel_type = st.selectbox("Type", type_options, key="local_type")
        filtered_df = filtered_df[filtered_df["Type"] == sel_type].copy()
        if filtered_df.empty:
            st.info("No wards found for the selected type.")
            return

        lb_options = sorted([lb for lb in filtered_df["LBName"].dropna().unique().tolist() if lb])
        if not lb_options:
            st.info("No local body names found for the chosen filters.")
            return

        sel_lb = st.selectbox("Local Body", lb_options, key="local_lb")
        lb_df = filtered_df[filtered_df["LBName"] == sel_lb].copy()
        if lb_df.empty:
            st.info("No wards found for the selected local body.")
            return

        lb_total = int(lb_df["TotalVoters"].sum())
        ward_count = int(len(lb_df))
        max_total = int(lb_df["TotalVoters"].max()) if ward_count else 0
        min_total = int(lb_df["TotalVoters"].min()) if ward_count else 0
        avg_total = (lb_total / ward_count) if ward_count else 0
        median_total = float(lb_df["TotalVoters"].median()) if ward_count else 0
        std_total = float(lb_df["TotalVoters"].std(ddof=0)) if ward_count else 0
        cv = (std_total / avg_total) if avg_total else 0

        max_row = lb_df.loc[lb_df["TotalVoters"].idxmax()] if ward_count else None
        min_row = lb_df.loc[lb_df["TotalVoters"].idxmin()] if ward_count else None
        max_min_ratio = (max_total / min_total) if min_total else 0
        max_min_pct = ((max_min_ratio - 1) * 100) if max_min_ratio else 0

        serial = lb_df["WardCode"].str[-3:].str.zfill(3)
        if lb_total:
            percent = lb_df["TotalVoters"].div(lb_total).fillna(0) * 100
        else:
            percent = pd.Series(0.0, index=lb_df.index)
        percent = percent.round(2)

        range_from_top = (max_total - lb_df["TotalVoters"]).astype(int)
        deviation = (lb_df["TotalVoters"] - avg_total).round().astype(int)
        if avg_total:
            rel_deviation_pct = ((lb_df["TotalVoters"] - avg_total) / avg_total * 100).round(2)
        else:
            rel_deviation_pct = pd.Series(0.0, index=lb_df.index)

        result_df = pd.DataFrame({
            "SLNo.": serial,
            "WardName": lb_df["WardName"],
            "Total Voters": lb_df["TotalVoters"],
            "% of Voters": percent,
            "Range": range_from_top,
            "Deviation": deviation,
            "Rel. Deviation (%)": rel_deviation_pct,
        })

        result_df = result_df.sort_values(by="Total Voters", ascending=False).reset_index(drop=True)

        above_25 = result_df[result_df["Rel. Deviation (%)"] >= 25].copy()
        below_25 = result_df[result_df["Rel. Deviation (%)"] <= -25].copy()

        gini_value = _gini(lb_df["TotalVoters"]) if ward_count else 0
        above_25_list = _format_dev_list(above_25)
        below_25_list = _format_dev_list(below_25)

        st.markdown("### Overview")
        st.markdown(
            "\n".join(
                [
                    f"- Number of wards: **{ward_count:,}**",
                    f"- Total registered voters: **{lb_total:,}**",
                    f"- Average voters per ward: **{avg_total:,.0f}**",
                    f"- Median voters per ward: **{median_total:,.0f}**",
                    f"- Standard deviation: **approx {std_total:,.0f}**",
                ]
            )
        )

        st.markdown("### Ward Size Distribution")
        if max_row is not None and min_row is not None:
            st.markdown(
                "\n".join(
                    [
                        f"- Largest ward: **{max_row['WardName']}** ({int(max_total):,} voters)",
                        f"- Smallest ward: **{min_row['WardName']}** ({int(min_total):,} voters)",
                        f"- Max/Min ratio: **{max_min_ratio:,.2f}x** (largest ward has {max_min_pct:,.0f}% more voters than the smallest)",
                    ]
                )
            )
        else:
            st.info("Ward size distribution statistics are unavailable for the current selection.")

        st.markdown("### Inequality Measures")
        st.markdown(
            "\n".join(
                [
                    "- Rel. Deviation (%) column shows each ward's distance from the mean.",
                    f"- Wards > +25%: {above_25_list}",
                    f"- Wards < -25%: {below_25_list}",
                    f"- Gini coefficient: **approx {gini_value:.3f}** (0 = perfect equality).",
                    f"- Coefficient of Variation (CV): **{cv:,.3f}** (approx {cv * 100:,.0f}% -> ward sizes are relatively balanced).",
                ]
            )
        )

        st.markdown("### Ward Voter Distribution")
        st.caption(
            f"Total voters in {sel_lb}: {lb_total:,}. Average per ward: {avg_total:,.0f} across {ward_count} wards."
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )


def _render_district_tab() -> None:
    with district_tab:
        st.markdown("### District Overview")
        sel_district = st.selectbox("District", district_options, key="district_tab_district")
        dist_df = wards_df[wards_df["District"] == sel_district].copy()
        if dist_df.empty:
            st.info("No local bodies found for the selected district.")
            return

        summary_rows: list[dict[str, object]] = []
        grouped = dist_df.groupby(["LBName", "Type"], dropna=False)
        for (lb_name, lb_type), group in grouped:
            metrics = _summarize_lb(group)
            summary_rows.append(
                {
                    "LBName": str(lb_name),
                    "Type": str(lb_type) if pd.notna(lb_type) else "",
                    **metrics,
                }
            )

        if not summary_rows:
            st.info("No local bodies found for the selected district.")
            return

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(by="Total Voters", ascending=False).reset_index(drop=True)

        st.caption(
            f"Total voters across {sel_district}: {int(summary_df['Total Voters'].sum()):,} across {int(summary_df['Number of Wards'].sum()):,} wards."
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )


_render_local_body_tab()
_render_district_tab()
