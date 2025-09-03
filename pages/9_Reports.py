import streamlit as st
import pandas as pd
import numpy as np

from lib.data import load_data, get_data_path, data_controls
from lib.ui import render_styled_table
from lib.colors import FRONT_BG_COLORS, DEFAULT_BG_COLOR


st.set_page_config(page_title="Reports – LSGD Explorer", page_icon="🗂️", layout="wide")
st.title("Reports")
st.caption("Ward-tier comparisons and summaries. Use tabs below.")

# Sidebar controls and data
data_controls()
df = load_data(get_data_path()).copy()

# Define sets
PARTIES_MUSLIM = ["IUML", "SDPI", "WPI", "INL", "NSC", "PDP"]
MAJOR_PARTIES = ["INC", "CPI(M)", "CPI", "BJP", "IUML"]
FRONTS = ["UDF", "LDF", "NDA", "OTH"]

# Basic validations
required_cols = {"Party", "District", "Votes", "Tier"}
if not required_cols.issubset(df.columns):
    missing = ", ".join(sorted(required_cols - set(df.columns)))
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Hygiene and Ward-tier scope
df = df.copy()
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype(str).str.strip()

df["PartyUpper"] = df["Party"].astype(str).str.upper().str.strip()
if "Front" in df.columns:
    df["FrontUpper"] = df["Front"].astype(str).str.upper().str.strip()
else:
    df["FrontUpper"] = ""

# Prefer normalized Tier if available
if "TierNorm" in df.columns:
    tier_series = df["TierNorm"].astype(str)
else:
    tier_series = df["Tier"].astype(str).str.title()

df_ward_all = df[tier_series == "Ward"].copy()


def _highlight_if_others_gt_iuml(row: pd.Series, parties: list[str]) -> list[str]:
    try:
        iuml_val = pd.to_numeric(row.get("IUML", 0), errors="coerce")
    except Exception:
        iuml_val = 0
    iuml_val = 0 if pd.isna(iuml_val) else float(iuml_val)
    others_vals = []
    for p in parties:
        if p == "IUML":
            continue
        try:
            v = pd.to_numeric(row.get(p, 0), errors="coerce")
        except Exception:
            v = 0
        v = 0 if pd.isna(v) else float(v)
        others_vals.append(v)
    highlight = any(v > iuml_val for v in others_vals)
    return ["background-color: #FFF0F0" if highlight else ""] * len(row)


def _row_color_by_max_front(row: pd.Series) -> list[str]:
    best_front = None
    best_val = -1
    for f in FRONTS:
        v = float(row.get(f.upper(), 0.0))
        if v > best_val:
            best_val = v
            best_front = f
    color = FRONT_BG_COLORS.get(best_front, DEFAULT_BG_COLOR)
    return [f"background-color: {color}"] * len(row)


# Tabs
tab_iuml, tab_general = st.tabs(["IUML", "General"])


with tab_iuml:
    st.subheader("Seats by District (Ward-tier): IUML vs Others")
    df_ward = df_ward_all[df_ward_all["PartyUpper"].isin(PARTIES_MUSLIM)].copy()
    if "Rank" not in df_ward.columns:
        st.info("Rank column not found; cannot compute seats won. Showing zeros.")
        seats_xt = pd.DataFrame(columns=["District"] + PARTIES_MUSLIM)
    else:
        df_ward["Rank"] = pd.to_numeric(df_ward["Rank"], errors="coerce").astype("Int64")
        winners_ward = df_ward[df_ward["Rank"] == 1].copy()
        if winners_ward.empty:
            seats_xt = pd.DataFrame(columns=["District"] + PARTIES_MUSLIM)
        else:
            seats = (
                pd.crosstab(winners_ward["District"], winners_ward["PartyUpper"]).reindex(columns=PARTIES_MUSLIM, fill_value=0)
            )
            seats.index.name = "District"
            seats_xt = seats.reset_index()

    # All Kerala totals row
    if not seats_xt.empty:
        totals = {p: int(seats_xt.get(p, pd.Series(dtype=int)).sum()) for p in PARTIES_MUSLIM}
        totals_row = {"District": "All Kerala", **totals}
        seats_xt = pd.concat([seats_xt.sort_values("District"), pd.DataFrame([totals_row])], ignore_index=True)

    seats_styler = seats_xt.style.apply(lambda r: _highlight_if_others_gt_iuml(r, PARTIES_MUSLIM), axis=1)
    render_styled_table(seats_styler, fmt={p: "{:,.0f}" for p in PARTIES_MUSLIM})

    st.divider()
    st.subheader("Votes by District (Ward-tier): IUML vs Others")
    df_ward_v = df_ward_all[df_ward_all["PartyUpper"].isin(PARTIES_MUSLIM)].copy()
    if df_ward_v.empty or "Votes" not in df_ward_v.columns:
        st.info("No Ward-tier vote rows available.")
        votes_xt = pd.DataFrame(columns=["District"] + PARTIES_MUSLIM)
    else:
        df_ward_v["Votes"] = pd.to_numeric(df_ward_v["Votes"], errors="coerce").fillna(0).astype(int)
        agg = df_ward_v.groupby(["District", "PartyUpper"], as_index=False)["Votes"].sum()
        votes = agg.pivot_table(index="District", columns="PartyUpper", values="Votes", aggfunc="sum", fill_value=0)
        votes = votes.reindex(columns=PARTIES_MUSLIM, fill_value=0)
        votes.index.name = "District"
        votes_xt = votes.reset_index()

    if not votes_xt.empty:
        totals_v = {p: int(votes_xt.get(p, pd.Series(dtype=int)).sum()) for p in PARTIES_MUSLIM}
        totals_row_v = {"District": "All Kerala", **totals_v}
        votes_xt = pd.concat([votes_xt.sort_values("District"), pd.DataFrame([totals_row_v])], ignore_index=True)

    votes_styler = votes_xt.style.apply(lambda r: _highlight_if_others_gt_iuml(r, PARTIES_MUSLIM), axis=1)
    render_styled_table(votes_styler, fmt={p: "{:,.0f}" for p in PARTIES_MUSLIM})

    st.divider()
    st.subheader("Top Assemblies for IUML (Ward-tier)")
    assembly_candidates = ["Assembly", "ACName", "AssemblyName", "Constituency"]
    assembly_col = next((c for c in assembly_candidates if c in df.columns), None)
    if assembly_col is None:
        st.info("No assembly column found (looked for: Assembly / ACName / AssemblyName / Constituency).")
    else:
        df_ward_all["Votes"] = pd.to_numeric(df_ward_all["Votes"], errors="coerce").fillna(0).astype(int)
        iuml_mask = df_ward_all["PartyUpper"] == "IUML"
        iuml_votes = (
            df_ward_all.loc[iuml_mask]
            .groupby(assembly_col, as_index=False)["Votes"].sum()
            .rename(columns={"Votes": "IUML Votes"})
        )
        total_votes = (
            df_ward_all.groupby(assembly_col, as_index=False)["Votes"].sum()
            .rename(columns={"Votes": "Total Votes"})
        )
        if "Front" in df_ward_all.columns:
            udf_votes = (
                df_ward_all.loc[df_ward_all["FrontUpper"] == "UDF"]
                .groupby(assembly_col, as_index=False)["Votes"].sum()
                .rename(columns={"Votes": "UDF Votes"})
            )
        else:
            udf_votes = pd.DataFrame(columns=[assembly_col, "UDF Votes"])

        strong = pd.merge(iuml_votes, total_votes, on=assembly_col, how="left")
        strong = pd.merge(strong, udf_votes, on=assembly_col, how="left")
        strong["UDF Votes"] = pd.to_numeric(strong.get("UDF Votes", 0), errors="coerce").fillna(0).astype(int)
        strong["IUML % of Total"] = np.where(strong["Total Votes"] > 0, strong["IUML Votes"] / strong["Total Votes"] * 100, 0.0)
        strong["IUML % of UDF"] = np.where(strong["UDF Votes"] > 0, strong["IUML Votes"] / strong["UDF Votes"] * 100, 0.0)
        strong = strong.sort_values(["IUML Votes", assembly_col], ascending=[False, True]).head(50)
        strong = strong.rename(columns={assembly_col: "Assembly"})[
            ["Assembly", "IUML Votes", "Total Votes", "IUML % of Total", "UDF Votes", "IUML % of UDF"]
        ]
        strong.insert(0, "Rank", range(1, len(strong) + 1))
        try:
            st.dataframe(strong, use_container_width=True, hide_index=True)
        except Exception:
            render_styled_table(strong)

    st.divider()
    st.subheader("IUML: Candidate Age Profile (Ward-tier)")
    iu = df_ward_all[df_ward_all["PartyUpper"] == "IUML"].copy()
    if iu.empty or "Age" not in iu.columns:
        st.info("No IUML rows with Age available at Ward tier.")
    else:
        iu["Age"] = pd.to_numeric(iu["Age"], errors="coerce")
        grp = iu.groupby("District", dropna=False)
        denom = grp["Age"].apply(lambda s: s.notna().sum()).rename("N")
        u30 = grp["Age"].apply(lambda s: (s < 30).sum()).rename("<30")
        u40 = grp["Age"].apply(lambda s: (s < 40).sum()).rename("<40")
        u50 = grp["Age"].apply(lambda s: (s < 50).sum()).rename("<50")
        age_df = pd.concat([denom, u30, u40, u50], axis=1).reset_index().rename(columns={"index": "District"})
        for col, out in [("<30", "% <30"), ("<40", "% <40"), ("<50", "% <50")]:
            age_df[out] = np.where(age_df["N"] > 0, age_df[col] / age_df["N"] * 100, 0.0)
        age_out = age_df[["District", "% <30", "% <40", "% <50"]].copy()
        N_all = int(denom.sum())
        u30_all = int(u30.sum())
        u40_all = int(u40.sum())
        u50_all = int(u50.sum())
        row_all = {
            "District": "All Kerala",
            "% <30": (u30_all / N_all * 100) if N_all > 0 else 0.0,
            "% <40": (u40_all / N_all * 100) if N_all > 0 else 0.0,
            "% <50": (u50_all / N_all * 100) if N_all > 0 else 0.0,
        }
        age_out = pd.concat([age_out.sort_values("District"), pd.DataFrame([row_all])], ignore_index=True)
        render_styled_table(age_out, fmt={"% <30": "{:,.2f}%", "% <40": "{:,.2f}%", "% <50": "{:,.2f}%"})


with tab_general:
    st.subheader("Candidate Age Profiles – Major Parties (Ward-tier)")
    if "Age" not in df_ward_all.columns or "District" not in df_ward_all.columns:
        st.info("Need Age and District columns for this table.")
    else:
        d = df_ward_all.copy()
        d["Age"] = pd.to_numeric(d["Age"], errors="coerce")
        d_party = d[d["PartyUpper"].isin([p.upper() for p in MAJOR_PARTIES])].copy()
        grp = d_party.groupby(["District", "PartyUpper"], dropna=False)["Age"]
        denom = grp.apply(lambda s: s.notna().sum()).rename("N")
        u40 = grp.apply(lambda s: (s < 40).sum()).rename("<40")
        u50 = grp.apply(lambda s: (s < 50).sum()).rename("<50")
        age_df = pd.concat([denom, u40, u50], axis=1).reset_index()
        for col, out in [("<40", "% <40"), ("<50", "% <50")]:
            age_df[out] = np.where(age_df["N"] > 0, age_df[col] / age_df["N"] * 100, 0.0)
        age_df = age_df[["District", "PartyUpper", "% <40", "% <50"]]
        wide = age_df.pivot_table(index="District", columns="PartyUpper", values=["% <40", "% <50"], aggfunc="first")
        wide.columns = [f"{party} {metric}" for metric, party in wide.columns]
        expected_cols = []
        for p in MAJOR_PARTIES:
            expected_cols.append(f"{p} % <40")
            expected_cols.append(f"{p} % <50")
        for col in expected_cols:
            if col not in wide.columns:
                wide[col] = 0.0
        wide = wide[sorted(wide.columns)]
        age_out = wide.reset_index()
        denom_all = d_party.groupby(["PartyUpper"], dropna=False)["Age"].apply(lambda s: s.notna().sum())
        u40_all = d_party.groupby(["PartyUpper"], dropna=False)["Age"].apply(lambda s: (pd.to_numeric(s, errors='coerce') < 40).sum())
        u50_all = d_party.groupby(["PartyUpper"], dropna=False)["Age"].apply(lambda s: (pd.to_numeric(s, errors='coerce') < 50).sum())
        all_row = {"District": "All Kerala"}
        for p in MAJOR_PARTIES:
            pu = p.upper()
            Np = int(denom_all.get(pu, 0))
            all_row[f"{p} % <40"] = (float(u40_all.get(pu, 0)) / Np * 100) if Np > 0 else 0.0
            all_row[f"{p} % <50"] = (float(u50_all.get(pu, 0)) / Np * 100) if Np > 0 else 0.0
        age_out = pd.concat([age_out.sort_values("District"), pd.DataFrame([all_row])], ignore_index=True)
        fmt = {col: "{:,.2f}%" for col in age_out.columns if col != "District"}
        render_styled_table(age_out, fmt=fmt)

    st.divider()
    st.subheader("Assembly-wise Front Vote Share (Ward-tier)")
    assembly_candidates = ["Assembly", "ACName", "AssemblyName", "Constituency"]
    assembly_col = next((c for c in assembly_candidates if c in df.columns), None)
    if assembly_col is None or "Votes" not in df_ward_all.columns or "FrontUpper" not in df_ward_all.columns:
        st.info("Need Assembly, Votes and Front columns for this table.")
    else:
        d = df_ward_all.copy()
        d["Votes"] = pd.to_numeric(d["Votes"], errors="coerce").fillna(0).astype(int)
        agg = d.groupby([assembly_col, "FrontUpper"], as_index=False)["Votes"].sum()
        total = agg.groupby(assembly_col, as_index=False)["Votes"].sum().rename(columns={"Votes": "Total Votes"})
        shares = pd.merge(agg, total, on=assembly_col, how="left")
        shares["Share %"] = np.where(shares["Total Votes"] > 0, shares["Votes"] / shares["Total Votes"] * 100, 0.0)
        wide = shares.pivot_table(index=assembly_col, columns="FrontUpper", values="Share %", aggfunc="first").fillna(0.0)
        for f in [f.upper() for f in FRONTS]:
            if f not in wide.columns:
                wide[f] = 0.0
        order_cols = [f.upper() for f in FRONTS if f.upper() in wide.columns]
        wide = wide[order_cols]
        wide = wide.reset_index().rename(columns={assembly_col: "Assembly"})
        styler = wide.style.apply(_row_color_by_max_front, axis=1).format({c: "{:,.2f}%" for c in order_cols})
        render_styled_table(styler)

