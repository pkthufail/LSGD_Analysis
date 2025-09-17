import io
import re
from datetime import datetime
import streamlit as st
import pandas as pd
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_REPORTLAB = True
except ModuleNotFoundError:
    HAS_REPORTLAB = False

from lib.data import load_data, get_data_path, data_controls


st.set_page_config(page_title="Download Reports · LSGD Explorer", page_icon="📄", layout="wide")
st.title("Download Reports")
st.warning("This page is under construction. Some features may be incomplete.")
if not HAS_REPORTLAB:
    st.info("PDF generation requires the 'reportlab' package. Install it with 'pip install -r requirements.txt' or 'pip install reportlab'.")

# Sidebar data controls
data_controls()
df = load_data(get_data_path()).copy()
for c in df.columns:
    if pd.api.types.is_string_dtype(df[c]):
        df[c] = df[c].astype(str).str.strip()

FRONT_ORDER = ["UDF", "LDF", "NDA", "OTH"]


def party_options_for_front(front: str) -> list[str]:
    pref = {"UDF": ["IUML", "INC"], "LDF": ["CPI(M)", "CPI"], "NDA": ["BJP"], "OTH": ["IND", "SDPI", "WPI"]}
    if {"Front", "Party"}.issubset(df.columns):
        parties = df[df["Front"].astype(str) == front]["Party"].dropna().astype(str).sort_values().unique().tolist()
    else:
        parties = sorted(df.get("Party", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    base = pref.get(front, [])
    seen, ordered = set(), []
    for p in base:
        if not parties or p in parties:
            if p not in seen:
                ordered.append(p)
                seen.add(p)
    for p in parties:
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered or base or parties


def _safe_filename(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9_.-]", "", text)
    return text or "report"


def _build_pdf_document(title: str, sections: list[tuple[str, pd.DataFrame]]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    elems: list = []

    elems.append(Paragraph(title, styles["Title"]))
    elems.append(Paragraph(datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S"), styles["BodyText"]))
    elems.append(Spacer(1, 8))

    header_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ])
    body_style = TableStyle([
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ])

    for sec_title, df in sections:
        elems.append(Spacer(1, 10))
        elems.append(Paragraph(sec_title, styles["Heading3"]))
        if df is None or df.empty:
            elems.append(Paragraph("No data available.", styles["BodyText"]))
            continue
        header = [df.index.name or ""] + [str(c) for c in df.columns]
        data = [header]
        rows = df.reset_index().values.tolist()
        formatted_rows = []
        for r in rows:
            fr = []
            for v in r:
                if isinstance(v, (int,)):
                    fr.append(f"{v:,}")
                elif isinstance(v, float):
                    fr.append(f"{v:,.2f}")
                else:
                    fr.append("" if v is None else str(v))
            formatted_rows.append(fr)
        data.extend(formatted_rows)
        tbl = Table(data, repeatRows=1)
        # Combine style command lists correctly
        ts_cmds = list(header_style.getCommands()) + list(body_style.getCommands())
        tbl.setStyle(TableStyle(ts_cmds))
        elems.append(tbl)

    doc.build(elems)
    return buf.getvalue()


# UI controls
st.header("Party Reports")
c1, c2 = st.columns(2)
with c1:
    fronts = [f for f in FRONT_ORDER if ("Front" not in df.columns) or (f in set(df.get("Front", [])))]
    if not fronts:
        fronts = sorted(df.get("Front", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) or FRONT_ORDER
    sel_front = st.selectbox("Front", fronts, index=0)
with c2:
    party_choices = party_options_for_front(sel_front)
    sel_party = st.selectbox("Party", party_choices, index=0 if party_choices else None)

st.markdown("---")
report_type = st.radio("Report Type", ["District", "Assembly", "Local Body"], horizontal=True)

scope = {}
if report_type == "District":
    districts = ["All Kerala"] + sorted(df.get("District", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(), key=lambda x: str(x))
    scope["District"] = st.selectbox("District", districts, index=0)
elif report_type == "Assembly":
    districts = ["All"] + sorted(df.get("District", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(), key=lambda x: str(x))
    sel_d = st.selectbox("District", districts, index=0, key="dl_ac_district")
    asm_candidates = ["Assembly", "ACName", "AssemblyName", "Constituency"]
    asm_col = next((c for c in asm_candidates if c in df.columns), None)
    dfx = df if sel_d == "All" else df[df["District"].astype(str) == str(sel_d)]
    asm_options = sorted(dfx[asm_col].dropna().astype(str).unique().tolist()) if asm_col else []
    scope["District"] = sel_d
    scope["Assembly"] = st.selectbox("Assembly", asm_options, index=0 if asm_options else None)
else:
    districts = sorted(df.get("District", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(), key=lambda x: str(x))
    sel_d = st.selectbox("District", districts, index=0, key="dl_lb_district")
    dfx = df[df["District"].astype(str) == str(sel_d)] if sel_d else df
    lbtypes = sorted(dfx.get("LBType", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(), key=lambda x: str(x))
    sel_lbt = st.selectbox("LB Type", lbtypes, index=0 if lbtypes else None)
    dfx2 = dfx[dfx["LBType"].astype(str) == str(sel_lbt)] if sel_lbt else dfx
    lbnames = sorted(dfx2.get("LBName", pd.Series(dtype=str)).dropna().astype(str).unique().tolist(), key=lambda x: str(x))
    sel_lbn = st.selectbox("LB Name", lbnames, index=0 if lbnames else None)
    scope.update({"District": sel_d, "LBType": sel_lbt, "LBName": sel_lbn})

# Build sections (for now: detailed tables for District = All Kerala)
sections: list[tuple[str, pd.DataFrame]] = []
dfw = df.copy()
if "Tier" in dfw.columns and "TierNorm" not in dfw.columns:
    dfw["TierNorm"] = dfw["Tier"].astype(str).str.title()
dfw["Votes"] = pd.to_numeric(dfw.get("Votes", 0), errors="coerce").fillna(0).astype(int)
dfw["Rank"] = pd.to_numeric(dfw.get("Rank", None), errors="coerce")

if report_type == "District" and scope.get("District") == "All Kerala":
    ward = dfw[dfw.get("TierNorm", dfw.get("Tier", "")).astype(str).str.title() == "Ward"].copy()
    part_rows = ward[ward.get("Party", "").astype(str) == str(sel_party)].copy()

    # 1) Seats Won by District × LBType
    wins = part_rows[part_rows["Rank"] == 1]
    if not wins.empty:
        pivot1 = wins.groupby(["District", "LBType"], dropna=False).size().rename("SeatsWon").reset_index()
        table1 = pivot1.pivot_table(index="District", columns="LBType", values="SeatsWon", aggfunc="sum", fill_value=0)
        total_row = table1.sum(axis=0).rename("All Kerala")
        table1 = pd.concat([table1.sort_index(), pd.DataFrame([total_row])])
        sections.append(("Seats Won by District × LBType", table1.fillna(0).astype(int)))

    # 2) Strike rate & Positions
    if not part_rows.empty:
        grp = part_rows.groupby(["District", "Rank"], dropna=False).size().rename("N").reset_index()
        xt = grp.pivot_table(index="District", columns="Rank", values="N", aggfunc="sum", fill_value=0)
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
        rank_numeric = sorted([int(c) for c in xt.columns if isinstance(c, (int, float))])
        xt = xt.rename(columns={c: _rank_to_label(c) for c in xt.columns})
        label_cols = [_rank_to_label(n) for n in rank_numeric]
        won_col = xt.get("Won", pd.Series(0, index=xt.index))
        xt["Contested"] = xt[[c for c in label_cols if c in xt.columns]].sum(axis=1)
        xt["Strike Rate (%)"] = (won_col / xt["Contested"]).replace([pd.NA, pd.NaT], 0).fillna(0) * 100
        total = xt.sum(numeric_only=True).to_dict()
        xt.loc["All Kerala", :] = xt.sum(numeric_only=True)
        xt.loc["All Kerala", "Strike Rate (%)"] = (total.get("Won", 0) / total.get("Contested", 1) * 100) if total.get("Contested", 0) else 0.0
        ordered = (["Won"] if "Won" in xt.columns else []) + [c for c in label_cols if c != "Won" and c in xt.columns] + ["Contested", "Strike Rate (%)"]
        xt = xt[ordered]
        # Format percentage column with % sign for nicer PDF
        xt["Strike Rate (%)"] = xt["Strike Rate (%)"].map(lambda x: f"{float(x):.2f}%")
        sections.append(("Strike Rate and Positions by District", xt))

    # 3) Votes by District
    ward_votes = dfw[dfw.get("TierNorm", dfw.get("Tier", "")).astype(str).str.title() == "Ward"]
    tot_v = ward_votes.groupby("District", dropna=False)["Votes"].sum().rename("TotalVotes") if not ward_votes.empty else pd.Series(dtype=int)
    pv = ward_votes[ward_votes.get("Party", "").astype(str) == str(sel_party)].groupby("District", dropna=False)["Votes"].sum().rename("PartyVotes") if not ward_votes.empty else pd.Series(dtype=int)
    if not ward_votes.empty and "Front" in ward_votes.columns:
        fv = ward_votes[ward_votes.get("Front", "").astype(str) == str(sel_front)].groupby("District", dropna=False)["Votes"].sum().rename("FrontVotes")
    else:
        fv = pd.Series(0, index=tot_v.index, name="FrontVotes")
    if not tot_v.empty:
        votes_df = pd.concat([pv, tot_v, fv], axis=1).fillna(0)
        votes_df["% of Total"] = (votes_df["PartyVotes"] / votes_df["TotalVotes"]).replace([pd.NA, pd.NaT], 0).fillna(0) * 100
        votes_df["% of Front"] = (votes_df["PartyVotes"] / votes_df["FrontVotes"]).replace([pd.NA, pd.NaT], 0).fillna(0) * 100
        total_row = pd.Series({
            "PartyVotes": int(votes_df["PartyVotes"].sum()),
            "TotalVotes": int(votes_df["TotalVotes"].sum()),
            "FrontVotes": int(votes_df["FrontVotes"].sum()),
        }, name="All Kerala")
        votes_df = votes_df.sort_index()
        votes_df.loc["All Kerala", ["PartyVotes", "TotalVotes", "FrontVotes"]] = total_row[["PartyVotes", "TotalVotes", "FrontVotes"]]
        votes_df.loc["All Kerala", "% of Total"] = (total_row["PartyVotes"] / total_row["TotalVotes"] * 100) if total_row["TotalVotes"] else 0.0
        votes_df.loc["All Kerala", "% of Front"] = (total_row["PartyVotes"] / total_row["FrontVotes"] * 100) if total_row["FrontVotes"] else 0.0
        # Format percentage columns
        for col in ["% of Total", "% of Front"]:
            votes_df[col] = votes_df[col].map(lambda x: f"{float(x):.2f}%")
        sections.append(("Votes by District", votes_df[["PartyVotes", "TotalVotes", "% of Total", "% of Front"]]))

    # 4) District × Strength category
    strength_series = None
    if "Strength" in part_rows.columns and part_rows["Strength"].notna().any():
        strength_series = part_rows["Strength"].astype(str)
    elif "Lead" in part_rows.columns:
        def _lead_to_strength_local(lead):
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
        strength_series = part_rows["Lead"].apply(_lead_to_strength_local)

    if strength_series is not None:
        s_df = pd.DataFrame({"District": part_rows["District"].astype(str), "Strength": strength_series}).dropna()
        order = ["-500 or less", "-200 to -499", "-100 to -199", "-50 to -99", "-1 to -49", "0", "1-49", "50-99", "100-199", "200-499", "500+"]
        s_pivot = s_df.groupby(["District", "Strength"]).size().rename("N").reset_index().pivot_table(index="District", columns="Strength", values="N", aggfunc="sum", fill_value=0)
        s_pivot = s_pivot.reindex(columns=order, fill_value=0)
        s_total = s_pivot.sum(axis=0).rename("All Kerala")
        s_pivot = pd.concat([s_pivot.sort_index(), pd.DataFrame([s_total])])
        sections.append(("District × Strength Category (wards)", s_pivot))

    # 5) District × VoteBin
    if "VoteBin" in part_rows.columns and part_rows["VoteBin"].notna().any():
        vb = part_rows[["District", "VoteBin"]].dropna()
        vb_pivot = vb.groupby(["District", "VoteBin"]).size().rename("N").reset_index().pivot_table(index="District", columns="VoteBin", values="N", aggfunc="sum", fill_value=0)
        vb_total = vb_pivot.sum(axis=0).rename("All Kerala")
        vb_pivot = pd.concat([vb_pivot.sort_index(), pd.DataFrame([vb_total])])
        sections.append(("District × VoteBin (wards)", vb_pivot))

    # 6) Major opponents by District (Top 3)
    keys = None
    if "WardCode" in ward.columns:
        keys = ["WardCode"]
    else:
        cand_sets = [["District", "LBName", "WardNo"], ["District", "LBName", "WardName"]]
        for ks in cand_sets:
            if all(k in ward.columns for k in ks):
                keys = ks
                break
        if keys is None:
            keys = [c for c in ["District", "LBName", "WardName"] if c in ward.columns]

    winners = ward[ward["Rank"] == 1][keys + ["District", "Party"]].rename(columns={"Party": "WinnerParty"})
    runners = ward[ward["Rank"] == 2][keys + ["District", "Party"]].rename(columns={"Party": "RunnerParty"})
    pair = pd.merge(winners, runners, on=keys + ["District"], how="left")

    won_sel = pair[pair["WinnerParty"].astype(str) == str(sel_party)]
    top_ru = won_sel.groupby(["District", "RunnerParty"], dropna=False).size().rename("N").reset_index().sort_values(["District", "N"], ascending=[True, False])
    sec_sel = pair[pair["RunnerParty"].astype(str) == str(sel_party)]
    top_win = sec_sel.groupby(["District", "WinnerParty"], dropna=False).size().rename("N").reset_index().sort_values(["District", "N"], ascending=[True, False])

    opp_rows = []
    for dist in sorted(dfw.get("District", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()):
        ru_rows = top_ru[top_ru["District"].astype(str) == dist].head(3)
        win_rows = top_win[top_win["District"].astype(str) == dist].head(3)
        desc = ", ".join([f"{str(p)} {int(n)}" for p, n in zip(ru_rows.get("RunnerParty", []), ru_rows.get("N", []))]) if not ru_rows.empty else "-"
        desc2 = ", ".join([f"{str(p)} {int(n)}" for p, n in zip(win_rows.get("WinnerParty", []), win_rows.get("N", []))]) if not win_rows.empty else "-"
        opp_rows.append({"District": dist, "When Won": desc, "When Second": desc2})
    if opp_rows:
        opp_df = pd.DataFrame(opp_rows).set_index("District")
        sections.append(("Major Opponents by District (Top 3)", opp_df))

st.markdown("---")
st.subheader("Generate PDF")
title = f"LSGD Report — {sel_party} ({sel_front}) — {report_type}"
if HAS_REPORTLAB:
    pdf_bytes = _build_pdf_document(title, sections)

    fname_bits = ["party", report_type.replace(" ", "").lower(), sel_front, sel_party]
    for k in ["District", "Assembly", "LBType", "LBName"]:
        if k in scope and scope[k]:
            fname_bits.append(str(scope[k]))
    file_name = _safe_filename("_".join(fname_bits) + ".pdf")
    st.download_button(label="Download PDF", data=pdf_bytes, file_name=file_name, mime="application/pdf")
else:
    st.error("Cannot generate PDF because 'reportlab' is not installed.")
