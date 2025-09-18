import io
import re
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except ModuleNotFoundError:
    HAS_REPORTLAB = False

from lib.colors import DEFAULT_BG_COLOR, FRONT_BG_COLORS, PARTY_BG_COLORS, FRONT_COLORS, PARTY_COLORS
from lib.data import data_controls, get_data_path, load_data

FRONT_ORDER = ["UDF", "LDF", "NDA", "OTH"]
FRONT_ORDER_MAP = {front: idx for idx, front in enumerate(FRONT_ORDER)}
STRENGTH_ORDER = [
    "-500 or less",
    "-200 to -499",
    "-100 to -199",
    "-50 to -99",
    "-1 to -49",
    "0",
    "1-49",
    "50-99",
    "100-199",
    "200-499",
    "500+",
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


def _safe_filename(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9_.-]", "", text)
    return text or "report"


def _front_sort_key(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: FRONT_ORDER_MAP.get(x, len(FRONT_ORDER))).astype(int)


def _rank_to_label(value: object) -> str:
    try:
        n = int(value)
    except Exception:
        return str(value)
    if n == 1:
        return "Won"
    suffix = "th"
    if not 11 <= (n % 100) <= 13:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _prepare_rank_pivot(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or "Rank" not in df.columns:
        base_cols = group_cols + ["Won", "2nd", "3rd", "Total"]
        return pd.DataFrame(columns=base_cols)

    cols = group_cols + ["Rank"]
    tmp = df[cols].copy()
    tmp["Rank"] = pd.to_numeric(tmp["Rank"], errors="coerce")
    tmp = tmp.dropna(subset=["Rank"])
    if tmp.empty:
        base_cols = group_cols + ["Won", "2nd", "3rd", "Total"]
        return pd.DataFrame(columns=base_cols)

    for col in group_cols:
        tmp[col] = tmp[col].astype(str)

    tmp["Rank"] = tmp["Rank"].astype(int)
    pivot = tmp.groupby(group_cols + ["Rank"], dropna=False).size().unstack(fill_value=0)

    rank_cols = sorted([c for c in pivot.columns if isinstance(c, int)])
    rename_map = {r: _rank_to_label(r) for r in rank_cols}
    pivot = pivot.rename(columns=rename_map)
    ordered_labels = [rename_map[r] for r in rank_cols]
    pivot["Total"] = pivot[ordered_labels].sum(axis=1)
    pivot = pivot.reset_index()

    for col in ordered_labels + ["Total"]:
        if col in pivot.columns:
            pivot[col] = pivot[col].astype(int)

    desired = group_cols + ordered_labels + ["Total"]
    for label in ["Won", "2nd", "3rd"]:
        if label not in desired:
            desired.insert(len(group_cols), label)
            pivot[label] = 0

    pivot = pivot[[c for c in desired if c in pivot.columns]]
    return pivot


def _strength_from_lead(lead: object) -> Optional[str]:
    try:
        x = float(lead)
    except Exception:
        return None
    if x <= -500:
        return "-500 or less"
    if -500 < x <= -200:
        return "-200 to -499"
    if -200 < x <= -100:
        return "-100 to -199"
    if -100 < x <= -50:
        return "-50 to -99"
    if -50 < x <= -1:
        return "-1 to -49"
    if x == 0:
        return "0"
    if 0 < x <= 49:
        return "1-49"
    if 50 <= x <= 99:
        return "50-99"
    if 100 <= x <= 199:
        return "100-199"
    if 200 <= x <= 499:
        return "200-499"
    if x >= 500:
        return "500+"
    return None


def _strength_color_from_value(value: object, invert: bool = False) -> str:
    try:
        val = float(value)
    except Exception:
        return DEFAULT_BG_COLOR
    if invert:
        val = -val
    band = _strength_from_lead(val)
    return STRENGTH_COLOR_MAP.get(band, DEFAULT_BG_COLOR)

def _resolve_ward_label(series_df: pd.DataFrame) -> pd.Series:
    label_cols = ["WardName", "Ward", "WardLabel", "WardNo", "WardCode", "BoothName"]
    for col in label_cols:
        if col in series_df.columns:
            return series_df[col].astype(str)
    return pd.Series("-", index=series_df.index)

def _build_strength_table(df: pd.DataFrame, sel_party: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Strength Band", "Ward Count", "Ward Names"])
    part = df[df["Party"].astype(str) == sel_party].copy()
    if part.empty:
        return pd.DataFrame(columns=["Strength Band", "Ward Count", "Ward Names"])

    if "Strength" in part.columns and part["Strength"].notna().any():
        strength = part["Strength"].astype(str)
    elif "Lead" in part.columns:
        strength = part["Lead"].apply(_strength_from_lead)
    else:
        return pd.DataFrame(columns=["Strength Band", "Ward Count", "Ward Names"])

    if strength.dropna().empty:
        return pd.DataFrame(columns=["Strength Band", "Ward Count", "Ward Names"])

    part = part.assign(Strength=strength)
    if "WardName" in part.columns:
        agg = (
            part.dropna(subset=["Strength"])
            .groupby("Strength", dropna=False)
            .agg(
                **{
                    "Ward Count": ("WardName", "count"),
                    "Ward Names": (
                        "WardName",
                        lambda x: ", ".join(
                            sorted({str(v).strip() for v in x if pd.notna(v) and str(v).strip()})
                        )
                        or "-",
                    ),
                }
            )
        )
    else:
        agg = (
            part.dropna(subset=["Strength"])
            .groupby("Strength", dropna=False)
            .agg(**{"Ward Count": ("Party", "count")})
        )
        agg["Ward Names"] = "-"

    agg = agg.reset_index().rename(columns={"Strength": "Strength Band"})
    agg = agg.set_index("Strength Band").reindex(STRENGTH_ORDER).reset_index()
    agg["Ward Count"] = agg["Ward Count"].fillna(0).astype(int)
    agg["Ward Names"] = agg["Ward Names"].fillna("-")
    agg = agg[agg["Ward Count"] > 0]
    if agg.empty:
        return pd.DataFrame(columns=["Strength Band", "Ward Count", "Ward Names"])
    return agg.reset_index(drop=True)

def _build_votebin_table(df: pd.DataFrame, sel_party: str) -> pd.DataFrame:
    if df.empty or "VoteBin" not in df.columns:
        return pd.DataFrame(columns=["VoteBin", "Won", "Not Won", "Total", "Won Wards", "Not Won Wards"])

    part = df[df["Party"].astype(str) == sel_party].copy()
    if part.empty:
        return pd.DataFrame(columns=["VoteBin", "Won", "Not Won", "Total", "Won Wards", "Not Won Wards"])

    part["VoteBin"] = part["VoteBin"].astype(str)
    part["Rank"] = pd.to_numeric(part.get("Rank"), errors="coerce")
    part["Status"] = np.where(part["Rank"] == 1, "Won", "Not Won")
    part["_WardLabel"] = _resolve_ward_label(part)

    pivot = (
        part.groupby(["VoteBin", "Status"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .rename(columns={"Won": "Won", "Not Won": "Not Won"})
    )

    for col in ["Won", "Not Won"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["Total"] = pivot[["Won", "Not Won"]].sum(axis=1)
    table = pivot.reset_index().sort_values("VoteBin").reset_index(drop=True)

    names = (
        part.groupby(["VoteBin", "Status"], dropna=False)["_WardLabel"]
        .apply(
            lambda x: ", ".join(
                sorted({str(v).strip() for v in x if pd.notna(v) and str(v).strip()})
            )
            or "-"
        )
        .unstack(fill_value="-")
    )
    names = names.reindex(table["VoteBin"], fill_value="-")
    table["Won Wards"] = names.get("Won", pd.Series("-", index=table.index)).astype(str)
    table["Not Won Wards"] = names.get("Not Won", pd.Series("-", index=table.index)).astype(str)

    return table


def _ward_join_keys(df: pd.DataFrame) -> list[str]:
    if "WardCode" in df.columns:
        return ["WardCode"]
    if all(c in df.columns for c in ["District", "LBName", "WardNo"]):
        return ["District", "LBName", "WardNo"]
    if all(c in df.columns for c in ["District", "LBName", "WardName"]):
        return ["District", "LBName", "WardName"]
    candidates = [c for c in ["WardName", "WardNo"] if c in df.columns]
    return candidates or [df.columns[0]]


def _build_opponent_table(df: pd.DataFrame, sel_party: str) -> pd.DataFrame:
    if df.empty or "Rank" not in df.columns:
        return pd.DataFrame(columns=["Party", "Runner-up (when Selected Won)", "Winners (when Selected Second)"])

    tmp = df.copy()
    tmp["Rank"] = pd.to_numeric(tmp["Rank"], errors="coerce").astype("Int64")
    tmp = tmp[tmp["Rank"].notna()]
    if tmp.empty:
        return pd.DataFrame(columns=["Party", "Runner-up (when Selected Won)", "Winners (when Selected Second)"])

    keys = _ward_join_keys(tmp)
    winners = tmp[tmp["Rank"] == 1][keys + ["Party"]].rename(columns={"Party": "WinnerParty"})
    runners = tmp[tmp["Rank"] == 2][keys + ["Party"]].rename(columns={"Party": "RunnerParty"})

    wins_sel = winners[winners["WinnerParty"].astype(str) == sel_party]
    ru_vs_selwin = (
        wins_sel.merge(runners, on=keys, how="left")
        .groupby("RunnerParty", dropna=True)
        .size()
        .rename("Runner-up (when Selected Won)")
    )
    ru_vs_selwin = ru_vs_selwin.rename_axis("Party").reset_index()
    ru_vs_selwin["Party"] = ru_vs_selwin["Party"].fillna("UNKNOWN").astype(str)

    sec_sel = runners[runners["RunnerParty"].astype(str) == sel_party]
    win_vs_selsec = (
        sec_sel.merge(winners, on=keys, how="left")
        .groupby("WinnerParty", dropna=True)
        .size()
        .rename("Winners (when Selected Second)")
    )
    win_vs_selsec = win_vs_selsec.rename_axis("Party").reset_index()
    win_vs_selsec["Party"] = win_vs_selsec["Party"].fillna("UNKNOWN").astype(str)

    combined = sorted(set(ru_vs_selwin["Party"].tolist()) | set(win_vs_selsec["Party"].tolist()))
    out = pd.DataFrame({"Party": combined})
    out["Party"] = out["Party"].astype(str)
    out = out.merge(ru_vs_selwin, on="Party", how="left")
    out = out.merge(win_vs_selsec, on="Party", how="left")
    out = out.fillna(0)
    for col in ["Runner-up (when Selected Won)", "Winners (when Selected Second)"]:
        out[col] = out[col].astype(int)
    out["Total"] = out[["Runner-up (when Selected Won)", "Winners (when Selected Second)"]].sum(axis=1)
    out = out.sort_values(["Total", "Party"], ascending=[False, True]).drop(columns="Total")
    return out.reset_index(drop=True)


def _build_candidate_tables(df: pd.DataFrame, sel_party: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "Rank" not in df.columns:
        empty_cols = ["Ward", "Candidate", "Votes", "Vote share (%)", "Lead"]
        empty = pd.DataFrame(columns=empty_cols)
        return empty, empty

    tmp = df.copy()
    tmp["Rank"] = pd.to_numeric(tmp["Rank"], errors="coerce").astype("Int64")
    tmp = tmp[tmp["Rank"].notna()]
    if tmp.empty:
        empty_cols = ["Ward", "Candidate", "Votes", "Vote share (%)", "Lead"]
        empty = pd.DataFrame(columns=empty_cols)
        return empty, empty

    keys = _ward_join_keys(tmp)
    totals = tmp.groupby(keys, dropna=False)["Votes"].sum().rename("TotalVotes").reset_index()

    winners = tmp[(tmp["Party"].astype(str) == sel_party) & (tmp["Rank"] == 1)].copy()
    losers = tmp[(tmp["Party"].astype(str) == sel_party) & (tmp["Rank"] != 1)].copy()

    runner_cols = keys + ["Party", "Candidate", "Votes"]
    runners = (
        tmp[tmp["Rank"] == 2][runner_cols]
        .rename(columns={"Party": "RunnerParty", "Candidate": "RunnerCandidate", "Votes": "RunnerVotes"})
    )
    winners = winners.merge(totals, on=keys, how="left").merge(runners, on=keys, how="left")
    winners["Lead"] = pd.to_numeric(winners.get("Lead"), errors="coerce")
    winners["Lead"] = winners["Lead"].fillna(
        pd.to_numeric(winners.get("Votes"), errors="coerce").fillna(0)
        - pd.to_numeric(winners.get("RunnerVotes"), errors="coerce").fillna(0)
    )
    winners["Vote share (%)"] = np.where(
        winners["TotalVotes"] > 0,
        winners["Votes"] / winners["TotalVotes"] * 100,
        0.0,
    )

    winners_tbl = pd.DataFrame({
        "Ward": winners.get("WardName", winners.get("WardNo", pd.Series(index=winners.index))).astype(str),
        "Candidate": winners.get("Candidate", pd.Series(index=winners.index)).astype(str),
        "Votes": pd.to_numeric(winners.get("Votes"), errors="coerce").fillna(0).astype(int),
        "Vote share (%)": winners["Vote share (%)"].round(2),
        "Lead": pd.to_numeric(winners.get("Lead"), errors="coerce").fillna(0).round(0).astype(int),
        "Runner Party": winners.get("RunnerParty", pd.Series("-", index=winners.index)).fillna("-"),
    })
    winners_tbl = winners_tbl.sort_values("Lead", ascending=False).reset_index(drop=True)

    winners_any = (
        tmp[tmp["Rank"] == 1][runner_cols]
        .rename(columns={"Party": "WinnerParty", "Candidate": "WinnerCandidate", "Votes": "WinnerVotes"})
    )
    losers = losers.merge(totals, on=keys, how="left").merge(winners_any, on=keys, how="left")
    losers["Trail"] = (
        pd.to_numeric(losers.get("WinnerVotes"), errors="coerce").fillna(0)
        - pd.to_numeric(losers.get("Votes"), errors="coerce").fillna(0)
    )
    losers["Vote share (%)"] = np.where(
        losers["TotalVotes"] > 0,
        losers["Votes"] / losers["TotalVotes"] * 100,
        0.0,
    )

    losers_tbl = pd.DataFrame({
        "Ward": losers.get("WardName", losers.get("WardNo", pd.Series(index=losers.index))).astype(str),
        "Candidate": losers.get("Candidate", pd.Series(index=losers.index)).astype(str),
        "Votes": pd.to_numeric(losers.get("Votes"), errors="coerce").fillna(0).astype(int),
        "Vote share (%)": losers["Vote share (%)"].round(2),
        "Trail": pd.to_numeric(losers.get("Trail"), errors="coerce").fillna(0).round(0).astype(int),
        "Winner Party": losers.get("WinnerParty", pd.Series("-", index=losers.index)).fillna("-"),
        "Position": pd.to_numeric(losers.get("Rank"), errors="coerce").astype("Int64"),
    })
    losers_tbl = losers_tbl.sort_values("Trail", ascending=False).reset_index(drop=True)

    return winners_tbl, losers_tbl

def _format_cell(value: object) -> str:
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        return f"{float(value):,.2f}"
    text = str(value)
    if text.strip() == "" or text.strip().lower() in {"nan", "none"}:
        return "-"
    return text


def _build_summary_lines(df: pd.DataFrame, sel_front: str, sel_party: str) -> List[str]:
    """Build HTML-ready summary lines for the selected scope."""
    lines: List[str] = ["Summary"]

    if df.empty:
        lines.append("No ward-level data available for this selection.")
        lines.append("")
        sel_color = PARTY_COLORS.get(sel_party, "#1f2937")
        lines.append(
            f'<font color="{sel_color}"><b>{sel_party}</b></font> secured 0 votes (0.00%) out of 0 total votes.'
        )
        lines.append("Won 0 seats out of 0 contested.")
        lines.append("Total seats: 0 | Majority mark: 0")
        return lines

    rank_series = pd.to_numeric(df.get("Rank"), errors="coerce")
    winners = df[rank_series == 1].copy()
    total_seats = int(len(winners))
    majority_mark = (total_seats // 2) + 1 if total_seats else 0

    if not winners.empty:
        winners["Front"] = winners.get("Front", "").astype(str)
        winners["Party"] = winners.get("Party", "").astype(str)
        front_counts = (
            winners.groupby("Front", dropna=False)
            .size()
            .rename("Wins")
            .astype(int)
            .loc[lambda s: s > 0]
        )
        if not front_counts.empty:
            sorted_fronts = list(front_counts.index)
            sorted_fronts.sort(
                key=lambda front: (
                    -front_counts[front],
                    FRONT_ORDER_MAP.get(str(front), len(FRONT_ORDER)),
                    str(front),
                )
            )
            for front in sorted_fronts:
                front_hex = FRONT_COLORS.get(str(front), "#1f2937")
                front_label = f'<font color="{front_hex}"><b>{front}</b></font>'
                party_counts = (
                    winners[winners["Front"] == front]
                    .groupby("Party", dropna=False)
                    .size()
                    .astype(int)
                    .loc[lambda s: s > 0]
                    .sort_values(ascending=False, kind="mergesort")
                )
                party_bits = []
                for party, count in party_counts.items():
                    party_hex = PARTY_COLORS.get(str(party), "#1f2937")
                    party_bits.append(f'<font color="{party_hex}">{party}</font> ({count})')
                party_text = ", ".join(party_bits) or "-"
                lines.append(f"{front_label} ({front_counts[front]}): {party_text}")
        else:
            lines.append("No wards won within this scope.")
    else:
        lines.append("No wards won within this scope.")

    ward_party = df[df.get("Party", "").astype(str) == sel_party].copy()
    total_votes = int(pd.to_numeric(df.get("Votes"), errors="coerce").fillna(0).sum())
    party_votes = int(pd.to_numeric(ward_party.get("Votes"), errors="coerce").fillna(0).sum())
    share = (party_votes / total_votes * 100) if total_votes > 0 else 0.0
    contested = int(len(ward_party))
    won = int((pd.to_numeric(ward_party.get("Rank"), errors="coerce") == 1).sum())

    sel_color = PARTY_COLORS.get(sel_party, "#1f2937")
    lines.append("")
    lines.append(
        f'<font color="{sel_color}"><b>{sel_party}</b></font> secured {party_votes:,} votes ({share:.2f}%) out of {total_votes:,} total votes.'
    )
    lines.append(f"Won {won} seats out of {contested} contested.")
    lines.append(f"Total seats: {total_seats:,} | Majority mark: {majority_mark:,}")
    return lines


def _build_pdf_document(
    title: str,
    summary_lines: Sequence[str],
    sections: Sequence[Tuple[str, pd.DataFrame, Optional[List[str]]]],
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    header_paragraph = ParagraphStyle(
        "TableHeader",
        parent=styles["Heading5"],
        alignment=1,
        fontSize=10,
        leading=12,
    )
    body_paragraph = ParagraphStyle(
        "TableBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        wordWrap="LTR",
    )

    elems: List = []
    elems.append(Paragraph(title, styles["Title"]))
    elems.append(Paragraph(datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S"), styles["BodyText"]))
    elems.append(Spacer(1, 8))

    if summary_lines:
        elems.append(Paragraph(summary_lines[0], styles["Heading3"]))
        for line in summary_lines[1:]:
            if not line.strip():
                elems.append(Spacer(1, 4))
            else:
                elems.append(Paragraph(line, styles["BodyText"]))
        elems.append(Spacer(1, 8))

    header_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ]
    )
    body_style = TableStyle(
        [
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ]
    )

    for sec_title, df, row_colors in sections:
        elems.append(Spacer(1, 10))
        elems.append(Paragraph(sec_title, styles["Heading3"]))
        if df is None or df.empty:
            elems.append(Paragraph("No data available.", styles["BodyText"]))
            continue

        table_df = df.reset_index(drop=True)
        header = [Paragraph(str(col), header_paragraph) for col in table_df.columns]
        data = [header]
        for _, row in table_df.iterrows():
            cells = [Paragraph(_format_cell(value), body_paragraph) for value in row.tolist()]
            data.append(cells)
        col_width_map = {
            "Strength Band": 90,
            "Ward Count": 65,
            "VoteBin": 80,
            "Won": 55,
            "Not Won": 65,
            "Total": 65,
        }
        col_widths = [col_width_map.get(str(col), None) for col in table_df.columns]
        tbl = Table(data, repeatRows=1, colWidths=col_widths)
        commands = list(header_style.getCommands()) + list(body_style.getCommands())
        if row_colors:
            for row_idx, color_hex in enumerate(row_colors, start=1):
                if color_hex:
                    try:
                        commands.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor(color_hex)))
                    except Exception:
                        commands.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor(DEFAULT_BG_COLOR)))
        tbl.setStyle(TableStyle(commands))
        elems.append(tbl)

    doc.build(elems)
    return buf.getvalue()


def _party_options_for_front(df: pd.DataFrame, front: str) -> List[str]:
    preferred = {
        "UDF": ["IUML", "INC"],
        "LDF": ["CPI(M)", "CPI"],
        "NDA": ["BJP"],
        "OTH": ["IND", "SDPI", "WPI"],
    }

    parties = df[df.get("Front", "").astype(str) == front]["Party"].dropna().astype(str).unique().tolist()
    parties = sorted(set(parties))

    ordered: List[str] = []
    seen: set[str] = set()
    for p in preferred.get(front, []):
        if not parties or p in parties:
            if p not in seen:
                ordered.append(p)
                seen.add(p)
    for p in parties:
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered or parties


def _filter_scope(df: pd.DataFrame, report_type: str, scope: dict) -> pd.DataFrame:
    d = df.copy()
    if report_type == "District":
        district = scope.get("District")
        if district and district not in {"All", "All Kerala"}:
            d = d[d.get("District", "").astype(str) == str(district)]
    elif report_type == "Assembly":
        district = scope.get("District")
        if district and district != "All":
            d = d[d.get("District", "").astype(str) == str(district)]
        assembly = scope.get("Assembly")
        if assembly:
            asm_candidates = ["Assembly", "ACName", "AssemblyName", "Constituency"]
            asm_col = next((c for c in asm_candidates if c in d.columns), None)
            if asm_col:
                d = d[d[asm_col].astype(str) == str(assembly)]
    else:
        district = scope.get("District")
        if district:
            d = d[d.get("District", "").astype(str) == str(district)]
        lb_type = scope.get("LBType")
        if lb_type:
            d = d[d.get("LBType", "").astype(str) == str(lb_type)]
        lb_name = scope.get("LBName")
        if lb_name:
            d = d[d.get("LBName", "").astype(str) == str(lb_name)]
    return d


def _generate_scope_sections(df: pd.DataFrame, sel_front: str, sel_party: str) -> Tuple[List[str], List[Tuple[str, pd.DataFrame, Optional[List[str]]]]]:
    ward = df[df.get("TierNorm", df.get("Tier", "")).astype(str).str.title() == "Ward"].copy()
    if ward.empty:
        ward = df.copy()

    summary_lines = _build_summary_lines(ward, sel_front, sel_party)
    sections: List[Tuple[str, pd.DataFrame, Optional[List[str]]]] = []

    vote_series = pd.to_numeric(ward.get("Votes"), errors="coerce").fillna(0)
    total_votes_scope = float(vote_series.sum())

    front_summary = _prepare_rank_pivot(ward, ["Front"])
    if not front_summary.empty:
        if "Won" in front_summary.columns:
            front_summary = front_summary.sort_values("Won", ascending=False).reset_index(drop=True)
        front_votes = ward.groupby("Front", dropna=False)["Votes"].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()).to_dict()
        front_summary["Total Votes"] = front_summary["Front"].map(lambda x: int(front_votes.get(str(x), 0)))
        front_summary["Vote share (%)"] = front_summary["Total Votes"].apply(lambda v: (v / total_votes_scope * 100) if total_votes_scope > 0 else 0.0).round(2)
        ordered_cols = ["Front", "Total Votes", "Vote share (%)"] + [c for c in front_summary.columns if c not in {"Front", "Total Votes", "Vote share (%)"}]
        front_summary = front_summary[ordered_cols]
        front_colors = [FRONT_BG_COLORS.get(str(row.get("Front", "")), DEFAULT_BG_COLOR) for _, row in front_summary.iterrows()]
        sections.append(("Front Performance", front_summary, front_colors))

    party_summary = _prepare_rank_pivot(ward, ["Party", "Front"])
    if not party_summary.empty:
        party_summary["Front"] = pd.Categorical(party_summary["Front"].astype(str), categories=FRONT_ORDER, ordered=True)
        party_summary = party_summary.sort_values(["Front", "Party"]).reset_index(drop=True)
        party_summary["Front"] = party_summary["Front"].astype(str)
        votes_by_pair = ward.groupby(["Front", "Party"], dropna=False)["Votes"].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()).to_dict()
        party_summary["Total Votes"] = [int(votes_by_pair.get((row["Front"], row["Party"]), 0)) for _, row in party_summary.iterrows()]
        party_summary["Vote share (%)"] = [
            (value / total_votes_scope * 100) if total_votes_scope > 0 else 0.0
            for value in party_summary["Total Votes"].astype(float)
        ]
        party_summary["Vote share (%)"] = party_summary["Vote share (%)"].round(2)
        metrics = [c for c in party_summary.columns if c not in {"Front", "Party", "Total Votes", "Vote share (%)"}]
        party_summary = party_summary[["Front", "Party", "Total Votes", "Vote share (%)", *metrics]]
        party_colors = [PARTY_BG_COLORS.get(str(row.get("Party", "")), DEFAULT_BG_COLOR) for _, row in party_summary.iterrows()]
        sections.append(("Party Performance", party_summary, party_colors))

    party_scope = ward[(ward.get("Party", "").astype(str) == sel_party) & (ward.get("Front", "").astype(str) == sel_front)].copy()
    if party_scope.empty:
        party_scope = ward[ward.get("Party", "").astype(str) == sel_party].copy()

    strength_table = _build_strength_table(party_scope, sel_party)
    if not strength_table.empty:
        strength_colors = [STRENGTH_COLOR_MAP.get(str(row.get("Strength Band", "")), DEFAULT_BG_COLOR) for _, row in strength_table.iterrows()]
        sections.append((f"{sel_party} - Lead Strength", strength_table, strength_colors))

    vote_bin = _build_votebin_table(party_scope, sel_party)
    if not vote_bin.empty:
        party_color = PARTY_BG_COLORS.get(sel_party, DEFAULT_BG_COLOR)
        vote_colors = [party_color for _ in range(len(vote_bin))]
        sections.append((f"{sel_party} - Vote Share Strength", vote_bin, vote_colors))

    opponent_table = _build_opponent_table(ward, sel_party)
    if not opponent_table.empty:
        opponent_table = opponent_table.rename(
            columns={
                "Runner-up (when Selected Won)": f"Runner-up when {sel_party} Won",
                "Winners (when Selected Second)": f"Winners when {sel_party} became second",
            }
        )
        opp_colors = [PARTY_BG_COLORS.get(str(row.get("Party", "")), DEFAULT_BG_COLOR) for _, row in opponent_table.iterrows()]
        sections.append((f"Opponent breakdown - {sel_party}", opponent_table, opp_colors))

    winners_tbl, losers_tbl = _build_candidate_tables(ward, sel_party)
    if not winners_tbl.empty:
        win_colors = [_strength_color_from_value(row.get("Lead")) for _, row in winners_tbl.iterrows()]
        sections.append((f"Winning candidates - {sel_party}", winners_tbl, win_colors))
    if not losers_tbl.empty:
        lose_colors = [_strength_color_from_value(row.get("Trail"), invert=True) for _, row in losers_tbl.iterrows()]
        sections.append((f"Losing candidates - {sel_party}", losers_tbl, lose_colors))

    return summary_lines, sections

def _generate_sections(df: pd.DataFrame, report_type: str, sel_front: str, sel_party: str) -> Tuple[List[str], List[Tuple[str, pd.DataFrame, Optional[List[str]]]]]:
    return _generate_scope_sections(df, sel_front, sel_party)


def main() -> None:
    st.set_page_config(page_title="Download Reports", page_icon="📄", layout="wide")
    st.title("Download Reports")
    if not HAS_REPORTLAB:
        st.warning("PDF generation requires the 'reportlab' package. Install it via 'pip install reportlab'.")

    data_controls()
    df = load_data(get_data_path()).copy()
    df["TierNorm"] = df.get("TierNorm", df.get("Tier", "")).astype(str).str.title()

    fronts_present = [f for f in FRONT_ORDER if f in df.get("Front", pd.Series(dtype=str)).astype(str).unique().tolist()]
    if not fronts_present:
        fronts_present = sorted(df.get("Front", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) or FRONT_ORDER

    c1, c2 = st.columns(2)
    with c1:
        sel_front = st.selectbox("Front", fronts_present, index=0)
    with c2:
        party_choices = _party_options_for_front(df, sel_front)
        sel_party = st.selectbox("Party", party_choices, index=0 if party_choices else None)

    st.markdown("---")

    report_type = st.radio("Report Type", ["District", "Assembly", "Local Body"], horizontal=True)
    scope: dict = {}

    if report_type == "District":
        districts = ["All Kerala"] + sorted(df.get("District", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        scope["District"] = st.selectbox("District", districts, index=0)
    elif report_type == "Assembly":
        districts = ["All"] + sorted(df.get("District", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        sel_d = st.selectbox("District", districts, index=0)
        scope["District"] = sel_d
        asm_candidates = ["Assembly", "ACName", "AssemblyName", "Constituency"]
        asm_col = next((c for c in asm_candidates if c in df.columns), None)
        dfx = df if sel_d == "All" else df[df.get("District", "").astype(str) == str(sel_d)]
        assemblies = sorted(dfx.get(asm_col, pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) if asm_col else []
        scope["Assembly"] = st.selectbox("Assembly", assemblies, index=0 if assemblies else None)
    else:
        districts = sorted(df.get("District", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        sel_d = st.selectbox("District", districts, index=0)
        scope["District"] = sel_d
        dfx = df[df.get("District", "").astype(str) == str(sel_d)]
        lb_types = sorted(dfx.get("LBType", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        sel_lb_type = st.selectbox("Local Body Type", lb_types, index=0 if lb_types else None)
        scope["LBType"] = sel_lb_type
        dfx2 = dfx[dfx.get("LBType", "").astype(str) == str(sel_lb_type)] if sel_lb_type else dfx
        lb_names = sorted(dfx2.get("LBName", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        scope["LBName"] = st.selectbox("Local Body", lb_names, index=0 if lb_names else None)

    scoped_df = _filter_scope(df, report_type, scope)
    summary_lines, sections = _generate_sections(scoped_df, report_type, sel_front, sel_party)

    st.markdown("---")
    st.subheader("Download")

    if not HAS_REPORTLAB:
        st.error("Cannot generate PDF because 'reportlab' is not installed.")
        return

    if not sections:
        st.caption("No data to export for the current selection.")
        return

    title_bits = ["LSGD Report", report_type, sel_front, sel_party]
    for key in ["District", "Assembly", "LBType", "LBName"]:
        if scope.get(key):
            title_bits.append(str(scope[key]))
    title = " - ".join(filter(None, title_bits))

    pdf_bytes = _build_pdf_document(title, summary_lines, sections)
    file_name = _safe_filename("_".join(filter(None, title_bits)) + ".pdf")

    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
    )


main()
