from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import html
import re

from lib.colors import FRONT_BG_COLORS, FRONT_COLORS, PARTY_BG_COLORS, PARTY_COLORS, DEFAULT_BG_COLOR

WINNING_WARD_COLOR = "#2e7d32"
LOSING_WARD_COLOR = "#c62828"
FRONT_ORDER = ["UDF", "LDF", "NDA", "OTH"]
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

@dataclass
class TableResult:
    frame: pd.DataFrame
    row_colors: Optional[List[str]] = None

def _rank_suffix(n: int) -> str:
    suffix = "th"
    if not 11 <= (n % 100) <= 13:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return suffix


def _ensure_rank_numeric(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "Rank" in data.columns:
        data["Rank"] = pd.to_numeric(data["Rank"], errors="coerce").astype("Int64")
    return data


def _resolve_ward_label(df: pd.DataFrame) -> pd.Series:
    label_cols = ["WardName", "Ward", "WardLabel", "WardNo", "WardCode", "BoothName"]
    for col in label_cols:
        if col in df.columns:
            return df[col].astype(str)
    return pd.Series("-", index=df.index)


def _format_lb_grouped_html(pairs: Iterable[Tuple[str, str]], color: str, bold_lb: bool = True) -> str:
    grouped: Dict[str, List[str]] = {}
    for lb, ward in pairs:
        if not str(lb).strip() or not str(ward).strip():
            continue
        grouped.setdefault(str(lb), []).append(str(ward).strip())
    if not grouped:
        return "-"
    pieces: List[str] = []
    for lb_name in sorted(grouped):
        wards = sorted({w for w in grouped[lb_name] if w})
        if not wards:
            continue
        ward_markup = ", ".join(f'<font color="{color}">{html.escape(w)}</font>' for w in wards)
        lb_markup = f"<b>{html.escape(lb_name)}</b>" if bold_lb else html.escape(lb_name)
        pieces.append(f"{lb_markup}: ({ward_markup})")
    return ", ".join(pieces) if pieces else "-"


def _vote_bin_color(label: object) -> str:
    if label is None:
        return DEFAULT_BG_COLOR
    text = str(label).strip()
    if not text:
        return DEFAULT_BG_COLOR
    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return DEFAULT_BG_COLOR
    avg = sum(numbers) / len(numbers)
    if avg < 30:
        return "#fde2e4"
    if avg < 45:
        return "#ffe5a1"
    if avg < 55:
        return "#f8f9fa"
    if avg < 70:
        return "#d4f3e4"
    if avg < 85:
        return "#bcefd4"
    return "#8bdcb3"


def _partition_scope(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    data = _ensure_rank_numeric(df)
    votes = pd.to_numeric(data.get("Votes"), errors="coerce").fillna(0)
    data["Votes"] = votes
    return data, float(votes.sum())


def summary_by_lb(df: pd.DataFrame, lb_counts: Dict[str, Dict[str, int]]) -> TableResult:
    scoped, _ = _partition_scope(df)
    if scoped.empty:
        empty = pd.DataFrame(columns=["LBName", "Wards (2020)", "Wards (2025)", "New Wards", "Total Votes"])
        return TableResult(empty)

    group = scoped.groupby(["LBCode", "LBName"], dropna=False)
    wards_2020 = group["WardCode"].nunique().rename("Wards (2020)")
    votes = group["Votes"].sum().rename("Total Votes")

    rows: List[Dict[str, object]] = []
    for (lb_code, lb_name), wards in wards_2020.items():
        lb_meta = lb_counts.get(str(lb_code), {"wards_2020": 0, "wards_2025": 0, "new_wards": 0})
        rows.append({
            "LBCode": lb_code,
            "LBName": lb_name,
            "Wards (2020)": int(wards),
            "Wards (2025)": int(lb_meta.get("wards_2025", 0)),
            "New Wards": int(lb_meta.get("new_wards", max(lb_meta.get("wards_2025", 0) - wards, 0))),
            "Total Votes": int(votes.loc[(lb_code, lb_name)]),
        })

    table = pd.DataFrame(rows).sort_values("LBName").reset_index(drop=True)
    totals = {
        "LBName": "Total",
        "Wards (2020)": table["Wards (2020)"].sum(),
        "Wards (2025)": table["Wards (2025)"].sum(),
        "New Wards": table["New Wards"].sum(),
        "Total Votes": table["Total Votes"].sum(),
    }
    table = pd.concat([table.drop(columns=["LBCode"], errors="ignore"), pd.DataFrame([totals])], ignore_index=True)
    return TableResult(table)


def _rank_table(df: pd.DataFrame, group_cols: List[str]) -> Tuple[pd.DataFrame, List[str], float]:
    scoped, total_votes = _partition_scope(df)
    if scoped.empty:
        return pd.DataFrame(columns=[*group_cols, "Won", "Contested", "Votes", "Vote share (%)", "Strike Rate (%)"]), [], 0.0

    rank_counts = (
        scoped.groupby(group_cols + ["Rank"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    rank_counts.columns = [int(c) for c in rank_counts.columns]
    df_rank = rank_counts.rename(columns=lambda r: "Won" if r == 1 else f"{int(r)}{_rank_suffix(int(r))}")
    rank_columns = [col for col in df_rank.columns if isinstance(col, str)]
    df_rank = df_rank.reset_index()

    vote_totals = scoped.groupby(group_cols, dropna=False)["Votes"].sum().reset_index(name="Votes")
    merged = df_rank.merge(vote_totals, on=group_cols, how="left")
    merged["Votes"] = merged["Votes"].fillna(0).astype(int)

    rank_only_cols = [col for col in merged.columns if any(col.endswith(suf) for suf in ["st", "nd", "rd", "th"]) or col == "Won"]
    merged["Contested"] = merged[rank_only_cols].sum(axis=1) if rank_only_cols else 0
    merged["Won"] = merged.get("Won", 0)
    merged["Vote share (%)"] = np.where(total_votes > 0, merged["Votes"] / total_votes * 100, 0.0)
    merged["Strike Rate (%)"] = np.where(merged["Contested"] > 0, merged["Won"] / merged["Contested"] * 100, 0.0)
    merged["Vote share (%)"] = merged["Vote share (%)"].round(2)
    merged["Strike Rate (%)"] = merged["Strike Rate (%)"].round(2)

    ordered_cols = group_cols + [col for col in ["Won"] + [c for c in rank_only_cols if c != "Won"]] + ["Contested", "Votes", "Vote share (%)", "Strike Rate (%)"]
    ordered_cols = [col for col in ordered_cols if col in merged.columns]
    merged = merged[ordered_cols]
    return merged, rank_only_cols, total_votes


def front_performance(df: pd.DataFrame) -> TableResult:
    table, _, total_votes = _rank_table(df, ["Front"])
    table = table.sort_values("Won", ascending=False).reset_index(drop=True)
    colors = [FRONT_BG_COLORS.get(str(row.get("Front", "")), DEFAULT_BG_COLOR) for _, row in table.iterrows()]
    return TableResult(table, colors)


def party_performance(df: pd.DataFrame) -> TableResult:
    table, _, _ = _rank_table(df, ["Party", "Front"])
    table["Front"] = pd.Categorical(table["Front"].astype(str), categories=FRONT_ORDER, ordered=True)
    table = table.sort_values(["Front", "Party"]).reset_index(drop=True)
    table["Front"] = table["Front"].astype(str)
    colors = [PARTY_BG_COLORS.get(str(row.get("Party", "")), DEFAULT_BG_COLOR) for _, row in table.iterrows()]
    return TableResult(table, colors)


def seats_by_front(df: pd.DataFrame) -> TableResult:
    scoped, _ = _partition_scope(df)
    winners = scoped[scoped["Rank"] == 1]
    if winners.empty:
        return TableResult(pd.DataFrame(columns=["LBName", *FRONT_ORDER, "Leader"]))
    matrix = winners.pivot_table(index="LBName", columns="Front", values="WardCode", aggfunc="nunique", fill_value=0)
    matrix = matrix.reindex(columns=FRONT_ORDER, fill_value=0)
    matrix["Leader"] = matrix.apply(lambda row: row.idxmax() if row.max() > 0 else "-", axis=1)
    total_row = matrix.sum(numeric_only=True)
    matrix.loc["Total"] = total_row
    matrix.loc["Total", "Leader"] = "-"
    matrix = matrix.reset_index().rename(columns={"index": "LBName"})
    colors = [FRONT_BG_COLORS.get(str(row.get("Leader", "")), DEFAULT_BG_COLOR) for _, row in matrix.iterrows()]
    return TableResult(matrix, colors)


def votes_by_front(df: pd.DataFrame) -> TableResult:
    scoped, _ = _partition_scope(df)
    matrix = scoped.pivot_table(index="LBName", columns="Front", values="Votes", aggfunc="sum", fill_value=0)
    matrix = matrix.reindex(columns=FRONT_ORDER, fill_value=0)
    matrix.loc["Total"] = matrix.sum(numeric_only=True)
    matrix = matrix.reset_index().rename(columns={"index": "LBName"})
    return TableResult(matrix)


def party_lb_performance(df: pd.DataFrame, sel_party: str) -> TableResult:
    filtered = df[df["Party"].astype(str) == sel_party]
    table, _, _ = _rank_table(filtered, ["LBName"])
    if table.empty:
        return TableResult(table)
    table = table.sort_values("Won", ascending=False).reset_index(drop=True)
    return TableResult(table)


def opponent_breakdown(df: pd.DataFrame, sel_party: str) -> TableResult:
    scoped, _ = _partition_scope(df)
    if scoped.empty or "Rank" not in scoped.columns:
        empty = pd.DataFrame(columns=["Party", f"Runner-up when {sel_party} Won", f"Winners when {sel_party} became second"])
        return TableResult(empty)

    keys = [col for col in ["WardCode", "District", "LBName", "WardName"] if col in scoped.columns]
    winners = scoped[scoped["Rank"] == 1][keys + ["Party"]].rename(columns={"Party": "WinnerParty"})
    runners = scoped[scoped["Rank"] == 2][keys + ["Party"]].rename(columns={"Party": "RunnerParty"})

    win_sel = winners[winners["WinnerParty"].astype(str) == sel_party]
    ru_counts = (
        win_sel.merge(runners, on=keys, how="left")
        .groupby("RunnerParty", dropna=True)
        .size()
        .rename(f"Runner-up when {sel_party} Won")
    )
    ru_counts = ru_counts.rename_axis("Party").reset_index()
    ru_counts["Party"] = ru_counts["Party"].fillna("UNKNOWN").astype(str)

    runner_sel = runners[runners["RunnerParty"].astype(str) == sel_party]
    win_counts = (
        runner_sel.merge(winners, on=keys, how="left")
        .groupby("WinnerParty", dropna=True)
        .size()
        .rename(f"Winners when {sel_party} became second")
    )
    win_counts = win_counts.rename_axis("Party").reset_index()
    win_counts["Party"] = win_counts["Party"].fillna("UNKNOWN").astype(str)

    combined = pd.merge(ru_counts, win_counts, on="Party", how="outer").fillna(0)
    combined[[col for col in combined.columns if col != "Party"]] = combined[[col for col in combined.columns if col != "Party"]].astype(int)
    colors = [PARTY_BG_COLORS.get(str(row.get("Party", "")), DEFAULT_BG_COLOR) for _, row in combined.iterrows()]
    combined = combined.sort_values(by=combined.columns[1:].tolist(), ascending=False).reset_index(drop=True)
    return TableResult(combined, colors)


def strength_table(df: pd.DataFrame, sel_party: str) -> TableResult:
    scoped = df[(df.get("TierNorm", df.get("Tier", "")).astype(str).str.title() == "Ward") & (df["Party"].astype(str) == sel_party)].copy()
    if scoped.empty:
        return TableResult(pd.DataFrame(columns=["Strength Band", "Ward Count", "Ward Names"]))
    if "Strength" in scoped.columns and scoped["Strength"].notna().any():
        scoped["StrengthBand"] = scoped["Strength"].astype(str)
    elif "Lead" in scoped.columns:
        scoped["StrengthBand"] = scoped["Lead"].apply(_lead_to_strength)
    else:
        return TableResult(pd.DataFrame(columns=["Strength Band", "Ward Count", "Ward Names"]))

    scoped = scoped.dropna(subset=["StrengthBand"])
    scoped["StrengthBand"] = pd.Categorical(scoped["StrengthBand"], categories=STRENGTH_ORDER, ordered=True)
    rows: List[Dict[str, object]] = []
    for strength, frame in scoped.groupby("StrengthBand", dropna=False, observed=False):
        count = len(frame)
        if count == 0:
            continue
        lb_series = frame.get("LBName", pd.Series('-', index=frame.index)).astype(str)
        ward_series = frame.get("WardName", _resolve_ward_label(frame))
        rows.append({
            "Strength Band": strength,
            "Ward Count": count,
            "Ward Names": _format_lb_grouped_html(zip(lb_series, ward_series), WINNING_WARD_COLOR, bold_lb=True),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return TableResult(result)
    result["Strength Band"] = pd.Categorical(result["Strength Band"], categories=STRENGTH_ORDER, ordered=True)
    result = result.sort_values("Strength Band").reset_index(drop=True)
    row_colors = [STRENGTH_COLOR_MAP.get(str(row.get("Strength Band", "")), DEFAULT_BG_COLOR) for _, row in result.iterrows()]
    return TableResult(result, row_colors)


def vote_share_strength(df: pd.DataFrame, sel_party: str) -> TableResult:
    columns = ["VoteBin", "Won", "Not Won", "Total", "Winning Wards", "Losing Wards"]
    scoped = df[(df.get("TierNorm", df.get("Tier", "")).astype(str).str.title() == "Ward") & (df["Party"].astype(str) == sel_party)].copy()
    if scoped.empty or "VoteBin" not in scoped.columns:
        return TableResult(pd.DataFrame(columns=columns))

    scoped["VoteBin"] = scoped["VoteBin"].astype(str)
    scoped["_WardLabel"] = _resolve_ward_label(scoped)
    scoped["LBName"] = scoped.get("LBName", "").astype(str)
    scoped["Status"] = np.where(scoped.get("Rank", 0).astype("Int64") == 1, "Won", "Not Won")

    pivot = scoped.groupby(["VoteBin", "Status"], dropna=False).size().unstack(fill_value=0)
    for status in ("Won", "Not Won"):
        if status not in pivot.columns:
            pivot[status] = 0
    pivot = pivot[[col for col in ("Won", "Not Won")]]
    pivot["Total"] = pivot.sum(axis=1)
    table = pivot.reset_index().sort_values("VoteBin").reset_index(drop=True)

    win_pairs = scoped[scoped["Status"] == "Won"][['VoteBin', 'LBName', '_WardLabel']]
    lose_pairs = scoped[scoped["Status"] != "Won"][['VoteBin', 'LBName', '_WardLabel']]

    win_map = {vb: list(zip(grp['LBName'], grp['_WardLabel'])) for vb, grp in win_pairs.groupby('VoteBin')}
    lose_map = {vb: list(zip(grp['LBName'], grp['_WardLabel'])) for vb, grp in lose_pairs.groupby('VoteBin')}

    table["Winning Wards"] = table["VoteBin"].map(lambda vb: _format_lb_grouped_html(win_map.get(vb, []), WINNING_WARD_COLOR))
    table["Losing Wards"] = table["VoteBin"].map(lambda vb: _format_lb_grouped_html(lose_map.get(vb, []), LOSING_WARD_COLOR))
    table = table[[col for col in columns if col in table.columns]]
    row_colors = [_vote_bin_color(row.get("VoteBin")) for _, row in table.iterrows()]
    return TableResult(table, row_colors)


def strongest_wards(df: pd.DataFrame, sel_party: str, threshold: float = 50.0, limit: int = 20) -> TableResult:
    return _ward_strength_list(df, sel_party, threshold, limit, stronger=True)


def weakest_wards(df: pd.DataFrame, sel_party: str, threshold: float = 45.0, limit: int = 20) -> TableResult:
    return _ward_strength_list(df, sel_party, threshold, limit, stronger=False)


def _ward_strength_list(df: pd.DataFrame, sel_party: str, threshold: float, limit: int, stronger: bool) -> TableResult:
    scoped = df[(df.get("TierNorm", df.get("Tier", "")).astype(str).str.title() == "Ward") & (df["Party"].astype(str) == sel_party)].copy()
    if scoped.empty:
        return TableResult(pd.DataFrame(columns=["LBName", "WardName", "Vote share (%)", "Rank"]))

    if "WardTotalVotes" in scoped.columns:
        scoped["WardTotalVotes"] = pd.to_numeric(scoped["WardTotalVotes"], errors="coerce")
    else:
        keys = [col for col in ["WardCode", "District", "LBName", "WardName"] if col in scoped.columns]
        scoped["WardTotalVotes"] = (
            scoped.groupby(keys, dropna=False)["Votes"].transform("sum")
        )

    scoped["Vote share (%)"] = np.where(scoped["WardTotalVotes"] > 0, scoped["Votes"] / scoped["WardTotalVotes"] * 100, 0.0)
    scope = scoped[['LBName', 'WardName', 'Vote share (%)', 'Rank']].copy()
    scope['LBName'] = scope['LBName'].astype(str)
    scope['WardName'] = scope['WardName'].astype(str)
    scope['Rank'] = pd.to_numeric(scope['Rank'], errors='coerce').astype('Int64')
    if stronger:
        filtered = scope[scope['Vote share (%)'] >= threshold].sort_values(['Vote share (%)', 'WardName'], ascending=[False, True])
    else:
        filtered = scope[scope['Vote share (%)'] <= threshold].sort_values(['Vote share (%)', 'WardName'], ascending=[True, True])
    filtered['Vote share (%)'] = filtered['Vote share (%)'].round(2)
    if limit:
        filtered = filtered.head(limit)
    return TableResult(filtered.reset_index(drop=True))


def strength_chart(df: pd.DataFrame, sel_party: str) -> Optional[px.bar]:
    scoped = df[(df.get("TierNorm", df.get("Tier", "")).astype(str).str.title() == "Ward") & (df["Party"].astype(str) == sel_party)].copy()
    if scoped.empty:
        return None
    if "Strength" in scoped.columns and scoped["Strength"].notna().any():
        strength = scoped["Strength"].astype(str)
    elif "Lead" in scoped.columns:
        strength = scoped["Lead"].apply(_lead_to_strength)
    else:
        return None
    summary = pd.DataFrame({"Strength": strength}).dropna()
    if summary.empty:
        return None
    summary["Strength"] = pd.Categorical(summary["Strength"], categories=STRENGTH_ORDER, ordered=True)
    summary = summary.groupby("Strength", as_index=False, observed=False).size().rename(columns={"size": "Wards"}).sort_values("Strength")
    mirror = summary.copy()
    mirror["Display_Wards"] = mirror.apply(lambda row: -row["Wards"] if str(row["Strength"]).startswith("-") else row["Wards"], axis=1)
    mirror["Status"] = mirror["Strength"].apply(lambda s: "Lost" if str(s).startswith("-") else "Won")
    fig = px.bar(
        mirror,
        x="Display_Wards",
        y="Strength",
        orientation="h",
        text="Wards",
        color="Status",
        color_discrete_map={"Won": "#6c80ac", "Lost": "#cc807c"},
    )
    fig.update_layout(
        xaxis_title="Number of Wards",
        yaxis_title="Strength Category",
        height=520,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="black"),
        legend=dict(title="Status", orientation="h", x=1.0, y=0, xanchor="right", yanchor="bottom"),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def vote_bin_chart(df: pd.DataFrame, sel_party: str) -> Optional[px.bar]:
    scoped = df[(df.get("TierNorm", df.get("Tier", "")).astype(str).str.title() == "Ward") & (df["Party"].astype(str) == sel_party)].copy()
    if scoped.empty or "VoteBin" not in scoped.columns:
        return None
    scoped["Status"] = np.where(scoped.get("Rank", 0).astype("Int64") == 1, "Won", "Not won")
    agg = scoped.groupby(["VoteBin", "Status"], as_index=False).size().rename(columns={"size": "Wards"})
    if agg.empty:
        return None
    fig = px.bar(
        agg,
        x="VoteBin",
        y="Wards",
        color="Status",
        barmode="stack",
        color_discrete_map={"Won": "#6c80ac", "Not won": "#cc807c"},
        text="Wards",
    )
    fig.update_traces(texttemplate="%{text}", textposition="inside", insidetextanchor="middle")
    fig.update_layout(
        xaxis_title="VoteBin",
        yaxis_title="Number of Wards",
        height=520,
        legend=dict(title="Status", orientation="h", x=1.0, y=1.05, xanchor="right"),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def _lead_to_strength(lead: float | int | None) -> str | None:
    if pd.isna(lead):
        return None
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

