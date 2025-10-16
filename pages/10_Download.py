import io
import re

import html
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.io as pio
except ModuleNotFoundError:
    pio = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image, PageBreak
    HAS_REPORTLAB = True
except ModuleNotFoundError:
    HAS_REPORTLAB = False

from lib.colors import DEFAULT_BG_COLOR, FRONT_BG_COLORS, PARTY_BG_COLORS, FRONT_COLORS, PARTY_COLORS
from lib.data import data_controls, get_data_path, load_data, load_wards_2025, lb_ward_count_lookup
from lib.report_assembly import (
    summary_by_lb,
    front_performance,
    party_performance,
    seats_by_front,
    votes_by_front,
    party_lb_performance,
    opponent_breakdown,
    strength_table,
    vote_share_strength,
    strongest_wards,
    weakest_wards,
    strength_chart,
    vote_bin_chart
)

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
WINNING_WARD_COLOR = "#2e7d32"
LOSING_WARD_COLOR = "#c62828"

LB_WARD_COUNTS: dict[str, dict[str, int]] = {}

MAX_CELL_TEXT_CHARS = 160
MAX_OTHER_POSITION_ITEMS = 6
LONG_TEXT_COLUMNS = {
    "WardNames {LBName in Bold: (Name of Wards from each LB in bracket)}",
    "Ward Names",
    "Winning Wards",
    "Losing Wards",
    "Other Positions",
}
_TAG_RE = re.compile(r"<[^>]+>")




HAS_KALEIDO: Optional[bool] = None

def _ensure_kaleido() -> bool:
    global HAS_KALEIDO
    if HAS_KALEIDO is None:
        try:
            import kaleido  # noqa: F401
            HAS_KALEIDO = True
        except ModuleNotFoundError:
            HAS_KALEIDO = False
    return bool(HAS_KALEIDO)


def _figure_to_image(fig, width: int = 800, height: int = 420, scale: int = 1) -> Optional[bytes]:
    if fig is None or pio is None:
        return None
    # Try the most compatible path first (no explicit engine),
    # then fall back to the figure method if available.
    try:
        return pio.to_image(fig, format="png", width=width, height=height, scale=scale)
    except Exception:
        try:
            if hasattr(fig, "to_image"):
                return fig.to_image(format="png", width=width, height=height, scale=scale)
        except Exception:
            pass
    return None


def _is_ordinal_column(name: object) -> bool:
    if not isinstance(name, str):
        return False
    value = name.strip()
    if value.lower() == "won":
        return False
    for suffix in ("st", "nd", "rd", "th"):
        if value.endswith(suffix) and value[:-len(suffix)].isdigit():
            return True
    return False


def _rank_label_to_int(label: object) -> Optional[int]:
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    if text.lower() == "won":
        return 1
    match = re.match(r"^(\d+)(st|nd|rd|th)$", text.lower())
    if match:
        return int(match.group(1))
    return None


def _rank_column_order(df: pd.DataFrame, base_cols: Sequence[str]) -> List[str]:
    order: List[str] = []
    base_list = [col for col in base_cols if col in df.columns]
    order.extend(base_list)
    rank_pairs: List[tuple[int, str]] = []
    for col in df.columns:
        pos = _rank_label_to_int(col)
        if pos is not None:
            rank_pairs.append((pos, str(col)))
    for _, col in sorted(rank_pairs, key=lambda item: item[0]):
        if col not in order and col in df.columns:
            order.append(col)
    for col in df.columns:
        if col not in order:
            order.append(col)
    return order


def _reshape_position_columns(df: pd.DataFrame, base_cols: Sequence[str]) -> pd.DataFrame:
    result = df.copy()
    rank_to_col: Dict[int, List[str]] = {}
    for col in result.columns:
        pos = _rank_label_to_int(col)
        if pos is not None:
            rank_to_col.setdefault(pos, []).append(col)
    other_cols: List[str] = []
    for pos, cols in rank_to_col.items():
        if pos and pos > 3:
            other_cols.extend(cols)
    if other_cols:
        result[other_cols] = result[other_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        result['Other Positions'] = result[other_cols].sum(axis=1).astype(int)
        result = result.drop(columns=other_cols, errors='ignore')
    ordered: List[str] = []
    ordered.extend([col for col in base_cols if col in result.columns])
    for label in ['Won', '2nd', '3rd', 'Other Positions']:
        if label in result.columns and label not in ordered:
            ordered.append(label)
    for col in result.columns:
        if col not in ordered:
            ordered.append(col)
    return result[ordered]


def _summarize_other_positions(row: pd.Series, columns: Sequence[str]) -> str:
    entries: List[tuple[str, int]] = []
    for col in columns:
        value = row.get(col, 0)
        if pd.isna(value):
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        entries.append((str(col), count))
    if not entries:
        return '-'
    if len(entries) > MAX_OTHER_POSITION_ITEMS:
        kept = entries[:MAX_OTHER_POSITION_ITEMS - 1]
        remainder = entries[MAX_OTHER_POSITION_ITEMS - 1:]
        others_total = sum(count for _, count in remainder)
        entries = kept + [("Others", others_total)]
    parts = [f"{label}: {count}" for label, count in entries]
    return ", ".join(parts)


def _collapse_rank_columns(table: pd.DataFrame, base_cols: Sequence[str]) -> pd.DataFrame:
    df = table.copy()
    ordinal_cols = [col for col in df.columns if _is_ordinal_column(col)]
    other_cols = [col for col in ordinal_cols if str(col).strip().lower() != "won"]
    if other_cols:
        df["Other Positions"] = df.apply(lambda row: _summarize_other_positions(row, other_cols), axis=1)
        df = df.drop(columns=other_cols, errors="ignore")
    else:
        df["Other Positions"] = "0"
    if "Other Positions" in df.columns:
        df["Other Positions"] = df["Other Positions"].replace({"0": "-"})
    ordered_cols: List[str] = list(base_cols)
    if "Won" in df.columns and "Won" not in ordered_cols:
        ordered_cols.append("Won")
    if "Other Positions" in df.columns and "Other Positions" not in ordered_cols:
        ordered_cols.append("Other Positions")
    trailing = ["Contested", "Votes", "Vote share (%)", "Vote Share (%)", "Strike Rate (%)"]
    for col in trailing:
        if col in df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    for col in df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    df = df[ordered_cols]
    return df


def _append_table_section(
    sections: List[Tuple],
    title: str,
    table: Optional[pd.DataFrame],
    row_colors: Optional[List[str]] = None,
    *,
    rename_map: Optional[dict[str, str]] = None,
    reorder: Optional[Sequence[str]] = None,
    collapse_ranks: bool = False,
    base_cols: Optional[Sequence[str]] = None,
) -> None:
    if table is None or table.empty:
        return
    df = table.copy()
    if rename_map:
        df = df.rename(columns=rename_map)
    if collapse_ranks and base_cols:
        df = _collapse_rank_columns(df, base_cols)
    df = df.reset_index(drop=True)
    if reorder:
        df = df[[col for col in reorder if col in df.columns]]
    df, row_colors = _expand_long_text_rows(df, row_colors, base_cols)
    sections.append((title, df, row_colors))


def _append_chart_section(
    sections: List[Tuple],
    title: str,
    fig,
    *,
    empty_message: str = "No data available for this chart.",
    export_message: str = "Chart export unavailable (install 'kaleido' to enable image export).",
) -> None:
    if fig is None:
        fallback = pd.DataFrame({"Info": [empty_message]})
        sections.append((title, fallback, None))
        return
    img_bytes = _figure_to_image(fig)
    if img_bytes is None:
        fallback = pd.DataFrame({"Info": [export_message]})
        sections.append((title, fallback, None))
        return
    sections.append((title, None, None, img_bytes))

def _strip_html_tags(text: object) -> str:
    if text is None:
        return ''
    return _TAG_RE.sub('', str(text)).strip()


def _chunk_plain_text(text: str, max_chars: int) -> List[str]:
    pieces: List[str] = []
    current: List[str] = []
    current_len = 0
    for token in [t.strip() for t in re.split(r',\s*', text) if t.strip()]:
        addition = len(token) + (2 if current else 0)
        if current and current_len + addition > max_chars:
            pieces.append(', '.join(current))
            current = [token]
            current_len = len(token)
        else:
            current.append(token)
            current_len += addition
    if current:
        pieces.append(', '.join(current))
    return pieces or [text]


def _split_group_text(group: str, max_chars: int) -> List[str]:
    cleaned = group.strip()
    if len(cleaned) <= max_chars:
        return [cleaned]
    if ':' not in cleaned:
        return _chunk_plain_text(cleaned, max_chars)
    head, tail = cleaned.split(':', 1)
    head = head.strip()
    tail = tail.strip()
    if tail.startswith('(') and tail.endswith(')'):
        wards = [w.strip() for w in tail[1:-1].split(',') if w.strip()]
        if not wards:
            return [cleaned]
        chunks: List[str] = []
        current: List[str] = []
        current_len = len(head) + 3
        for ward in wards:
            addition = len(ward) + (2 if current else 0)
            if current and current_len + addition + 1 > max_chars:
                label = head if not chunks else f"{head} (cont.)"
                chunks.append(f"{label}: (" + ', '.join(current) + ')')
                current = [ward]
                current_len = len(head) + 3 + len(ward)
            else:
                current.append(ward)
                current_len += addition
        if current:
            label = head if not chunks else f"{head} (cont.)"
            chunks.append(f"{label}: (" + ', '.join(current) + ')')
        return chunks
    return _chunk_plain_text(cleaned, max_chars)


def _chunk_grouped_text(value: object, max_chars: int) -> List[str]:
    plain = _strip_html_tags(value)
    if not plain or plain == '-':
        return [plain or '-']
    groups: List[str] = []
    segment = ''
    depth = 0
    for char in plain:
        if char == '(':
            depth += 1
        elif char == ')':
            if depth > 0:
                depth -= 1
        if char == ',' and depth == 0:
            if segment.strip():
                groups.append(segment.strip())
            segment = ''
        else:
            segment += char
    if segment.strip():
        groups.append(segment.strip())

    expanded: List[str] = []
    for group in groups:
        expanded.extend(_split_group_text(group, max_chars))

    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_len = 0
    for item in expanded:
        addition = len(item) + (2 if current_chunk else 0)
        if current_chunk and current_len + addition > max_chars:
            chunks.append(current_chunk)
            current_chunk = [item]
            current_len = len(item)
        else:
            current_chunk.append(item)
            current_len += addition
    if current_chunk:
        chunks.append(current_chunk)
    if not chunks:
        return [html.escape(plain)]
    return ['<br/>'.join(html.escape(part) for part in chunk) for chunk in chunks]


def _ensure_columns(df: pd.DataFrame, columns: list[str], default=0) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = default
    return out


def _normalize_party_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a Party-wise Performance/Party Performance table to:
    Front, Party, Won, 2nd, 3rd, Other Positions, Contested,
    Strike Rate, Total Votes, Vote Share
    - Creates missing columns with sensible defaults
    - Aggregates rank columns >3 into 'Other Positions'
    - Renames percentages to match required labels
    - Drops any extra columns
    """
    if df is None or df.empty:
        cols = ["Front", "Party", "Won", "2nd", "3rd", "Other Positions", "Contested", "Strike Rate", "Total Votes", "Vote Share"]
        return pd.DataFrame(columns=cols)

    table = df.copy()
    # Standardize column names
    table = table.rename(columns={
        "Vote share (%)": "Vote Share",
        "Vote Share (%)": "Vote Share",
        "Strike Rate (%)": "Strike Rate",
        "Votes": "Total Votes",
    })

    # Ensure Front/Party exist
    for col in ("Front", "Party"):
        if col not in table.columns:
            table[col] = "-"

    # Aggregate ordinal columns >3 into 'Other Positions'
    table = _reshape_position_columns(table, ("Front", "Party"))

    # Ensure core numeric columns exist
    need_numeric = ["Won", "2nd", "3rd", "Other Positions", "Contested", "Total Votes"]
    table = _ensure_columns(table, need_numeric, default=0)
    for col in need_numeric:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0).astype(int)

    # Derive Contested if missing or zero from ranks
    if ("Contested" not in table.columns) or (table["Contested"].fillna(0) == 0).all():
        contested = (
            pd.to_numeric(table.get("Won", 0), errors="coerce").fillna(0)
            + pd.to_numeric(table.get("2nd", 0), errors="coerce").fillna(0)
            + pd.to_numeric(table.get("3rd", 0), errors="coerce").fillna(0)
            + pd.to_numeric(table.get("Other Positions", 0), errors="coerce").fillna(0)
        )
        table["Contested"] = contested.astype(int)

    # Derive Strike Rate if not present
    if "Strike Rate" not in table.columns:
        won = pd.to_numeric(table.get("Won", 0), errors="coerce").fillna(0)
        cont = pd.to_numeric(table.get("Contested", 0), errors="coerce").fillna(0)
        table["Strike Rate"] = np.where(cont > 0, (won / cont) * 100, 0.0)

    # Derive Vote Share if not present
    if "Vote Share" not in table.columns:
        tv = pd.to_numeric(table.get("Total Votes", 0), errors="coerce").fillna(0)
        grand = float(tv.sum())
        table["Vote Share"] = np.where(grand > 0, tv / grand * 100, 0.0)

    # Round percentage-like columns
    if "Vote Share" in table.columns:
        table["Vote Share"] = pd.to_numeric(table["Vote Share"], errors="coerce").fillna(0.0).round(2)
    if "Strike Rate" in table.columns:
        table["Strike Rate"] = pd.to_numeric(table["Strike Rate"], errors="coerce").fillna(0.0).round(2)

    desired = [
        "Front",
        "Party",
        "Won",
        "2nd",
        "3rd",
        "Other Positions",
        "Contested",
        "Strike Rate",
        "Total Votes",
        "Vote Share",
    ]
    # Keep only desired columns in that order
    table = table[[c for c in desired if c in table.columns]]
    # Add any missing desired columns at end as zeros
    for col in desired:
        if col not in table.columns:
            table[col] = 0 if col not in ("Front", "Party") else "-"
    # Reorder exactly
    table = table[desired]
    return table


def _expand_long_text_rows(
    df: pd.DataFrame,
    row_colors: Optional[List[str]],
    base_cols: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    targets = [col for col in df.columns if col in LONG_TEXT_COLUMNS]
    if not targets:
        return df, row_colors
    base_cols = list(base_cols or ([df.columns[0]] if len(df.columns) else []))
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    new_rows: List[pd.Series] = []
    new_colors: List[str] = []

    df_reset = df.reset_index(drop=True)
    for idx, row in df_reset.iterrows():
        chunk_map: dict[str, List[str]] = {}
        max_chunks = 1
        for col in targets:
            pieces = _chunk_grouped_text(row.get(col, ''), MAX_CELL_TEXT_CHARS)
            chunk_map[col] = pieces
            if len(pieces) > max_chunks:
                max_chunks = len(pieces)
        for part_idx in range(max_chunks):
            new_row = row.copy()
            for col in targets:
                parts = chunk_map.get(col, [''])
                new_row[col] = parts[part_idx] if part_idx < len(parts) else ''
            if part_idx > 0:
                for col in df.columns:
                    if col in targets:
                        continue
                    if col in base_cols:
                        value = row.get(col, '')
                        new_row[col] = f"{value} (cont.)" if part_idx == 1 and str(value).strip() else ''
                    else:
                        new_row[col] = ''
                for col in numeric_cols:
                    if col not in targets:
                        new_row[col] = ''
            new_rows.append(new_row)
            if row_colors is not None:
                if idx < len(row_colors):
                    new_colors.append(row_colors[idx])
    if not new_rows:
        return df, row_colors
    new_df = pd.DataFrame(new_rows, columns=df.columns)
    return new_df, (new_colors if row_colors is not None else row_colors)


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


def _format_name_list(names: list[str], color: str) -> str:
    cleaned = [str(name).strip() for name in names if str(name).strip()]
    if not cleaned:
        return '-'
    tagged = [f'<font color="{color}">{html.escape(name)}</font>' for name in cleaned]
    return ', '.join(tagged)

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
        return pd.DataFrame(columns=["VoteBin", "Won", "Not Won", "Total", "Winning Wards", "Losing Wards"])

    part = df[df["Party"].astype(str) == sel_party].copy()
    if part.empty:
        return pd.DataFrame(columns=["VoteBin", "Won", "Not Won", "Total", "Winning Wards", "Losing Wards"])

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

    def _collect(status: str) -> dict[str, list[str]]:
        subset = part[part["Status"] == status]
        grouped = (
            subset.groupby("VoteBin", dropna=False)["_WardLabel"]
            .apply(lambda x: sorted({str(v).strip() for v in x if str(v).strip()}))
        )
        return {str(k): (list(v) if isinstance(v, (list, tuple)) else [str(v)]) for k, v in grouped.items()}

    winning = _collect("Won")
    losing = _collect("Not Won")
    table["Winning Wards"] = table["VoteBin"].map(lambda vb: _format_name_list(winning.get(str(vb), []), WINNING_WARD_COLOR))
    table["Losing Wards"] = table["VoteBin"].map(lambda vb: _format_name_list(losing.get(str(vb), []), LOSING_WARD_COLOR))
    table = table[["VoteBin", "Won", "Not Won", "Total", "Winning Wards", "Losing Wards"]]
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


def _build_summary_lines(df: pd.DataFrame, sel_front: str, sel_party: str, *, include_majority: bool = True) -> List[str]:
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
        lines.append("Total seats: 0")
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
    if include_majority:
        lines.append(f"Total seats: {total_seats:,} | Majority mark: {majority_mark:,}")
    else:
        lines.append(f"Total seats: {total_seats:,}")
    lb_code_series = df.get("LBCode")
    if lb_code_series is not None:
        lb_values = lb_code_series.dropna().astype(str).unique()
        if len(lb_values) == 1:
            counts_info = LB_WARD_COUNTS.get(lb_values[0])
            if counts_info:
                lines.append(f"No of Wards in 2025: {counts_info['wards_2025']:,} | No of New Wards: {counts_info['new_wards']:,}")
    return lines



def _build_pdf_document(
    title: str,
    summary_lines: Sequence[str],
    sections: Sequence[Tuple],
    header_subtitle: Optional[str] = None,
    header_info: Optional[str] = None,
    page_header: Optional[str] = None,
    front_page: Optional[dict] = None,
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

    # Define custom header styles for medium and small font, if needed
    medium_header = ParagraphStyle(
        "MediumHeader",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",  # ensure non-italic
        alignment=1,  # center
        fontSize=14,
        leading=16,
        spaceAfter=4,
    )
    small_header = ParagraphStyle(
        "SmallHeader",
        parent=styles["BodyText"],
        fontSize=10,
        leading=12,
        spaceAfter=6,
    )
    # Non-italic subheading for section titles
    sub_header = ParagraphStyle(
        "SubHeader",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",  # bold, non-italic
        alignment=0,  # left
        fontSize=12,
        leading=14,
        spaceBefore=6,
        spaceAfter=4,
    )

    # Optional front page (used for Assembly reports)
    if front_page:
        big_header = ParagraphStyle(
            "FrontTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            alignment=1,  # center
            spaceAfter=12,
        )
        label_style = ParagraphStyle(
            "FrontLabel",
            parent=styles["Heading4"],
            fontName="Helvetica",
            fontSize=12,
            leading=14,
            alignment=0,  # left
            spaceAfter=6,
        )
        elems.append(Paragraph(str(front_page.get("title", "")).strip() or title, big_header))
        fp_2020 = front_page.get("wards_2020", 0)
        fp_2025 = front_page.get("wards_2025", 0)
        elems.append(Paragraph(f"No of Wards: 2020: {int(fp_2020):,}, 2025: {int(fp_2025):,}", label_style))
        elems.append(Spacer(1, 12))

        # Front page metrics table
        fp_table_df = front_page.get("table")
        if isinstance(fp_table_df, pd.DataFrame) and not fp_table_df.empty:
            header = [Paragraph(str(col), header_paragraph) for col in fp_table_df.columns]
            data = [header]
            for _, row in fp_table_df.iterrows():
                cells = [Paragraph(_format_cell(value), body_paragraph) for value in row.tolist()]
                data.append(cells)
            # Fill full page width with 6 columns: Metric + five numeric columns
            # widths: [Metric, 2020, 2025, Strike Rate, Expected Win, Expected Gain]
            fixed_sum = 70 + 80 + 70 + 80 + 80
            metric_width = max(doc.width - fixed_sum, 160)
            col_widths = [metric_width, 70, 80, 70, 80, 80]
            tbl = Table(data, repeatRows=1, colWidths=col_widths)
            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 1), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
            ]
            # Optional row background colors from front_page
            row_colors = front_page.get("row_colors") if isinstance(front_page, dict) else None
            if row_colors:
                for idx, color_hex in enumerate(row_colors, start=1):  # +1 to offset header row
                    try:
                        style_cmds.append(("BACKGROUND", (0, idx), (-1, idx), colors.HexColor(str(color_hex))))
                    except Exception:
                        style_cmds.append(("BACKGROUND", (0, idx), (-1, idx), colors.HexColor(DEFAULT_BG_COLOR)))
            tbl.setStyle(TableStyle(style_cmds))
            elems.append(tbl)
            # Independents lines (UDF & LDF) right after the first table
            if isinstance(front_page, dict):
                indep_lines = front_page.get("independents_lines")
                if indep_lines:
                    elems.append(Spacer(1, 6))
                    for label, value in indep_lines:
                        elems.append(Paragraph(f"{label}: {int(value):,}", label_style))
            # Footnote for target percentage
            if isinstance(front_page, dict) and front_page.get("target_pct") is not None:
                elems.append(Spacer(1, 4))
                elems.append(Paragraph(f"(+Target% applied: {int(front_page['target_pct'])}%)", styles["BodyText"]))

        # (Second table removed as requested: values merged into first table)
        elems.append(PageBreak())

    if header_subtitle is not None or header_info is not None:
        # New heading layout for Assembly / Local Body
        elems.append(Paragraph(title, medium_header))
        if header_subtitle:
            elems.append(Paragraph(header_subtitle, medium_header))
        if header_info:
            elems.append(Paragraph(header_info, small_header))
        elems.append(Paragraph(datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S"), styles["BodyText"]))
        elems.append(Spacer(1, 8))
    else:
        # Default heading layout
        elems.append(Paragraph(title, styles["Title"]))
        elems.append(Paragraph(datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S"), styles["BodyText"]))
        elems.append(Spacer(1, 8))

    if summary_lines:
        elems.append(Paragraph(summary_lines[0], sub_header))
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

    for section in sections:
        chart_bytes = None
        if isinstance(section, tuple) and len(section) == 4:
            sec_title, df, row_colors, chart_bytes = section
        elif isinstance(section, tuple) and len(section) == 3:
            sec_title, df, row_colors = section
        else:
            # unexpected structure; best effort unpacking
            sec_title = section[0]
            df = section[1] if len(section) > 1 else None
            row_colors = section[2] if len(section) > 2 else None
            chart_bytes = section[3] if len(section) > 3 else None

        elems.append(Spacer(1, 10))
        elems.append(Paragraph(sec_title, sub_header))

        if chart_bytes:
            try:
                img = Image(io.BytesIO(chart_bytes))
                img._restrictSize(doc.width, 320)
                img.hAlign = "CENTER"
                elems.append(img)
            except Exception:
                elems.append(Paragraph("Unable to render chart image.", styles["BodyText"]))
            continue

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            elems.append(Paragraph("No data available.", styles["BodyText"]))
            continue

        table_df = df.reset_index(drop=True)
        header = [Paragraph(str(col), header_paragraph) for col in table_df.columns]
        data = [header]
        for _, row in table_df.iterrows():
            cells = [Paragraph(_format_cell(value), body_paragraph) for value in row.tolist()]
            data.append(cells)
        col_width_map = {
            "Front": 70,
            "Party": 70,
            "Strength Band": 100,
            "No. of Wards": 75,
            "No. of Wards in 2025": 90,
            "No of Wards": 75,
            "Ward Count": 75,
            "VoteBin": 55,
            "Won": 40,
            "Not Won": 45,
            "Total": 45,
            "Votes": 65,
            "Total Votes": 70,
            "Contested": 50,
            "Other Positions": 60,
            "Vote Share (%)": 60,
            "Vote Share": 60,
            "Strike Rate (%)": 60,
            "Strike Rate": 60,
            "WardNames {LBName in Bold: (Name of Wards from each LB in bracket)}": 320,
            "Winning Wards": 180,
            "Losing Wards": 180,
        }
        col_widths = [col_width_map.get(str(col), None) for col in table_df.columns]
        # Ensure specified widths fit within page width; leave room for auto columns if any
        if any(w is not None for w in col_widths):
            fixed_widths = [float(w) for w in col_widths if isinstance(w, (int, float))]
            total_fixed = sum(fixed_widths)
            auto_cols = sum(1 for w in col_widths if w is None)
            target_width = float(doc.width) * (0.85 if auto_cols > 0 else 1.0)
            if total_fixed > 0 and total_fixed > target_width:
                scale = target_width / total_fixed
                col_widths = [(float(w) * scale) if isinstance(w, (int, float)) else None for w in col_widths]
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

    def _draw_header_footer(canvas, doc, show_header: bool) -> None:
        canvas.saveState()
        footer_text = f"Page {doc.page}"
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(doc.pagesize[0] / 2.0, 20, footer_text)
        if show_header:
            canvas.setFont("Helvetica-Bold", 9)
            header_text = page_header if page_header else title
            canvas.drawString(doc.leftMargin, doc.pagesize[1] - 20, header_text)
        canvas.restoreState()

    doc.build(
        elems,
        onFirstPage=lambda canvas, doc: _draw_header_footer(canvas, doc, False),
        onLaterPages=lambda canvas, doc: _draw_header_footer(canvas, doc, True),
    )
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




def _generate_assembly_sections(df: pd.DataFrame, sel_front: str, sel_party: str) -> Tuple[List[str], List[Tuple]]:
    ward = df[df.get("TierNorm", df.get("Tier", "")).astype(str).str.title() == "Ward"].copy()
    if ward.empty:
        ward = df.copy()

    summary_lines = _build_summary_lines(ward, sel_front, sel_party, include_majority=False)
    sections: List[Tuple] = []

    summary_result = summary_by_lb(ward, LB_WARD_COUNTS)
    summary_table = getattr(summary_result, "frame", pd.DataFrame())
    if not summary_table.empty:
        summary_table = summary_table.rename(
            columns={
                "Wards (2020)": "No. of Wards",
                "Wards (2025)": "No. of Wards in 2025",
            }
        )
        _append_table_section(
            sections,
            "Summary Table",
            summary_table,
            getattr(summary_result, "row_colors", None),
            reorder=["LBName", "No. of Wards", "No. of Wards in 2025", "New Wards", "Total Votes"],
        )

    front_result = front_performance(ward)
    front_table = getattr(front_result, "frame", pd.DataFrame())
    if not front_table.empty:
        front_table = front_table.rename(columns={"Vote share (%)": "Vote Share (%)"})
        front_table = _reshape_position_columns(front_table, ("Front",))
        front_order = _rank_column_order(front_table, ("Front",))
        _append_table_section(
            sections,
            "Front-wise Performance",
            front_table,
            getattr(front_result, "row_colors", None),
            collapse_ranks=False,
            base_cols=("Front",),
            reorder=front_order,
        )

    party_result = party_performance(ward)
    party_table = getattr(party_result, "frame", pd.DataFrame())
    if not party_table.empty:
        party_table = _normalize_party_performance_table(party_table)
        _append_table_section(
            sections,
            "Party-wise Performance",
            party_table,
            getattr(party_result, "row_colors", None),
            collapse_ranks=False,
            base_cols=("Front", "Party"),
            reorder=["Front", "Party", "Won", "2nd", "3rd", "Other Positions", "Contested", "Strike Rate", "Total Votes", "Vote Share"],
        )

    seats_result = seats_by_front(ward)
    seats_table = getattr(seats_result, "frame", pd.DataFrame())
    if not seats_table.empty:
        numeric_cols = seats_table.select_dtypes(include="number").columns
        if len(numeric_cols):
            seats_table[numeric_cols] = seats_table[numeric_cols].fillna(0).astype(int)
        _append_table_section(
            sections,
            "Front-wise Seats Won",
            seats_table,
            getattr(seats_result, "row_colors", None),
            reorder=["LBName", "UDF", "LDF", "NDA", "OTH", "Leader"],
        )

    votes_result = votes_by_front(ward)
    votes_table = getattr(votes_result, "frame", pd.DataFrame())
    if not votes_table.empty:
        front_cols = [col for col in FRONT_ORDER if col in votes_table.columns]
        total_mask = votes_table.get("LBName", pd.Series(dtype=str)).astype(str).str.lower() == "total"
        data_rows = votes_table[~total_mask].copy()
        total_row = votes_table[total_mask].copy() if total_mask.any() else pd.DataFrame(columns=votes_table.columns)
        if front_cols:
            for frame in (data_rows, total_row):
                if not frame.empty:
                    frame[front_cols] = frame[front_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
            leader_colors = []
            for _, row in data_rows.iterrows():
                votes = {front: float(row.get(front, 0) or 0) for front in front_cols}
                best_front = max(votes, key=votes.get) if votes else None
                top_value = votes.get(best_front, 0) if votes else 0
                if list(votes.values()).count(top_value) > 1:
                    leader_colors.append(DEFAULT_BG_COLOR)
                else:
                    leader_colors.append(FRONT_BG_COLORS.get(best_front, DEFAULT_BG_COLOR))
        else:
            leader_colors = [DEFAULT_BG_COLOR] * len(data_rows)
        front_totals = (total_row.iloc[0][front_cols].astype(float) if not total_row.empty else data_rows[front_cols].sum(numeric_only=True)) if front_cols else pd.Series(dtype=float)
        grand_total = float(front_totals.sum()) if not front_totals.empty else 0.0
        share_values = {front: (front_totals.get(front, 0.0) / grand_total * 100) if grand_total > 0 else 0.0 for front in front_cols}
        share_row = {"LBName": "Vote Share %"}
        share_row.update({front: f"{share_values.get(front, 0.0):.2f}%" for front in front_cols})
        position_series = front_totals.rank(ascending=False, method="dense") if not front_totals.empty else pd.Series(dtype=float)
        position_row = {"LBName": "Front Position"}
        position_row.update({front: int(position_series.get(front, np.nan)) if not pd.isna(position_series.get(front, np.nan)) else '' for front in front_cols})
        assembled = [data_rows]
        if share_row:
            assembled.append(pd.DataFrame([share_row]))
        if position_row:
            assembled.append(pd.DataFrame([position_row]))
        if not total_row.empty:
            assembled.append(total_row)
        votes_display = pd.concat(assembled, ignore_index=True, sort=False)
        votes_display_columns = ["LBName", *front_cols]
        votes_display = votes_display[[col for col in votes_display_columns if col in votes_display.columns]]
        display_colors = leader_colors
        if share_row:
            display_colors.append(DEFAULT_BG_COLOR)
        if position_row:
            display_colors.append(DEFAULT_BG_COLOR)
        if not total_row.empty:
            display_colors.extend([DEFAULT_BG_COLOR] * len(total_row))
        _append_table_section(
            sections,
            "Votes by Front",
            votes_display,
            display_colors,
            reorder=votes_display_columns,
        )

    lb_result = party_lb_performance(ward, sel_party)
    lb_table = getattr(lb_result, "frame", pd.DataFrame())
    if not lb_table.empty:
        lb_table = lb_table.rename(columns={"Vote share (%)": "Vote Share (%)"})
        lb_table = _reshape_position_columns(lb_table, ("LBName",))
        lb_order = _rank_column_order(lb_table, ("LBName",))
        _append_table_section(
            sections,
            f"{sel_party} - LB-wise Performance",
            lb_table,
            getattr(lb_result, "row_colors", None),
            collapse_ranks=False,
            base_cols=("LBName",),
            reorder=lb_order,
        )

    opponent_result = opponent_breakdown(ward, sel_party)
    opponent_table = getattr(opponent_result, "frame", pd.DataFrame())
    if not opponent_table.empty:
        opp_colors = [PARTY_BG_COLORS.get(str(row.get("Party", "")), DEFAULT_BG_COLOR) for _, row in opponent_table.iterrows()]
        sections.append((f"Opponent Breakdown - {sel_party}", opponent_table.reset_index(drop=True), opp_colors))

    strength_fig = strength_chart(ward, sel_party)
    _append_chart_section(
        sections,
        "Number of Strong and Weak Wards",
        strength_fig,
        empty_message="No ward-level strength data available for the selected party.",
    )

    vote_bin_fig = vote_bin_chart(ward, sel_party)
    _append_chart_section(
        sections,
        "VoteBin Analysis (Won vs Not Won)",
        vote_bin_fig,
        empty_message="No VoteBin distribution available for the selected party.",
    )

    strength_result = strength_table(ward, sel_party)
    strength_frame = getattr(strength_result, "frame", pd.DataFrame())
    if not strength_frame.empty:
        strength_frame = strength_frame.rename(
            columns={
                "Ward Count": "No of Wards",
            }
        )
        _append_table_section(
            sections,
            f"{sel_party} - Lead Strength",
            strength_frame,
            getattr(strength_result, "row_colors", None),
            reorder=["Strength Band", "No of Wards", "Ward Names"],
            base_cols=("Strength Band",),
        )

    vote_strength_result = vote_share_strength(ward, sel_party)
    vote_strength_frame = getattr(vote_strength_result, "frame", pd.DataFrame())
    if not vote_strength_frame.empty:
        _append_table_section(
            sections,
            f"{sel_party} - Vote Share Strength",
            vote_strength_frame,
            getattr(vote_strength_result, "row_colors", None),
            reorder=["VoteBin", "Won", "Not Won", "Total", "Winning Wards", "Losing Wards"],
            base_cols=("VoteBin",),
        )

    strong_result = strongest_wards(ward, sel_party, threshold=50.0, limit=20)
    strong_frame = getattr(strong_result, "frame", pd.DataFrame())
    if not strong_frame.empty:
        strong_frame = strong_frame.rename(columns={"Vote share (%)": "Vote Share (%)"})
        if "LBName" in strong_frame.columns:
            strong_frame["LBName"] = strong_frame["LBName"].astype(str).map(lambda v: f"<b>{html.escape(v)}</b>" if v.strip() else v)
        _append_table_section(
            sections,
            "Strongest Wards",
            strong_frame,
            getattr(strong_result, "row_colors", None),
            reorder=["LBName", "WardName", "Vote Share (%)", "Rank"],
            base_cols=("LBName",),
        )

    weak_result = weakest_wards(ward, sel_party, threshold=45.0, limit=20)
    weak_frame = getattr(weak_result, "frame", pd.DataFrame())
    if not weak_frame.empty:
        weak_frame = weak_frame.rename(columns={"Vote share (%)": "Vote Share (%)"})
        if "LBName" in weak_frame.columns:
            weak_frame["LBName"] = weak_frame["LBName"].astype(str).map(lambda v: f"<b>{html.escape(v)}</b>" if v.strip() else v)
        _append_table_section(
            sections,
            "Weakest Wards",
            weak_frame,
            getattr(weak_result, "row_colors", None),
            reorder=["LBName", "WardName", "Vote Share (%)", "Rank"],
            base_cols=("LBName",),
        )

    return summary_lines, sections

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
        # Add Total Votes and Vote Share
        votes_by_pair = ward.groupby(["Front", "Party"], dropna=False)["Votes"].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()).to_dict()
        party_summary["Total Votes"] = [int(votes_by_pair.get((row["Front"], row["Party"]), 0)) for _, row in party_summary.iterrows()]
        party_summary["Vote share (%)"] = [
            (value / total_votes_scope * 100) if total_votes_scope > 0 else 0.0
            for value in party_summary["Total Votes"].astype(float)
        ]
        party_summary["Vote share (%)"] = party_summary["Vote share (%)"].round(2)
        # Normalize to required columns and order
        party_summary = _normalize_party_performance_table(party_summary)
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
    if report_type == "Assembly":
        return _generate_assembly_sections(df, sel_front, sel_party)
    return _generate_scope_sections(df, sel_front, sel_party)


def main() -> None:
    st.set_page_config(page_title="Download Reports", page_icon="📄", layout="wide")
    st.title("Download Reports")
    if not HAS_REPORTLAB:
        st.warning("PDF generation requires the 'reportlab' package. Install it via 'pip install reportlab'.")

    data_controls()
    df = load_data(get_data_path()).copy()
    global LB_WARD_COUNTS
    wards_2025_df = load_wards_2025()
    LB_WARD_COUNTS = lb_ward_count_lookup(df, wards_2025_df)
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
        districts = sorted(df.get("District", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        scope["District"] = st.selectbox("District", districts, index=0)
    elif report_type == "Assembly":
        districts = sorted(df.get("District", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        sel_d = st.selectbox("District", districts, index=0)
        scope["District"] = sel_d
        asm_candidates = ["Assembly", "ACName", "AssemblyName", "Constituency"]
        asm_col = next((c for c in asm_candidates if c in df.columns), None)
        dfx = df if sel_d == "All" else df[df.get("District", "").astype(str) == str(sel_d)]
        assemblies = sorted(dfx.get(asm_col, pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) if asm_col else []
        scope["Assembly"] = st.selectbox("Assembly", assemblies, index=0 if assemblies else None)
        # Target percentage points (1-10) for projection
        target_options = list(range(1, 11))
        target_pct = st.selectbox("Target % (+ points)", target_options, index=0)
    else:
        districts = sorted(df.get("District", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        sel_d = st.selectbox("District", districts, index=0)
        scope["District"] = sel_d
        dfx = df[df.get("District", "").astype(str) == str(sel_d)]
        allowed_lb_types = {"Grama", "Municipality", "Corporation"}
        lb_types_raw = sorted(dfx.get("LBType", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        lb_types = [lb for lb in lb_types_raw if lb in allowed_lb_types]
        if not lb_types:
            lb_types = lb_types_raw
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

    # Prepare custom headings for Assembly and Local Body; keep District default
    header_subtitle = None
    header_info = None
    title_for_pdf = title
    page_header = None
    if report_type in ("Assembly", "Local Body"):
        title_for_pdf = "Performance Report"
        if report_type == "Assembly":
            asm_name = str(scope.get("Assembly") or "").strip()
            header_subtitle = (f"Assembly: {asm_name}" if asm_name else "Assembly").upper()
            type_text = "Assembly"
            header_segment = f"{type_text}: {asm_name}" if asm_name else type_text
        else:
            lb_type_raw = str(scope.get("LBType") or "").strip()
            lb_type_map = {"Grama": "Panchayath", "Municipality": "Municipality", "Corporation": "Corporation"}
            lb_type = lb_type_map.get(lb_type_raw, lb_type_raw or "Local Body")
            lb_name = str(scope.get("LBName") or "").strip()
            header_subtitle = (f"{lb_type}: {lb_name}" if lb_name else lb_type).upper()
            type_text = lb_type
            header_segment = f"{lb_type}: {lb_name}" if lb_name else lb_type
        district = str(scope.get("District") or "").strip()
        header_info = ", ".join([
            f"Front: {sel_front}",
            f"Party: {sel_party}",
            f"District: {district}" if district else "District: -",
        ])
        page_header = " - ".join([
            str(sel_party or "").strip() or "Party",
            str(sel_front or "").strip() or "Front",
            district or "District",
            header_segment,
        ])

    # Build optional front page for Assembly
    front_page = None
    if report_type == "Assembly":
        asm_name = str(scope.get("Assembly") or "").strip()
        ward_scope = scoped_df[scoped_df.get("TierNorm", scoped_df.get("Tier", "")).astype(str).str.title() == "Ward"].copy()
        if ward_scope.empty:
            ward_scope = scoped_df.copy()
        # Determine unique ward keys
        keys = _ward_join_keys(ward_scope) if isinstance(ward_scope, pd.DataFrame) else []
        if keys:
            wards_unique = ward_scope.dropna(subset=[k for k in keys if k in ward_scope.columns]).drop_duplicates(subset=[k for k in keys if k in ward_scope.columns])
            seats_2020_total = int(len(wards_unique))
        else:
            seats_2020_total = int(len(ward_scope))
        # Compute 2025 seats by summing wards_2025 for LB codes within scope
        lb_codes = ward_scope.get("LBCode")
        if lb_codes is not None:
            codes = [str(c).strip() for c in lb_codes.dropna().astype(str).unique().tolist()]
        else:
            codes = []
        seats_2025_total = 0
        for code in codes:
            info = LB_WARD_COUNTS.get(code)
            if info and isinstance(info.get("wards_2025"), (int, float)):
                seats_2025_total += int(info.get("wards_2025", 0))
        # Fallback if nothing was found
        if seats_2025_total == 0:
            seats_2025_total = seats_2020_total

        # Helper to count unique ward participation for a filter
        def _count_unique(filter_mask: pd.Series, winners_only: bool = False) -> int:
            part = ward_scope[filter_mask].copy()
            if winners_only and "Rank" in part.columns:
                part = part[pd.to_numeric(part["Rank"], errors="coerce") == 1]
            if part.empty:
                return 0
            if keys:
                part_u = part.dropna(subset=[k for k in keys if k in part.columns]).drop_duplicates(subset=[k for k in keys if k in part.columns])
                return int(len(part_u))
            return int(len(part))

        # Selected Front
        front_mask = ward_scope.get("Front", "").astype(str) == str(sel_front)
        front_cont_20 = _count_unique(front_mask, winners_only=False)
        front_won_20 = _count_unique(front_mask, winners_only=True)

        # Selected Party
        party_mask = ward_scope.get("Party", "").astype(str) == str(sel_party)
        party_cont_20 = _count_unique(party_mask, winners_only=False)
        party_won_20 = _count_unique(party_mask, winners_only=True)

        # Major Party of Selected Front
        major_map = {"UDF": "INC", "LDF": "CPI(M)", "NDA": "BJP", "OTH": "IND"}
        major_party = major_map.get(str(sel_front), "-")
        major_mask = ward_scope.get("Party", "").astype(str) == major_party
        major_cont_20 = _count_unique(major_mask, winners_only=False)
        major_won_20 = _count_unique(major_mask, winners_only=True)

        # Front-wise independents for UDF & LDF
        udf_cont_20 = _count_unique(ward_scope.get("Front", "").astype(str) == "UDF", winners_only=False)
        ldf_cont_20 = _count_unique(ward_scope.get("Front", "").astype(str) == "LDF", winners_only=False)
        udf_indep_2020 = max(0, seats_2020_total - int(udf_cont_20))
        ldf_indep_2020 = max(0, seats_2020_total - int(ldf_cont_20))

        def _project(val_2020: int) -> int:
            if seats_2020_total > 0 and seats_2025_total >= 0:
                return int(round((float(val_2020) / float(seats_2020_total)) * float(seats_2025_total)))
            return 0

        front_proj_total = _project(front_cont_20)
        party_proj_total = _project(party_cont_20)
        major_proj_total = _project(major_cont_20)
        rows = [
            (f"{sel_front} Total", front_cont_20, front_proj_total),
            (f"{sel_front} Win",   front_won_20,  _project(front_won_20)),
            (f"{sel_party} Total", party_cont_20,  party_proj_total),
            (f"{sel_party} Win",   party_won_20,   _project(party_won_20)),
            (f"{major_party} Total", major_cont_20,  major_proj_total),
            (f"{major_party} Win",   major_won_20,   _project(major_won_20)),
        ]
        fp_table = pd.DataFrame(rows, columns=["Metric", "2020 Actuals", "2025 Projection"]) if rows else pd.DataFrame()
        # Row colors for the metric table, aligned with Front/Party palettes
        front_color = FRONT_BG_COLORS.get(str(sel_front), DEFAULT_BG_COLOR)
        party_color = PARTY_BG_COLORS.get(str(sel_party), DEFAULT_BG_COLOR)
        major_color = PARTY_BG_COLORS.get(str(major_party), DEFAULT_BG_COLOR)
        fp_row_colors = [front_color, front_color, party_color, party_color, major_color, major_color]
        # Build strike-rate projection table
        tgt = float(target_pct) if 'target_pct' in locals() else 0.0
        def _strike_rate(won: int, cont: int) -> float:
            return round((float(won) / float(cont) * 100) if cont > 0 else 0.0, 2)
        def _expected_wins(sr_pct: float, proj_seats_2025: int) -> int:
            pct = max(0.0, min(100.0, sr_pct + float(tgt)))
            return int(round(pct / 100.0 * float(max(0, proj_seats_2025))))
        # Build expected wins using group-specific 2025 Total projection, and seat gain as (Wins 2020 - Expected Win 2025), with sign
        front_sr = _strike_rate(front_won_20, front_cont_20)
        front_exp = _expected_wins(front_sr, front_proj_total)
        front_gain = f"{(int(front_exp) - int(front_won_20)):+,}"

        party_sr = _strike_rate(party_won_20, party_cont_20)
        party_exp = _expected_wins(party_sr, party_proj_total)
        party_gain = f"{(int(party_exp) - int(party_won_20)):+,}"

        major_sr = _strike_rate(major_won_20, major_cont_20)
        major_exp = _expected_wins(major_sr, major_proj_total)
        major_gain = f"{(int(major_exp) - int(major_won_20)):+,}"

        proj_rows = [
            (str(sel_front), front_sr, float(tgt), front_exp, front_gain),
            (str(sel_party), party_sr, float(tgt), party_exp, party_gain),
            (str(major_party), major_sr, float(tgt), major_exp, major_gain),
        ]
        # Second table removed; projections merged into first table.

        # Extend first table with Strike Rate/Expected Win/Gain for the Win rows
        for col in ["Strike Rate (%)", "Expected Win 2025", "Expected Seat Gain"]:
            if col not in fp_table.columns:
                fp_table[col] = "-"
        mapping = {
            f"{sel_front} Win": (front_sr, front_exp, front_gain),
            f"{sel_party} Win": (party_sr, party_exp, party_gain),
            f"{major_party} Win": (major_sr, major_exp, major_gain),
        }
        for label, (sr_val, exp_val, gain_val) in mapping.items():
            mask = fp_table["Metric"].astype(str) == label
            fp_table.loc[mask, "Strike Rate (%)"] = round(float(sr_val), 2)
            fp_table.loc[mask, "Expected Win 2025"] = int(exp_val)
            fp_table.loc[mask, "Expected Seat Gain"] = str(gain_val)

        front_page = {
            "title": asm_name,
            "wards_2020": seats_2020_total,
            "wards_2025": seats_2025_total,
            "table": fp_table,
            "row_colors": fp_row_colors,
            "independents_lines": [
                ("Estimated No of UDF Independents in 2020", udf_indep_2020),
                ("Estimated No of LDF Independents in 2020", ldf_indep_2020),
            ],
            "target_pct": int(tgt),
        }

    pdf_bytes = _build_pdf_document(
        title_for_pdf,
        summary_lines,
        sections,
        header_subtitle=header_subtitle,
        header_info=header_info,
        page_header=page_header,
        front_page=front_page,
    )
    file_name = _safe_filename("_".join(filter(None, title_bits)) + ".pdf")

    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
    )


main()
















