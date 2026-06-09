from __future__ import annotations
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import numpy as np
import streamlit as st

DEFAULT_DATA_BASENAME = "data_2025.csv"

# ---------------- safe secrets helper ----------------
def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Return st.secrets[key] if available; otherwise default (no exceptions)."""
    try:
        return st.secrets[key]  # type: ignore[index]
    except Exception:
        return default

# ---------------- path helpers ----------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _default_repo_csv() -> Path:
    return _repo_root() / DEFAULT_DATA_BASENAME

def _candidate_paths() -> list[Path]:
    cwd = Path.cwd()
    return [
        Path(DEFAULT_DATA_BASENAME),     # ./data.csv (current working dir)
        _default_repo_csv(),             # <repo>/data.csv
        cwd / "Data" / "lsgd_data.csv",  # legacy
        _repo_root() / "Data" / "lsgd_data.csv",
    ]

def get_data_path() -> str:
    """
    Returns the selected data path, preferring:
      1) DATA_URL in secrets (if set)
      2) user override in session_state
      3) <repo>/data.csv
    """
    # 1) Safe secrets access
    url = _get_secret("DATA_URL", None)
    if url:
        return str(url)

    # 2) user override (from sidebar)
    if "data_path" in st.session_state and st.session_state["data_path"]:
        return str(st.session_state["data_path"])

    # 3) default repo path
    return str(_default_repo_csv())

def set_data_path(path: str | Path):
    st.session_state["data_path"] = str(path)

# ---------------- loading ----------------
@st.cache_data(show_spinner=True, ttl=300)
def load_data(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load CSV with fallbacks:
      - DATA_URL secret (if present)
      - explicit path or session default
      - common repo-relative locations
    """
    # 1) secrets URL (safe)
    secrets_url = _get_secret("DATA_URL", None)
    if secrets_url:
        return _read_csv_robust(secrets_url)

    # 2) explicit path or session default
    if path is None:
        path = get_data_path()

    # Path may be URL or filesystem path
    try:
        p = Path(path)  # may raise if it's a URL (that's fine)
        if p.exists():
            return _read_csv_robust(p)
    except Exception:
        # not a local path → try reading directly (e.g., URL)
        return _read_csv_robust(path)

    # 3) candidate locations
    for cand in _candidate_paths():
        if cand.exists():
            return _read_csv_robust(cand)

    # 4) helpful error
    raise FileNotFoundError(
        "Could not locate the data file.\n"
        f"- Tried: {path}\n"
        f"- Also tried: {', '.join(map(str, _candidate_paths()))}\n"
        f"- CWD: {Path.cwd()}\n"
        "Tip: Place 'data.csv' at the repo root, or set a URL in Streamlit Secrets as DATA_URL."
    )

def _read_csv_robust(src: Union[str, Path]) -> pd.DataFrame:
    encs = ["utf-8-sig", "utf-8", "cp1252"]
    last_err: Exception | None = None
    for enc in encs:
        try:
            df = pd.read_csv(src, encoding=enc, low_memory=False)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"Could not read CSV at {src}: {last_err}")

    # normalizations
    if "Votes" in df.columns:
        df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").astype("Int64")
    if "Rank" in df.columns:
        df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce").astype("Int64")

    if "WardTotalVotes" not in df.columns and {"WardCode", "Votes"}.issubset(df.columns):
        df["WardTotalVotes"] = df.groupby("WardCode", dropna=False)["Votes"].transform("sum")

    if "VotePercentage" not in df.columns and {"Votes", "WardTotalVotes"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["VotePercentage"] = (df["Votes"] / df["WardTotalVotes"] * 100)

    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()

    if "Tier" in df.columns:
        df["Tier"] = df["Tier"].astype(str).str.title()
        # Provide a normalized Tier column for downstream pages/utilities
        df["TierNorm"] = df["Tier"].astype(str).str.title()
    if "LBType" in df.columns:
        df["LBType"] = df["LBType"].astype(str).str.title()

    return df

# ---------------- sidebar controls ----------------
def data_controls():
    with st.sidebar:
        st.subheader("Data")
        current = get_data_path()
        new_path = st.text_input(
            "CSV path or URL",
            value=current,
            help="Use a relative path like 'data.csv' or a full https:// URL. If you configure DATA_URL in secrets, that will be used automatically."
        )
        if new_path != current:
            set_data_path(new_path)
            load_data.clear()

        if st.button("Reload data"):
            load_data.clear()
            st.toast("Data cache cleared. It will reload on next access.")


@st.cache_data(show_spinner=False, ttl=300)
def load_wards_2025(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load ward master for 2025 elections if available."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend([
        Path('Wards_2025.csv'),
        Path.cwd() / 'Wards_2025.csv',
        _repo_root() / 'Wards_2025.csv',
    ])
    seen: set[str] = set()
    for cand in candidates:
        cand_path = Path(cand)
        key = str(cand_path.resolve()) if cand_path.exists() else str(cand_path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if cand_path.exists():
                df = pd.read_csv(cand_path, dtype=str, low_memory=False)
                for col in ['LBCode', 'WardCode']:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.strip()
                return df
        except Exception:
            continue
    return pd.DataFrame(columns=['District', 'LBCode', 'LBName', 'WardCode'])


def lb_ward_count_lookup(df: pd.DataFrame, wards_2025: Optional[pd.DataFrame] = None) -> dict[str, dict[str, int]]:
    """Return ward counts for each LBCode for 2020 dataset and 2025 master."""
    if wards_2025 is None:
        wards_2025 = load_wards_2025()

    base = df.copy()
    if 'LBCode' not in base.columns or 'WardCode' not in base.columns:
        counts_2020 = pd.Series(dtype='int64')
    else:
        tier_col = 'TierNorm' if 'TierNorm' in base.columns else ('Tier' if 'Tier' in base.columns else None)
        if tier_col is not None:
            base = base[base[tier_col].astype(str).str.title() == 'Ward']
        base['LBCode'] = base['LBCode'].astype(str).str.strip()
        base['WardCode'] = base['WardCode'].astype(str).str.strip()
        counts_2020 = base.groupby('LBCode', dropna=True)['WardCode'].nunique()
        counts_2020.index = counts_2020.index.astype(str)

    if wards_2025 is None or wards_2025.empty or 'LBCode' not in wards_2025.columns or 'WardCode' not in wards_2025.columns:
        counts_2025 = pd.Series(dtype='int64')
    else:
        w25 = wards_2025.copy()
        w25['LBCode'] = w25['LBCode'].astype(str).str.strip()
        w25['WardCode'] = w25['WardCode'].astype(str).str.strip()
        counts_2025 = w25.groupby('LBCode', dropna=True)['WardCode'].nunique()
        counts_2025.index = counts_2025.index.astype(str)

    lb_codes = sorted(set(counts_2020.index.tolist()) | set(counts_2025.index.tolist()))
    lookup: dict[str, dict[str, int]] = {}
    for lb_code in lb_codes:
        wards_20 = int(counts_2020.get(lb_code, 0))
        wards_25 = int(counts_2025.get(lb_code, 0))
        lookup[lb_code] = {
            'wards_2020': wards_20,
            'wards_2025': wards_25,
            'new_wards': max(wards_25 - wards_20, 0),
        }
    return lookup
