from __future__ import annotations
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import numpy as np
import streamlit as st

# We expect the CSV to live at the REPO ROOT with this exact name
DEFAULT_DATA_BASENAME = "data.csv"

# ---------------- path helpers ----------------
def _repo_root() -> Path:
    # lib/data.py -> parent is repo root
    return Path(__file__).resolve().parent.parent

def _default_repo_csv() -> Path:
    return _repo_root() / DEFAULT_DATA_BASENAME

def _candidate_paths() -> list[Path]:
    """Where we’ll look for the CSV if the provided path doesn’t exist."""
    cwd = Path.cwd()
    return [
        Path(DEFAULT_DATA_BASENAME),          # ./data.csv (current working dir)
        _default_repo_csv(),                  # <repo>/data.csv
        cwd / "Data" / "lsgd_data.csv",      # legacy name/dir (backward compat)
        _repo_root() / "Data" / "lsgd_data.csv",
    ]

def get_data_path() -> str:
    """
    Returns the currently selected data path, preferring:
      1) st.secrets["DATA_URL"] (if set)
      2) st.session_state["data_path"] (if user overrode in UI)
      3) <repo>/data.csv (default)
    NOTE: This returns a string because we show it in a text_input.
    """
    # 1) URL via secrets (e.g., https://.../data.csv). Let load_data read the URL directly.
    if hasattr(st, "secrets") and "DATA_URL" in st.secrets and st.secrets["DATA_URL"]:
        return str(st.secrets["DATA_URL"])

    # 2) user override in session (from sidebar control)
    if "data_path" in st.session_state and st.session_state["data_path"]:
        return str(st.session_state["data_path"])

    # 3) default repo path
    return str(_default_repo_csv())

def set_data_path(path: str | Path):
    st.session_state["data_path"] = str(path)

# ---------------- loading ----------------
@st.cache_data(show_spinner=True)
def load_data(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load the CSV with robust fallbacks:
      - If st.secrets["DATA_URL"] is set, that wins.
      - Else use 'path' if provided.
      - Else auto-discover using common locations.
    Works with local paths (Path/str) or HTTP(S) URLs.
    """
    # 1) use DATA_URL secret if present
    secrets_url = st.secrets.get("DATA_URL", None) if hasattr(st, "secrets") else None
    if secrets_url:
        return _read_csv_robust(secrets_url)

    # 2) explicit 'path' argument or session default
    if path is None:
        path = get_data_path()

    # Path may be URL or filesystem path
    try:
        p = Path(path)
        if p.exists():
            return _read_csv_robust(p)
    except Exception:
        # not a filesystem path (likely a URL) -> try read directly
        return _read_csv_robust(path)

    # 3) try candidate locations
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
    """Try multiple encodings; then normalize common columns."""
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

    # ---- light normalizations ----
    # numeric coercions
    if "Votes" in df.columns:
        df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").astype("Int64")
    if "Rank" in df.columns:
        df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce").astype("Int64")

    # totals by ward (if available)
    if "WardTotalVotes" not in df.columns and {"WardCode", "Votes"}.issubset(df.columns):
        df["WardTotalVotes"] = df.groupby("WardCode", dropna=False)["Votes"].transform("sum")

    # vote percentage at ward level
    if "VotePercentage" not in df.columns and {"Votes", "WardTotalVotes"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["VotePercentage"] = (df["Votes"] / df["WardTotalVotes"] * 100)

    # tidy strings
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()

    # standardize a few label columns (helps filters)
    if "Tier" in df.columns:
        df["Tier"] = df["Tier"].astype(str).str.title()
    if "LBType" in df.columns:
        df["LBType"] = df["LBType"].astype(str).str.title()

    return df

# ---------------- optional sidebar controls ----------------
def data_controls():
    """Sidebar controls to override the data path and clear cache."""
    with st.sidebar:
        st.subheader("Data")
        current = get_data_path()
        new_path = st.text_input("CSV path or URL", value=current, help="Use a relative path like 'data.csv' or a full https:// URL")
        if new_path != current:
            set_data_path(new_path)
            # Clear cache so next load uses the new path
            load_data.clear()

        if st.button("Reload data"):
            load_data.clear()
            st.toast("Data cache cleared. It will reload on next access.")
