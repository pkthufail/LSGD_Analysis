from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

DEFAULT_DATA_PATH = r"C:\Users\OFFICE DESK\OneDrive\Desktop\LSGD_Analysis\data.csv"

def get_data_path() -> str:
    return st.session_state.get("data_path", DEFAULT_DATA_PATH)

def set_data_path(path: str):
    st.session_state["data_path"] = path

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    encs = ["utf-8-sig", "utf-8", "cp1252"]
    last_err = None
    for enc in encs:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"Could not read CSV at {path}: {last_err}")

    # light normalizations
    if "Votes" in df.columns:
        df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").astype("Int64")

    if "WardTotalVotes" not in df.columns and {"WardCode","Votes"}.issubset(df.columns):
        df["WardTotalVotes"] = df.groupby("WardCode")["Votes"].transform("sum")

    if "VotePercentage" not in df.columns and {"Votes","WardTotalVotes"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["VotePercentage"] = (df["Votes"] / df["WardTotalVotes"] * 100).round(2)

    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip()

    return df

def data_controls():
    """Optional global controls to change data path and clear cache."""
    with st.sidebar:
        st.subheader("Data")
        path = st.text_input("CSV path", value=get_data_path())
        if path != get_data_path():
            set_data_path(path)
        if st.button("Reload data"):
            load_data.clear()
