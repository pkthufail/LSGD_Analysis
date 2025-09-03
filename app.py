import streamlit as st
from lib.data import load_data, get_data_path, data_controls

st.set_page_config(page_title="LSGD Explorer", page_icon="🗳️", layout="wide")

# optional: tiny CSS polish for sidebar links
st.markdown(
    """
    <style>
      section[data-testid="stSidebar"] .stMarkdown a { display:block; padding:6px 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LSGD Election Explorer")
st.caption("Use the sidebar pages to navigate.")

# Global data controls in sidebar (applies across pages)
data_controls()

# Quick snapshot (so you know data is loading fine)
try:
    df = load_data(get_data_path())
    st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    st.dataframe(df.head(25), use_container_width=True)
except Exception as e:
    st.error(f"Failed to load data: {e}")

