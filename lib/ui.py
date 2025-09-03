import streamlit as st
import pandas as pd


def hide_index(styler: pd.io.formats.style.Styler) -> pd.io.formats.style.Styler:
    """Hide index compatible across pandas versions."""
    try:
        return styler.hide(axis="index")
    except Exception:
        return styler.hide_index()


def render_styled_table(obj, fmt: dict | None = None):
    """
    Render a DataFrame or Styler with optional format mapping and responsive CSS.
    - obj: DataFrame or Styler
    - fmt: dict of {column: format_string}
    """
    styler = obj if isinstance(obj, pd.io.formats.style.Styler) else obj.style
    if fmt:
        styler = styler.format(fmt)
    styler = hide_index(styler)

    html = styler.to_html()
    st.markdown(
        """
        <style>
          .tbl-wrap { width: 100%; overflow-x: auto; }
          .tbl-wrap table { width: 100%; border-collapse: collapse; table-layout: auto; }
          .tbl-wrap th, .tbl-wrap td { padding: 6px 8px; }
          @media (max-width: 1200px) {
            .tbl-wrap th, .tbl-wrap td { font-size: 0.9rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='tbl-wrap'>{html}</div>", unsafe_allow_html=True)

