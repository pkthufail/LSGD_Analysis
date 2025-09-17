from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

try:
    from pandas.io.formats.style import Styler as _PandasStyler
except (ImportError, AttributeError):
    _PandasStyler = None

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler
else:
    Styler = Any


def hide_index(styler: "Styler") -> "Styler":
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
    if _PandasStyler is not None and isinstance(obj, _PandasStyler):
        styler = obj
    elif hasattr(obj, "style"):
        styler = obj.style
    else:
        raise TypeError("Expected a pandas Styler or DataFrame-like object")
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


def render_color_legend(items: dict[str, str], title: str | None = None):
    """Render a compact legend of color swatches.
    items: mapping of label -> hex color (background)
    """
    if not items:
        return
    title_html = f"<div style='font-weight:600; margin-bottom:4px'>{title}</div>" if title else ""
    swatches = []
    for label, color in items.items():
        color = str(color)
        swatches.append(
            f"<div style='display:flex; align-items:center; margin:2px 8px 2px 0'>"
            f"  <span style='display:inline-block; width:14px; height:14px; border:1px solid #999; background:{color}; margin-right:6px'></span>"
            f"  <span style='font-size:0.9rem'>{label}</span>"
            f"</div>"
        )
    html = (
        "<div class='legend-wrap' style='display:flex; flex-wrap:wrap; align-items:center; margin:6px 0 4px'>"
        + "".join(swatches) + "</div>"
    )
    st.markdown(title_html + html, unsafe_allow_html=True)