import streamlit as st

st.set_page_config(page_title="LSGD Explorer", page_icon="dY", layout="wide")

st.title("LSGD Election Insights Dashboard")

st.markdown(
    """
    Dive into the Kerala Local Self-Government (LSGD) election story with a dashboard
    built for strategists, data teams, and campaign analysts. Use the sidebar to move
    between focused analysis tabs - every page loads data only when you open it, so the
    experience stays fast and lightweight.
    """
)

st.subheader("Highlights at a Glance")
st.markdown(
    """
    - **Complete coverage**: District, Assembly, Local Body, and Ward level snapshots with consistent colour coding for fronts and parties.
    - **Actionable breakdowns**: Strike-rate, vote-share, and positional tables that surface where momentum is gained or lost.
    - **Assembly-ready decks**: PDF exports stitch tables, charts, and formatted ward lists together for instant sharing.
    - **Micro insights**: Strong vs weak ward rolls, vote-bin distributions, and opponent matchups reveal ground reality quickly.
    - **Flexible filters**: Pick your district, assembly, front, or party from the sidebar to drill straight into the stories you need.
    """
)

st.subheader("Where to go next")
st.markdown(
    """
    - **Front / Party pages**: Track performance, vote bins, and strength bands with export-ready visuals.
    - **Assembly & Local Body views**: Compare alliances across institutions and spot shifts between 2020 and 2025 ward maps.
    - **Reports**: Generate curated PDFs for stakeholder briefings, including ward groupings and chart snapshots.
    - **Ward explorer**: Zoom right down to candidate-level numbers, leads, and trails.
    """
)

st.markdown("---")
st.caption("Navigate via the sidebar to start exploring - every selection refreshes the insights in seconds.")
