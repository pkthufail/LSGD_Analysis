import streamlit as st

st.set_page_config(page_title="LSGD Explorer", page_icon="dY-3�,?", layout="wide")

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
st.caption("Use the sidebar to open any page.")

# Landing page: no dataset is loaded here.
st.markdown(
    """
    Explore Kerala Local Self-Government (LSGD) election results across districts,
    assemblies, local bodies, and wards. Analyze party and front performance,
    compare trends, and browse ready-to-share reports. This home page keeps things
    light — data is only loaded when you open a specific page from the sidebar.
    """
)

st.markdown("---")
st.subheader("Pages")
st.markdown(
    """
    - **Overall**: High-level summary of votes, seats, and trends across the state.
    - **District**: Drill down into results and metrics by district.
    - **Assembly**: View outcomes aggregated by assembly segments.
    - **Local Body**: Analyze performance by local body type and institution.
    - **Ward**: Ward-level details including winners, margins, and turnout.
    - **Front**: Alliance-wise performance and comparisons across tiers.
    - **Party**: Selected party’s vote share, seats, and key opponent breakdowns.
    - **Other**: Additional cuts, comparisons, and exploratory views.
    - **Reports**: Curated tables and exports for quick sharing.
    """
)

