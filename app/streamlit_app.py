"""Run from any directory with: streamlit run app/streamlit_app.py."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.views import ablation, comparison, explorer, footer, pinn
from app.theme import theme_switcher

st.set_page_config(
    page_title="FloodLab AI · Scientific ML", page_icon=":material/water:", layout="wide"
)
st.html(ROOT / "app/style.css")
theme_switcher()
pages = [
    st.Page(
        explorer, title="Flood Explorer", icon=":material/water:", default=True, url_path="explorer"
    ),
    st.Page(
        comparison, title="Model Comparison", icon=":material/bar_chart:", url_path="comparison"
    ),
    st.Page(ablation, title="Sensor Ablation", icon=":material/sensors:", url_path="sensors"),
    st.Page(pinn, title="PINN Explorer", icon=":material/functions:", url_path="physics"),
]
page = st.navigation(pages, position="hidden")
with st.sidebar:
    st.html(
        '<div class="brand">FloodLab <span>AI</span></div><div class="sidebar-caption">Flood reconstruction & scientific ML</div>'
    )
    st.html('<div class="sidebar-label">Workspace</div>')
    for destination in pages:
        st.page_link(destination)
    st.divider()
    st.markdown("[Source code ↗](https://github.com/dols30/Physics-Informed-Flood-Reconstruction)")
    st.markdown(
        "[Kaggle notebook ↗](https://www.kaggle.com/code/bashcode223/physics-informed-flood-reconstruction-ml-vs-pinn)"
    )
    st.html('<div class="sidebar-credit">Dol Raj Bashyal<br>Research project</div>')

try:
    page.run()
    footer()
except (FileNotFoundError, ValueError) as error:
    st.error("The experiment data could not be loaded.")
    with st.expander("Details"):
        st.code(str(error))
        st.markdown(
            "Restore the experiment files or see the dashboard setup guide in the repository."
        )
    st.stop()
