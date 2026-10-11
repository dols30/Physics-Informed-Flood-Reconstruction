"""A compact header control that uses Streamlit's native per-user themes."""

from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
_switcher = st.components.v2.component(
    "floodlab_theme",
    html="""<div class="theme-switcher" role="radiogroup" aria-label="Theme">
      <span class="theme-label" aria-hidden="true">Theme</span>
      <button type="button" role="radio" data-theme="System" aria-checked="false" title="Follow system appearance">System</button>
      <button type="button" role="radio" data-theme="Light" aria-checked="false">Light</button>
      <button type="button" role="radio" data-theme="Dark" aria-checked="true" title="Default appearance">Dark</button>
      <span class="theme-status" role="status"></span>
    </div>""",
    css=(HERE / "theme_switcher.css").read_text(),
    js=(HERE / "theme_switcher.js").read_text(),
)


def theme_switcher():
    current = st.session_state.get("ui_appearance", st.context.theme.type)
    result = _switcher(
        key="header_theme",
        height=0,
        data={"appearance": current},
        on_appearance_change=lambda: None,
    )
    if result.appearance in {"light", "dark"}:
        st.session_state["ui_appearance"] = result.appearance
