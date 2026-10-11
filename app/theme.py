"""A compact header control that uses Streamlit's native per-user themes."""

from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
_switcher = st.components.v2.component(
    "floodlab_theme",
    html="""<div class="theme-switcher">
      <button type="button" aria-label="Switch to light theme" title="Switch to light theme" aria-pressed="true">
        <svg class="sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true">
          <circle cx="12" cy="12" r="4"/>
          <path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>
        </svg>
        <svg class="moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M20.8 13A9 9 0 0 1 11 3.2 9 9 0 1 0 20.8 13Z"/>
        </svg>
      </button>
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
