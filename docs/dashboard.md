# FloodLab AI

A Streamlit research dashboard built from the executed Kaggle showcase and paired
sensor experiments. It serves saved predictions, with no training, PyTorch, model
downloads, accounts, or secrets needed at runtime.

## Run

From the repository root, using Python 3.11 or later:

```bash
python -m venv .venv-app
source .venv-app/bin/activate
python -m pip install -r app/requirements.txt
python -m streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`. The tested environment is Python 3.13 on an
Apple M4 Mac. Streamlit and Plotly versions are pinned in `app/requirements.txt`.
The top-right sun/moon icon toggles the full native light or dark theme.
Dark is the default for new visitors; choices use Streamlit's per-browser persistence.
The icon sits beside the main menu in the toolbar flow, leaving room for Community
Cloud's GitHub and Share controls. Its tooltip and accessible label describe the
next appearance. Typography uses Apple's system font stack: San Francisco on Apple
devices, with native system and sans-serif fallbacks elsewhere.
Detailed protocols and data attribution are expandable below the visualizations.
Controls, charts, and details share an open layout with consistent alignment;
subtle dividers replace enclosing cards. Export actions remain available as
compact text buttons with download icons.
See the [design review and references](design-review.md) for the redesign rationale.

The centered workspace is capped at 1,568 CSS pixels on large monitors. Controls
and charts reflow according to the available main-column width, including space
occupied by the sidebar. Narrow screens stack plots and controls while retaining
two-column metric groups. Typography uses a 16 px body, 14 px labels and notes,
13–15 px chart text, and 24–28 px metrics with tabular numerals. Font sizes remain bounded on
ultrawide displays; browser zoom remains available for individual reading needs.

## Views

- **Flood Explorer:** choose a measured wave, one of four reconstruction models,
  and an observation frame. A short introduction explains the task, and metric
  help icons explain RMSE, MAE, and whole-wave error. Compare measured WSE,
  predicted WSE, and absolute error in three maps, or rotate the 3D
  surface, play an optional animation, and download the displayed frame as CSV.
- **Model Comparison:** whole-wave scores, measured/predicted/absolute-error maps,
  and frame RMSE over time. Wave 1 uses held-out times and spatial locations.
  Wave 3 uses frozen wave 1 models with no wave 3 fitting observations.
- **Sensor Ablation:** 16/32/64/128 sensors, mean error and sample SD, and individual
  runs for seeds 42/43/44. Select wave 1 test or wave 3 transfer evaluation.
- **PINN Explorer:** depth and discharge profiles, signed conservation residuals,
  equations, objective, and metrics for the separate synthetic experiment.

The sidebar groups Source code and Kaggle notebook under Project links, with a
short About This Research disclosure covering the Nepal motivation and the
laboratory/simulation data used here.

Measured maps show WSE relative to the initial bed in millimeters. Blank grid cells
remain blank: the app does not invent measurements or interpolate a ground-truth
surface. 3D elevations are exaggerated and have a different vertical unit from the
horizontal axes. Synthetic values and residuals are dimensionless.

The source study is one laboratory configuration, not a Nepal event forecast.
Sensor variation spans only three seeds; it is not uncertainty across flood events.
The PINN result is a single-seed controlled simulation and its depth error is slightly
higher than the data-only network. The dashboard does not expose physics weights
or training controls that would suggest experiments have been rerun.

## Data provenance and regeneration

`reports/dashboard/` contains compact compressed showcase predictions,
source-derived observations, numerical reference states, and residual diagnostics.
The earlier CLI artifacts in `reports/artifacts/measured_predictions.npz` and
`synthetic_predictions.npz` belong to a different experiment run and are not used
as dashboard predictions.

The exporter reuses the showcase notebook's exact inference and physics definitions.
It verifies every whole-wave measured RMSE, the original held-out prediction array,
both simulated prediction errors, and both residual RMS values against the published
showcase CSVs. Its manifest records input and output checksums, the notebook checksum,
training configuration, and reference solver diagnostics. Runtime checksums detect
changed artifacts before the data are served. Models/checkpoints are not bundled
into the serving app.

To regenerate after running the notebook in its full training environment:

```bash
floods run-notebook
python scripts/export_dashboard.py
# If the notebook wrote somewhere else:
python scripts/export_dashboard.py --run outputs/kaggle_notebook
```

Measured observations and derived fields retain the Zaragoza dataset's **CC BY 4.0**
license and attribution. See [data provenance](data_sources.md), the dashboard About
section, and [artifact notes](../reports/dashboard/README.md). Code uses Apache 2.0.

## Checks

```bash
python -m unittest discover -s app/tests -v
```

Checks verify that displayed arrays reproduce published scores, sample variation
matches the three-seed study, downloads use the selected frame and correct units,
and all four Streamlit views run with their interactive controls. They require only
the lightweight dashboard environment. The existing scientific notebook tests remain
separate and require the full notebook environment.

## Streamlit Community Cloud

Deploy from the GitHub repository with **`app/streamlit_app.py`** as the entrypoint
and Python **3.13**. Keep `app/requirements.txt`, `.streamlit/config.toml`, and
`reports/dashboard/` checked in. Community Cloud selects the dependency file next to
the entrypoint, so this app uses the lightweight app dependencies rather than the
full notebook environment. No secrets are required. Deployment has not been performed
by creating these local files.

Official references: [multipage navigation](https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation),
[caching](https://docs.streamlit.io/develop/concepts/architecture/caching),
[app testing](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest),
and [Community Cloud dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

### Theme control maintenance

The header selector uses a trusted [v2 custom component](https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v2.component)
to forward choices to Streamlit 1.65.0's native theme menu. Its DOM selectors and
theme-cache key are version-specific; verify them before updating the pinned
Streamlit dependency. It does not set global server configuration or repaint
widgets with independent CSS colors. Native themes, plots, and browser persistence
remain synchronized. The component cleans up its header element, observers, and
timers when replaced. Check repeated switches, keyboard arrows, navigation,
refresh, system appearance, and phone widths when changing this control.
