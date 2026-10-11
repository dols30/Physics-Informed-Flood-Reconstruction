"""The four FloodLab AI research views. Every chart uses executed experiment data."""

import json
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from app import charts
from app.data import (
    DATA,
    ROOT,
    MODELS,
    elevation_limits,
    frame_table,
    load_archive,
    load_wave,
    measured_results,
    scores,
    sensor_results,
    summarize_sensors,
)


@st.cache_data(show_spinner=False)
def wave_data(number):
    return load_wave(number)


@st.cache_data(show_spinner=False)
def synthetic_data():
    return load_archive("synthetic.npz")


@st.cache_data(show_spinner=False)
def runs_data():
    return sensor_results()


def heading(title, description, track):
    st.html(
        f'<div class="page-heading"><div><h1>{escape(title)}</h1><p>{escape(description)}</p></div>'
        f'<span class="dataset-tag">{escape(track)}</span></div>'
    )


def plot(fig, label, key=None):
    dark = st.session_state.get("ui_appearance", st.context.theme.type) == "dark"
    color = "#DAE6EC" if dark else "#243746"
    fig.update_layout(font=dict(color=color), title_font_size=15, legend_font_size=13)
    fig.update_xaxes(
        gridcolor="#34444F" if dark else "#E3E9EE", title_font_size=14, tickfont_size=13
    )
    fig.update_yaxes(
        gridcolor="#34444F" if dark else "#E3E9EE", title_font_size=14, tickfont_size=13
    )
    for trace in fig.data:
        if trace.name == "Reference":
            trace.line.color = color
    if any(trace.type == "surface" for trace in fig.data):
        axis = dict(
            backgroundcolor="#1C2934" if dark else "#F0F4F6",
            color=color,
            gridcolor="#34444F" if dark else "#DAE2E7",
        )
        fig.update_scenes(bgcolor="rgba(0,0,0,0)", xaxis=axis, yaxis=axis, zaxis=axis)
    st.plotly_chart(
        fig,
        theme=None,
        width="stretch",
        config={"displaylogo": False, "scrollZoom": False},
        alt=label,
        key=key,
    )


def responsive_columns(widths, *, key):
    """Group chart columns so they can stack when the workspace becomes narrow."""
    with st.container(key=key):
        return st.columns(widths)


def wave_controls(prefix, model=True):
    columns = st.columns([1, 1, 2] if model else [1, 3])
    with columns[0]:
        label = st.selectbox(
            "Laboratory wave",
            ["Wave 1 · held-out test", "Wave 3 · frozen transfer"],
            key=f"{prefix}_wave",
        )
    number = 1 if label.startswith("Wave 1") else 3
    wave = wave_data(number)
    name = None
    if model:
        with columns[1]:
            name = st.selectbox("Reconstruction model", list(MODELS), key=f"{prefix}_model")
    with columns[-1]:
        frame = (
            st.slider(
                "Observation frame",
                1,
                len(wave["time_s"]),
                max(1, len(wave["time_s"]) // 2),
                key=f"{prefix}_frame",
            )
            - 1
        )
        st.caption(f"{wave['time_s'][frame]:.3f} s · {len(wave['xy_m']):,} locations")
    return number, wave, name, frame


def wave_note(number):
    if number == 1:
        st.caption("128 training sensors · seed 42 · held-out locations and times")
    else:
        st.caption(
            "Wave 1 models · zero wave 3 fitting observations · common locations and overlapping times"
        )


def download_frame(wave, name, frame, number):
    st.download_button(
        "Export frame",
        frame_table(wave, name, frame).to_csv(index=False),
        f"floodlab_wave{number}_{MODELS[name]}_frame{frame+1}.csv",
        "text/csv",
        icon=":material/download:",
        type="tertiary",
    )


def explorer():
    heading(
        "Flood Explorer",
        "This dashboard explores how machine learning reconstructs missing floodwater "
        "measurements from limited sensors. Select an experiment, choose a model, and "
        "explore how predicted water levels compare with real laboratory observations.",
        "Zaragoza laboratory data",
    )
    with st.container(key="controls"):
        number, wave, name, frame = wave_controls("explorer")
        wave_note(number)
    ref, pred = wave["reference_m"][frame], wave[MODELS[name]][frame]
    result = scores(ref, pred)
    whole_wave_rmse = scores(wave["reference_m"], wave[MODELS[name]])["RMSE (mm)"]
    with st.container(key="stats"):
        cols = st.columns(4)
        for col, (label, value, explanation) in zip(
            cols,
            [
                (
                    "Frame RMSE",
                    f"{result['RMSE (mm)']:.2f} mm",
                    "Prediction error that penalizes larger mistakes more strongly. "
                    "Measured in millimeters; lower is better.",
                ),
                (
                    "Frame MAE",
                    f"{result['MAE (mm)']:.2f} mm",
                    "Average absolute difference between predicted and measured levels. "
                    "Measured in millimeters; lower is better.",
                ),
                (
                    "Whole-wave RMSE",
                    f"{whole_wave_rmse:.2f} mm",
                    "Prediction error across all evaluated frames, rather than just the "
                    "currently selected frame. Measured in millimeters; lower is better.",
                ),
                ("Observation time", f"{wave['time_s'][frame]:.3f} s", None),
            ],
        ):
            col.metric(label, value, help=explanation)
    limits = elevation_limits(wave)
    with st.container(key="plot-panel"):
        toolbar, options = st.columns([1, 1])
        with toolbar:
            view = st.segmented_control(
                "Visualization",
                ["Elevation maps", "3D surface"],
                default="Elevation maps",
                required=True,
                label_visibility="collapsed",
            )
        with options:
            show = st.checkbox(
                "Training sensors", value=False, disabled=number == 3 or view != "Elevation maps"
            )
        if view == "Elevation maps":
            left, middle, right = responsive_columns(3, key="field-row")
            with left:
                plot(
                    charts.field_map(wave["xy_m"], ref * 1000, "Measured", limits),
                    "Measured WSE at held-out locations",
                )
            with middle:
                plot(
                    charts.field_map(
                        wave["xy_m"],
                        pred * 1000,
                        f"Predicted · {name}",
                        limits,
                        sensors=wave["sensor_xy_m"] if show and number == 1 else None,
                    ),
                    "Saved model reconstruction",
                )
            with right:
                error_mm = np.abs(pred - ref) * 1000
                # Use one error scale across this model's frames, so time comparisons are fair.
                whole_wave_error = np.abs(wave[MODELS[name]] - wave["reference_m"]) * 1000
                error_limits = (0, max(float(whole_wave_error.max()), 1))
                plot(
                    charts.field_map(
                        wave["xy_m"], error_mm, "Absolute error", error_limits, error=True
                    ),
                    "Absolute difference between predicted and measured WSE, in millimeters",
                )
            st.html(
                '<div class="plot-note">Water-surface elevation (mm), relative to initial bed. '
                "Measured and predicted maps share a color scale; absolute error has its own "
                "scale. Blank cells are unobserved.</div>"
            )
        else:
            plot(
                charts.elevation_3d(wave["xy_m"], pred * 1000, name, limits),
                "Rotatable reconstructed WSE surface",
            )
            st.caption("Exaggerated vertical scale · horizontal axes: m · elevation: mm")
    with st.expander("Wave animation"):
        animate = st.checkbox("Load animation", key="animate_wave")
        if animate:
            plot(
                charts.animated_wave(wave, MODELS[name], limits),
                "Animated saved predictions at available observation times",
            )
            st.caption("Animation has its own frame control. Export uses the frame selected above.")
    download_frame(wave, name, frame, number)


def comparison():
    heading(
        "Model Comparison",
        "Reconstruction errors on the same evaluation observations.",
        "Zaragoza laboratory data",
    )
    with st.container(key="controls"):
        number, wave, name, frame = wave_controls("comparison")
        wave_note(number)
    results = measured_results(number)
    left, right = responsive_columns([1, 1.2], key="paired-scores")
    with left:
        st.subheader("Whole-wave RMSE")
        plot(charts.model_bars(results), "All four models ranked by whole-wave RMSE")
    with right:
        st.subheader("Evaluation scores")
        st.dataframe(
            results,
            hide_index=True,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="%.3f") for c in results if c != "Model"
            },
        )
    st.subheader(f"Frame comparison · {wave['time_s'][frame]:.3f} s")
    limits = elevation_limits(wave)
    ref, pred = wave["reference_m"][frame], wave[MODELS[name]][frame]
    error = np.abs(pred - ref) * 1000
    # Keep the error scale fixed across frames for the selected model.
    whole_wave_error = np.abs(wave[MODELS[name]] - wave["reference_m"]) * 1000
    error_limits = (0, max(float(whole_wave_error.max()), 1))
    columns = responsive_columns(3, key="field-row")
    for col, values, title, is_error in zip(
        columns,
        [ref * 1000, pred * 1000, error],
        ["Measured", name, "Absolute error"],
        [False, False, True],
    ):
        with col:
            plot(
                charts.field_map(
                    wave["xy_m"],
                    values,
                    title,
                    error_limits if is_error else limits,
                    error=is_error,
                    height=245,
                ),
                title + " elevation map",
            )
    st.caption("WSE and absolute error: mm · error colors use a separate scale")
    st.subheader("Error over time")
    plot(charts.error_over_time(wave, frame), "Frame RMSE over time for all measured-data models")
    with st.expander("Evaluation protocol"):
        st.markdown(
            "Model selection used wave 1 validation data. Wave 1 scores use held-out locations and times; frozen wave 3 evaluation uses common coordinates and overlapping times. Wave 3 shares the geometry but has an evolved bed."
        )
        st.markdown(
            "WSE is relative to the initial bed. Measured and reconstructed WSE share a fixed color scale; absolute error has a separate scale fixed across this model's frames."
        )
    st.download_button(
        "Export scores",
        results.to_csv(index=False),
        f"floodlab_wave{number}_scores.csv",
        "text/csv",
        icon=":material/download:",
        type="tertiary",
    )


def ablation():
    heading(
        "Sensor Ablation",
        "Reconstruction error as the observation budget changes.",
        "3 seeds · 4 budgets",
    )
    with st.container(key="controls"):
        left, right = st.columns([1.5, 1])
        with left:
            budget = st.segmented_control(
                "Virtual sensor budget", [16, 32, 64, 128], default=32, required=True
            )
        with right:
            split_label = st.selectbox(
                "Evaluation", ["Wave 1 · held-out test", "Wave 3 · frozen transfer"]
            )
    split = "test" if split_label.startswith("Wave 1") else "transfer"
    runs = runs_data()
    runs = runs[runs.Split == split]
    summary = summarize_sensors(runs)
    chosen = summary[summary.Sensors == budget].set_index("Model")
    with st.container(key="stats"):
        cols = st.columns(4)
        for col, name in zip(cols, MODELS):
            row = chosen.loc[name]
            col.metric(name, f"{row['mean']:.2f} mm")
            col.caption(f"± {row['std']:.2f} mm SD")
    left, right = responsive_columns([1.2, 1], key="paired-ablation")
    with left:
        st.subheader("Sensor budget vs. RMSE")
        plot(
            charts.sensor_curves(summary),
            "Sensor budget versus mean RMSE, with sample-standard-deviation bars",
        )
    with right:
        st.subheader(f"Individual runs · {budget} sensors")
        plot(
            charts.seed_points(runs[runs.Sensors == budget]),
            "Individual RMSE scores for three seeds",
        )
    st.caption("Mean ± sample SD · seeds 42, 43, 44")
    with st.expander("Study protocol"):
        st.markdown(
            "Nested sensor sets use fixed unseen locations within each seed. MLP runs use 1,500 steps. Error bars show sample SD over three seeds, not confidence intervals or uncertainty across flood events."
        )
        st.markdown(
            "This paired experiment uses a different spatial split from the showcase. Its 128-sensor result is separate; the small MLP advantage at 16 sensors does not establish statistical superiority."
        )
    with st.expander("Inspect individual runs"):
        cols = ["Sensors", "Seed", "Model", "RMSE (mm)", "MAE (mm)", "R²"]
        st.dataframe(runs.loc[runs.Sensors == budget, cols], hide_index=True, width="stretch")
    st.download_button(
        "Export runs",
        runs.to_csv(index=False),
        f"floodlab_sensor_{split}_runs.csv",
        "text/csv",
        icon=":material/download:",
        type="tertiary",
    )


def pinn():
    heading(
        "PINN Explorer",
        "Data-only and physics-informed shallow-water predictions.",
        "Synthetic · dimensionless",
    )
    data = synthetic_data()
    results = pd.read_csv(ROOT / "reports/notebook_results/synthetic_results.csv").set_index(
        "Model"
    )
    reduction = 100 * (
        1 - results.loc["PINN", "Discharge RMSE"] / results.loc["Data-only NN", "Discharge RMSE"]
    )
    with st.container(key="controls"):
        frame = st.slider("Synthetic observation frame", 1, len(data["time"]), 4) - 1
        st.caption(f"t* = {data['time'][frame]:.5f} · held-out frame · seed 42")
    with st.container(key="stats"):
        cols = st.columns(3)
        cols[0].metric("Discharge error reduction", f"{reduction:.1f}%")
        cols[1].metric(
            "Data-only discharge RMSE", f"{results.loc['Data-only NN','Discharge RMSE']:.5f}"
        )
        cols[2].metric("PINN discharge RMSE", f"{results.loc['PINN','Discharge RMSE']:.5f}")
    left, right = responsive_columns(2, key="paired-profiles")
    with left:
        st.subheader("Water depth h*")
        plot(
            charts.synthetic_lines(data, frame, 0), "Reference, data-only, and PINN depth profiles"
        )
    with right:
        st.subheader("Discharge q*")
        plot(
            charts.synthetic_lines(data, frame, 1),
            "Reference, data-only, and PINN discharge profiles",
        )
    st.caption(
        "Single-seed benchmark · discharge error improves; PINN depth RMSE is slightly higher"
    )
    st.subheader("Physics residuals")
    component_label = st.segmented_control(
        "Physics residual",
        ["Continuity", "Momentum"],
        default="Continuity",
        required=True,
        label_visibility="collapsed",
    )
    component = 0 if component_label == "Continuity" else 1
    limit = max(
        float(np.abs(data[key + "_residual"][:, component]).max()) for key in ("data_only", "pinn")
    )
    left, right = responsive_columns(2, key="paired-residuals")
    with left:
        st.markdown("**Data-only NN**")
        plot(
            charts.residual_map(data, "data_only", component, limit),
            "Data-only conservation residuals at diagnostic points",
        )
    with right:
        st.markdown("**PINN**")
        plot(
            charts.residual_map(data, "pinn", component, limit),
            "PINN conservation residuals at the same diagnostic points",
        )
    st.caption("Signed residuals · 1,024 matched diagnostic points · shared color scale")
    with st.expander("Full benchmark scores"):
        st.dataframe(
            results.reset_index().drop(columns="Selected step"),
            hide_index=True,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(format="%.5f")
                for c in results
                if c != "Selected step"
            },
        )
    with st.expander("Shallow-water equations and training objective"):
        st.latex(r"h_t + q_x = 0, \qquad q_t + \left(\frac{q^2}{h}+\frac{h^2}{2}\right)_x = 0")
        st.latex(
            r"\mathcal{L}=\mathcal{L}_{obs}+\mathcal{L}_{initial}+\mathcal{L}_{boundary}+\lambda\mathcal{L}_{PDE}"
        )
        st.markdown(
            "The data-only NN uses λ = 0; the PINN ramps λ up to 0.01 over 200 steps. Interior observations provide depth only; both networks receive initial and boundary depth and discharge. Checkpoints are selected by validation depth RMSE."
        )
        st.markdown(
            "The numerical reference uses a conservative Rusanov solver on a flat, fixed, frictionless wet bed with dimensionless gravity g* = 1. These are simulated values, separate from the laboratory measurements."
        )
        st.markdown(
            "Both networks use seed 42, matched initial weights and observations, and 1,500 steps. Residual RMS is a conservation diagnostic, separate from held-out prediction RMSE."
        )
    st.download_button(
        "Export metrics",
        results.reset_index().to_csv(index=False),
        "floodlab_synthetic_metrics.csv",
        "text/csv",
        icon=":material/download:",
        type="tertiary",
    )


def footer():
    with st.container(key="footer"), st.expander("About this research & data"):
        st.markdown(
            "I am from Nepal. Seeing the effects of Himalayan glacial lake outburst floods motivated me to explore how machine learning can help us understand floodwater movement. **Dol Raj Bashyal**"
        )
        st.markdown(
            "The measured experiments use the Zaragoza O4 laboratory dataset in Spain, with WSE relative to the initial bed. This research demo evaluates reconstruction and same-geometry wave transfer; it is not an operational Nepal flood forecast. The PINN track is a separate controlled simulation."
        )
        st.markdown(
            "Laboratory observations: Segovia-Burillo, Martínez-Aranda, Morales-Hernández, Fernández-Pato, and García-Navarro · [Zenodo release 17777387](https://zenodo.org/records/17777387) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes include unit conversion, masking, subsampling, and derived predictions."
        )
        st.markdown(
            "[GitHub repository](https://github.com/dols30/Physics-Informed-Flood-Reconstruction) · [Kaggle notebook](https://www.kaggle.com/code/bashcode223/physics-informed-flood-reconstruction-ml-vs-pinn)"
        )
        manifest = json.loads((DATA / "manifest.json").read_text())
        st.download_button(
            "Export provenance",
            json.dumps(manifest, indent=2),
            "floodlab_manifest.json",
            "application/json",
            icon=":material/download:",
            type="tertiary",
        )
