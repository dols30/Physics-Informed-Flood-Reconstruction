"""Interactive figures with explicit units and consistent comparison scales."""

import numpy as np
import plotly.graph_objects as go

from app.data import MODELS

COLORS = {
    "Interpolation": "#197268",
    "Random Forest": "#678CC4",
    "MLP": "#BA683C",
    "Polynomial Ridge": "#929F73",
    "Reference": "#18372F",
    "Data-only NN": "#BA683C",
    "PINN": "#197268",
}
WATER = [[0, "#123B58"], [0.25, "#266C91"], [0.55, "#53B1B0"], [0.8, "#B4DAB6"], [1, "#F3ECA2"]]
ERROR = [[0, "#FAF8F0"], [0.35, "#EFC998"], [0.7, "#C56D43"], [1, "#852F28"]]


def style(fig, height=335):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=13, color="#243746"),
        margin=dict(l=45, r=25, t=60, b=45),
        legend=dict(orientation="h", y=1.03, yanchor="bottom", x=0, font=dict(size=13)),
        hovermode="closest",
    )
    fig.update_xaxes(
        gridcolor="#DFE5DB", zeroline=False, title_font_size=14, tickfont_size=13, automargin=True
    )
    fig.update_yaxes(
        gridcolor="#DFE5DB", zeroline=False, title_font_size=14, tickfont_size=13, automargin=True
    )
    return fig


def grid(xy, values):
    """Place available observations on their original grid; leave missing cells blank."""
    x, xi = np.unique(np.round(xy[:, 0], 6), return_inverse=True)
    y, yi = np.unique(np.round(xy[:, 1], 6), return_inverse=True)
    z = np.full((len(y), len(x)), np.nan)
    z[yi, xi] = values
    return x, y, z


def field_map(xy, values_mm, title, limits, *, error=False, sensors=None, height=250):
    x, y, z = grid(xy, values_mm)
    fig = go.Figure(
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            zmin=limits[0],
            zmax=limits[1],
            colorscale=ERROR if error else WATER,
            connectgaps=False,
            colorbar=dict(title="mm", thickness=10, len=0.7),
            hoverongaps=False,
            hovertemplate="x %{x:.3f} m · y %{y:.3f} m<br>%{z:.2f} mm<extra></extra>",
        )
    )
    if sensors is not None and len(sensors):
        fig.add_trace(
            go.Scatter(
                x=sensors[:, 0],
                y=sensors[:, 1],
                mode="markers",
                name="Training sensors",
                showlegend=False,
                marker=dict(size=5, symbol="circle-open", color="#142D27", line=dict(width=1)),
                hovertemplate="Wave 1 training sensor<extra></extra>",
            )
        )
    style(fig, height)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)), margin=dict(l=45, r=25, t=40, b=45)
    )
    fig.update_xaxes(title="x (m)", range=[xy[:, 0].min() - 0.025, xy[:, 0].max() + 0.025])
    fig.update_yaxes(
        title="y (m)",
        scaleanchor="x",
        scaleratio=1,
        range=[xy[:, 1].min() - 0.025, xy[:, 1].max() + 0.025],
        constrain="domain",
    )
    return fig


def elevation_3d(xy, values_mm, title, limits):
    x, y, z = grid(xy, values_mm)
    fig = go.Figure(
        go.Surface(
            x=x,
            y=y,
            z=z,
            cmin=limits[0],
            cmax=limits[1],
            colorscale=WATER,
            connectgaps=False,
            colorbar=dict(title="WSE (mm)", thickness=12),
            hovertemplate="x %{x:.3f} m · y %{y:.3f} m<br>WSE %{z:.2f} mm<extra></extra>",
        )
    )
    style(fig, 470)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="WSE (mm)",
            zaxis=dict(range=list(limits)),
            aspectmode="manual",
            aspectratio=dict(x=2, y=0.65, z=0.6),
            camera=dict(eye=dict(x=1.5, y=-1.7, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=35, b=0),
    )
    return fig


def animated_wave(wave, key, limits):
    fig = field_map(
        wave["xy_m"],
        wave[key][0] * 1000,
        "Saved reconstruction · t = %.3f s" % wave["time_s"][0],
        limits,
    )
    fig.frames = [
        go.Frame(
            name=str(i),
            data=[go.Heatmap(z=grid(wave["xy_m"], row * 1000)[2])],
            layout=go.Layout(title=f"Saved reconstruction · t = {t:.3f} s"),
        )
        for i, (row, t) in enumerate(zip(wave[key], wave["time_s"]))
    ]
    fig.update_layout(
        height=450,
        margin=dict(b=100),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0,
                y=-0.18,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=130, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                x=0.25,
                len=0.75,
                y=-0.17,
                currentvalue=dict(prefix="Frame "),
                steps=[
                    dict(
                        label=str(i + 1),
                        method="animate",
                        args=[
                            [str(i)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for i in range(len(fig.frames))
                ],
            )
        ],
    )
    return fig


def model_bars(results, metric="RMSE (mm)"):
    ordered = results.sort_values(metric, ascending=False)
    fig = go.Figure(
        go.Bar(
            x=ordered[metric],
            y=ordered.Model,
            orientation="h",
            marker_color=[COLORS[m] for m in ordered.Model],
            text=[f"{x:.2f}" for x in ordered[metric]],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.3f} mm<extra></extra>",
        )
    )
    style(fig, 260)
    fig.update_layout(margin=dict(l=125, r=20, t=20, b=40))
    fig.update_xaxes(title=metric, range=[0, ordered[metric].max() * 1.2])
    fig.update_yaxes(showgrid=False)
    return fig


def error_over_time(wave, frame):
    fig = go.Figure()
    for name, key in MODELS.items():
        rmse = np.sqrt(np.mean(((wave[key] - wave["reference_m"]) * 1000) ** 2, axis=1))
        fig.add_trace(
            go.Scatter(
                x=wave["time_s"],
                y=rmse,
                name=name,
                mode="lines",
                line=dict(color=COLORS[name], width=2),
                hovertemplate="t %{x:.3f} s<br>RMSE %{y:.2f} mm<extra>%{fullData.name}</extra>",
            )
        )
    style(fig, 280)
    fig.add_vline(x=float(wave["time_s"][frame]), line_dash="dot", line_color="#193831")
    fig.update_xaxes(title="Observation time (s)")
    fig.update_yaxes(title="Frame RMSE (mm)", rangemode="tozero")
    return fig


def sensor_curves(summary):
    fig = go.Figure()
    for name in MODELS:
        rows = summary[summary.Model == name].sort_values("Sensors")
        fig.add_trace(
            go.Scatter(
                x=rows.Sensors,
                y=rows["mean"],
                mode="lines+markers",
                name=name,
                line=dict(color=COLORS[name], width=2),
                error_y=dict(type="data", array=rows["std"], visible=True),
                hovertemplate="%{x} sensors<br>Mean RMSE %{y:.2f} mm<extra>%{fullData.name}</extra>",
            )
        )
    style(fig, 390)
    fig.update_xaxes(
        title="Virtual sensors", type="log", tickvals=[16, 32, 64, 128], ticktext=[16, 32, 64, 128]
    )
    fig.update_yaxes(title="RMSE (mm)", rangemode="tozero")
    return fig


def seed_points(runs):
    fig = go.Figure()
    for seed, symbol in zip(sorted(runs.Seed.unique()), ["circle", "diamond", "square"]):
        rows = runs[runs.Seed == seed].set_index("Model").reindex(MODELS)
        fig.add_trace(
            go.Scatter(
                x=rows.index,
                y=rows["RMSE (mm)"],
                mode="markers",
                name=f"Seed {seed}",
                marker=dict(size=10, symbol=symbol, color=[COLORS[m] for m in rows.index]),
                hovertemplate="%{x}<br>RMSE %{y:.2f} mm<extra>%{fullData.name}</extra>",
            )
        )
    style(fig, 390)
    fig.update_yaxes(title="RMSE (mm)", rangemode="tozero")
    return fig


def synthetic_lines(data, frame, component):
    fig = go.Figure()
    for name, key in (("Reference", "reference"), ("Data-only NN", "data_only"), ("PINN", "pinn")):
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data[key][frame, :, component],
                name=name,
                mode="lines",
                line=dict(
                    color=COLORS[name], width=2.5, dash="solid" if key != "data_only" else "dash"
                ),
            )
        )
    style(fig, 350)
    label = "Depth h*" if component == 0 else "Discharge q*"
    fig.update_xaxes(title="Position x*")
    fig.update_yaxes(title=label)
    return fig


def residual_map(data, key, component, limit):
    points = data["diagnostic_xt"]
    fig = go.Figure(
        go.Scattergl(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            marker=dict(
                size=6,
                color=data[key + "_residual"][:, component],
                colorscale="RdBu",
                cmin=-limit,
                cmax=limit,
                cmid=0,
                colorbar=dict(title="Residual", thickness=10),
            ),
            hovertemplate="x* %{x:.3f} · t* %{y:.3f}<br>Residual %{marker.color:.3f}<extra></extra>",
        )
    )
    style(fig, 310)
    fig.update_xaxes(title="Position x*")
    fig.update_yaxes(title="Time t*")
    return fig
