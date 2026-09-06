"""Plotly views of completed results. No editor observers or decoder calls."""

import numpy as np
import plotly.graph_objects as go

PLOT_CONFIG = dict(scrollZoom=True, displaylogo=False, responsive=True)


def _style(figure, title, height):
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="var(--vscode-font-family, sans-serif)", size=12),
        margin=dict(l=65, r=30, t=65, b=50),
        hovermode="closest",
        dragmode="zoom",
        title=title,
        height=height,
    )
    grid = dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.3)")
    figure.update_xaxes(**grid)
    figure.update_yaxes(**grid)
    return figure


def metrics_figure(result):
    figure = go.Figure()
    t = np.arange(len(result.history))
    for name, values in result.scores.items():
        figure.add_trace(go.Scatter(x=t, y=values, name=name, mode="lines"))
    figure.update_layout(xaxis_title="Iteration", yaxis_title="Metric value")
    return _style(figure, "Iteration diagnostics", 460)


def bits_figure(result, bits=None):
    """All N traces by default; select a subset explicitly for large experiments."""
    figure = go.Figure()
    t = np.arange(len(result.history))
    indices = range(len(result.truth)) if bits is None else tuple(bits)
    for i in indices:
        if not isinstance(i, (int, np.integer)) or not 0 <= i < len(result.truth):
            raise ValueError("Bit index outside the decoded word")
        signed = result.history[:, i] * (1 if result.truth[i] == 0 else -1)
        figure.add_trace(go.Scattergl(x=t, y=signed, name=f"bit {i}", mode="lines"))
    figure.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Signed LLR",
        legend_title="Bit",
    )
    figure.add_hline(y=0, line_width=1, line_color="#657384", line_dash="dot")
    return _style(figure, "Per-bit signed LLR", 460)


def evolution_figure(result):
    # Only the visualization allocates a color matrix. Diagnostics never create
    # (T,N,w) neighborhood arrays or keep a duplicate truth-aligned history.
    colors = np.empty_like(result.history)
    np.abs(result.history, out=colors)
    colors *= -0.5
    np.expm1(colors, out=colors)
    colors *= -1
    for i, bit in enumerate(result.truth):
        colors[:, i] *= np.sign(result.history[:, i]) * (1 if bit == 0 else -1)
    figure = go.Figure(
        go.Heatmap(
            x=np.arange(len(result.truth)),
            y=np.arange(len(result.history)),
            z=colors,
            customdata=result.history,
            zmin=-1,
            zmax=1,
            colorscale=[[0, "rgb(225,68,68)"], [0.5, "white"], [1, "rgb(45,170,92)"]],
            colorbar=dict(
                title="Signed strength", tickvals=[-1, 0, 1], ticktext=["Wrong", "Zero", "Correct"]
            ),
            hovertemplate="t=%{y}, bit=%{x}<br>LLR=%{customdata:.6g}<extra></extra>",
            xgap=1,
            ygap=1,
        )
    )
    figure.update_layout(xaxis_title="Bit index", yaxis_title="Iteration")
    figure.update_yaxes(autorange="reversed")
    return _style(figure, "Evolution of y' - red: wrong, white: zero, green: correct", 540)
