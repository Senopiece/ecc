"""Plotly views of completed results. No editor observers or decoder calls."""

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go

from ..application.metrics import metrics as compute_metrics
from ..application.metrics import validate_inputs
from ..core.lfsr import DEFAULT_TERMS
from .plot_controls import PlotControls

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
    figure.update_yaxes(**grid, ticklabelstandoff=8, title_standoff=16, automargin=True)
    return figure


def metrics(history, true, terms=DEFAULT_TERMS):
    scores = compute_metrics(history, true, terms)
    figure = go.Figure()
    t = np.arange(len(history))
    for name, values in scores.items():
        figure.add_trace(go.Scatter(x=t, y=values, name=name, mode="lines"))
    figure.update_layout(xaxis_title="Iteration", yaxis_title="Metric value")
    return _style(figure, "Iteration diagnostics", 460)


def bits(history, true, indices=None):
    """All N traces by default; select a subset explicitly for large experiments."""
    history, true = validate_inputs(history, true)
    figure = go.Figure()
    t = np.arange(len(history))
    indices = range(len(true)) if indices is None else tuple(indices)
    for i in indices:
        if not isinstance(i, (int, np.integer)) or not 0 <= i < len(true):
            raise ValueError("Bit index outside the decoded word")
        signed = history[:, i] * (1 if true[i] == 0 else -1)
        figure.add_trace(go.Scattergl(x=t, y=signed, name=f"bit {i}", mode="lines"))
    figure.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Signed LLR",
        legend_title="Bit",
    )
    figure.add_hline(y=0, line_width=1, line_color="#657384", line_dash="dot")
    return _style(figure, "Per-bit signed LLR", 460)


def evolution(history, true):
    history, true = validate_inputs(history, true)
    # Only the visualization allocates a color matrix. Diagnostics never create
    # (T,N,w) neighborhood arrays or keep a duplicate truth-aligned history.
    colors = np.empty_like(history)
    np.abs(history, out=colors)
    colors *= -0.5
    np.expm1(colors, out=colors)
    colors *= -1
    for i, bit in enumerate(true):
        colors[:, i] *= np.sign(history[:, i]) * (1 if bit == 0 else -1)
    figure = go.Figure(
        go.Heatmap(
            x=np.arange(len(true)),
            y=np.arange(len(history)),
            z=colors,
            customdata=history,
            zmin=-1,
            zmax=1,
            colorscale=[[0, "rgb(225,68,68)"], [0.5, "white"], [1, "rgb(45,170,92)"]],
            colorbar=dict(
                tickvals=[-1, 0, 1],
                ticktext=["Wrong", "Zero", "Correct"],
                thickness=12,
                outlinewidth=0,
                tickfont=dict(size=11, color="#cccccc"),
                len=0.92,
                x=1,
                xanchor="left",
                xpad=12,
                ticks="outside",
                ticklen=4,
            ),
            hovertemplate="t=%{y}, bit=%{x}<br>LLR=%{customdata:.6g}<extra></extra>",
            xgap=0.5,
            ygap=0.5,
        )
    )
    figure.update_layout(xaxis_title="Bit index", yaxis_title="Iteration")
    figure.update_yaxes(autorange="reversed")
    _style(figure, "LLR evolution", max(300, min(480, 105 + 12 * len(history))))
    figure.update_layout(
        margin_r=100,
        hoverlabel=dict(bgcolor="#252526", bordercolor="#454545", font_size=12),
    )
    figure.update_xaxes(showgrid=False, zeroline=False, tickmode="linear", dtick=1)
    figure.update_yaxes(showgrid=False, zeroline=False)
    return figure


class ComparisonView(widgets.GridBox):
    """One chart with independent selector groups; updates only trace visibility."""

    def __init__(self, figure, groups, selected, trace_keys, kind):
        self.controls = PlotControls(groups, selected)
        figure.update_layout(
            autosize=True,
            width=None,
            paper_bgcolor="#1f1f1f",
            plot_bgcolor="#1f1f1f",
            margin=dict(l=75, r=100 if kind == "evolution" else 30, t=55, b=50),
            showlegend=False,
        )
        self.chart = go.FigureWidget(figure)
        # Assign a new dict so traitlets synchronizes configuration to the browser.
        self.chart._config = {**self.chart._config, **PLOT_CONFIG}
        self.controls.layout.max_height = f"{figure.layout.height}px"
        self.controls.layout.overflow = "hidden"
        self._trace_keys = trace_keys
        self._kind = kind
        super().__init__(
            [self.chart, self.controls],
            layout=widgets.Layout(
                width="100%",
                min_width="0",
                margin="0",
                padding="0",
                border="0",
                grid_template_columns="minmax(0, 1fr) minmax(200px, 30%)",
                align_items="flex-start",
            ),
        )
        self.add_class("pr-comparison")
        self.controls.observe(self._selection_changed, names="selected")
        self.controls.on_msg(self._resize_chart)
        self._selection_connected = True
        self._selection_changed()

    def _resize_chart(self, widget, event, buffers):
        width = event.get("width")
        if event.get("type") == "chart_resize" and isinstance(width, int) and width >= 10:
            # Use the actual grid column width after mounting, not Plotly's
            # provisional size while the notebook output is still being laid out.
            self.chart.update_layout(width=width, autosize=False)

    def _selection_changed(self, change=None):
        selected = self.controls.selected
        decoders = set(selected.get("decoders", []))
        series = set(selected.get("series", []))
        with self.chart.batch_update():
            for trace, (decoder, name) in zip(self.chart.data, self._trace_keys, strict=True):
                trace.visible = decoder in decoders and (
                    self._kind == "evolution" or name in series
                )
            # Fit the newly selected data, including when switching away from large margins.
            self.chart.update_xaxes(autorange=True)
            self.chart.update_yaxes(autorange="reversed" if self._kind == "evolution" else True)

    def close(self):
        if getattr(self, "_selection_connected", False):
            self._selection_connected = False
            self.controls.unobserve(self._selection_changed, names="selected")
            self.controls.on_msg(self._resize_chart, remove=True)
            self.controls.close()
            self.chart.close()
        super().close()


def compare(histories, true, *, kind="metrics", terms=DEFAULT_TERMS, previous=None):
    """Compare stored histories on one plot, with independent decoder/series filters.

    Metrics and bits allow multiple decoders; evolution selects exactly one.
    Nothing here calls a decoder. All numerical curves are prepared once.
    """
    if not histories:
        raise ValueError("Supply at least one decoder history")
    if kind not in ("metrics", "bits", "evolution"):
        raise ValueError("kind must be metrics, bits, or evolution")
    names = list(histories)
    from plotly.colors import qualitative

    palette = qualitative.Plotly
    dashes = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")
    sample_dashes = ("", "8,8", "2,4", "8,4,2,4", "12,6", "12,4,2,4")
    figure = None
    trace_keys = []
    series = []
    for decoder_index, (decoder, history) in enumerate(histories.items()):
        source = (
            metrics(history, true, terms=terms)
            if kind == "metrics"
            else bits(history, true)
            if kind == "bits"
            else evolution(history, true)
        )
        if figure is None:
            figure = go.Figure(layout=source.layout)
            series = [trace.name for trace in source.data] if kind != "evolution" else []
        for index, trace in enumerate(source.data):
            name = trace.name
            trace_keys.append((decoder, name))
            if kind != "evolution":
                trace.update(
                    name=f"{name} / {decoder}",
                    line=dict(
                        color=palette[index % len(palette)],
                        dash=dashes[decoder_index % len(dashes)],
                    ),
                    hovertemplate="Iteration %{x}<br>%{y:.6g}<extra>%{fullData.name}</extra>",
                )
            figure.add_trace(trace)
    groups = [
        dict(
            id="decoders",
            label="Decoders",
            options=names,
            multiple=kind != "evolution",
            samples=[
                dict(color="#cccccc", dash=sample_dashes[i % len(sample_dashes)])
                for i in range(len(names))
            ]
            if kind != "evolution"
            else [],
        )
    ]
    selected = dict(decoders=names if kind == "metrics" else names[:1])
    if kind != "evolution":
        groups.append(
            dict(
                id="series",
                label="Metrics" if kind == "metrics" else "Bits",
                options=series,
                multiple=True,
                samples=[dict(color=palette[i % len(palette)]) for i in range(len(series))],
            )
        )
        selected["series"] = series[:3] if kind == "metrics" else series
    view = ComparisonView(figure, groups, selected, trace_keys, kind)
    if previous is not None:
        previous.close()
    return view
