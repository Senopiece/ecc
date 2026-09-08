"""Timing table and conditional SER distributions for paired AWGN experiments."""

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from ..application.marginal_comparison import conditional_ser
from .plots import ComparisonView, _style


def timing_table(experiment):
    frame = pd.DataFrame(
        [
            {"Decoder": name, "Mean decode time (ns/sample)": values["ns_per_sample"]}
            for name, values in experiment["decoders"].items()
        ]
    )
    return (
        frame.style.hide(axis="index")
        .format({"Mean decode time (ns/sample)": "{:,.1f}"})
        .set_properties(
            **{
                "background-color": "#1f1f1f",
                "color": "#cccccc",
                "padding": "10px 16px",
                "border-bottom": "1px solid #353535",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#252526"),
                        ("color", "#cccccc"),
                        ("padding", "10px 16px"),
                        ("text-align", "left"),
                    ],
                },
                {
                    "selector": "",
                    "props": [
                        ("border-collapse", "collapse"),
                        ("font-family", "system-ui"),
                    ],
                },
            ]
        )
    )


def distributions(experiment, *, bins=30, ser_bins=64, previous=None):
    """Mean SER and column-normalized histogram cells.

    Linear opacity encodes cell probability on a fixed [0,1] scale for all
    decoders. Empty columns are gaps; zero-probability cells are transparent.
    """
    figure = go.Figure()
    keys = []
    names = list(experiment["decoders"])
    colors = qualitative.Plotly
    for index, (name, values) in enumerate(experiment["decoders"].items()):
        color = colors[index % len(colors)]
        rgb = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
        summary = conditional_ser(
            experiment["snr_db"], values["ser"], experiment["snr_db_range"], bins, ser_bins
        )
        figure.add_trace(
            go.Heatmap(
                x=summary["snr_edges"],
                y=summary["ser_edges"],
                z=summary["probability"],
                customdata=summary["cell_count"],
                zmin=0,
                zmax=1,
                zsmooth=False,
                showscale=False,
                hoverongaps=False,
                colorscale=[
                    [0, f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0)"],
                    [1, f"rgba({rgb[0]},{rgb[1]},{rgb[2]},1)"],
                ],
                name=name,
                hovertemplate="SNR %{x:.3f} dB<br>SER cell %{y:.4f}"
                "<br>P(cell | SNR bin) %{z:.2%}<br>Count %{customdata:.0f}"
                "<extra>%{fullData.name}</extra>",
            )
        )
        keys.append((name, "Distribution"))
        figure.add_trace(
            go.Scatter(
                x=summary["snr"],
                y=summary["mean"],
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                customdata=summary["count"],
                hovertemplate="SNR %{x:.2f} dB<br>Mean SER %{y:.5f}<br>%{customdata} samples"
                "<extra>%{fullData.name}</extra>",
            )
        )
        keys.append((name, "Mean SER"))
    _style(figure, "SER versus SNR", 500)
    figure.update_layout(xaxis_title="SNR · Es/N0 (dB)", yaxis_title="Sign error rate")
    groups = [
        dict(
            id="decoders",
            label="Decoders",
            options=names,
            multiple=True,
            samples=[dict(color=colors[i % len(colors)]) for i in range(len(names))],
        ),
        dict(
            id="series",
            label="Layers",
            options=["Distribution", "Mean SER"],
            multiple=True,
        ),
    ]
    view = ComparisonView(
        figure,
        groups,
        dict(decoders=names, series=["Distribution", "Mean SER"]),
        keys,
        "marginal",
    )
    if previous is not None:
        previous.close()
    return view
