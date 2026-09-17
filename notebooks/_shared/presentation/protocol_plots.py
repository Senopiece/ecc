"""Pairwise protocol success heatmaps with swappable axes."""

import numpy as np
import plotly.graph_objects as go

from ..application.protocol_experiment import success_slices
from .plots import ComparisonView, _style


class ProtocolView(ComparisonView):
    def __init__(
        self,
        table,
        *,
        axes=("Overhead", "SNR"),
        overhead_range=(0.1, 0.5),
        snr_db_range=(-3, 3),
        bins=(24, 24),
    ):
        if table.empty:
            raise ValueError("Run some experiments before plotting")
        overhead_edges = np.linspace(*overhead_range, bins[0] + 1)
        snr_edges = np.linspace(*snr_db_range, bins[1] + 1)
        names = list(table["Proto"].unique())
        sizes = sorted(table["K"].unique())
        if len(axes) != 2 or len(set(axes)) != 2 or not set(axes) <= {"K", "Overhead", "SNR"}:
            raise ValueError("Choose two distinct axes from K, Overhead, SNR")
        self._axes = tuple(axes)
        k_edges = np.r_[sizes[0] - 0.5, (np.asarray(sizes[:-1]) + sizes[1:]) / 2, sizes[-1] + 0.5]
        edges = {"K": k_edges, "Overhead": overhead_edges, "SNR": snr_edges}
        self._coordinates = {
            "K": list(map(str, sizes)),
            "Overhead": overhead_edges,
            "SNR": snr_edges,
        }
        self._filter_axis = next(name for name in edges if name not in axes)
        filter_edges = edges[self._filter_axis]
        if self._filter_axis == "K":
            options = list(map(str, sizes))
        else:
            options = [
                f"[{lo:.6g}, {hi:.6g}{']' if i == len(filter_edges) - 2 else ')'}"
                for i, (lo, hi) in enumerate(zip(filter_edges[:-1], filter_edges[1:], strict=True))
            ]
        self._grids = {}
        for name in names:
            slices = success_slices(
                table.loc[table["Proto"] == name],
                axes=axes,
                filter_axis=self._filter_axis,
                edges=edges,
            )
            self._grids.update(
                {
                    (name, option): grid
                    for option, grid in zip(["ALL", *options], slices, strict=True)
                }
            )
        figure = go.Figure(
            go.Heatmap(
                zmin=0,
                zmax=1,
                colorscale=[
                    [0, "#252b3a"],
                    [0.25, "#315975"],
                    [0.5, "#278e9d"],
                    [0.75, "#63bfa1"],
                    [1, "#d5e9ae"],
                ],
                colorbar=dict(title="p(succ)", thickness=12, outlinewidth=0, len=0.9),
                hoverongaps=False,
            )
        )
        _style(figure, "Message recovery probability", 520)
        figure.update_layout(xaxis_title="Overhead", yaxis_title="SNR · Es/N0 (dB)")
        groups = [
            dict(id="decoders", label="Protocols", options=names, multiple=False),
            dict(id="axes", label="Axes", kind="toggle", options=["Swap X/Y"]),
        ]
        groups.append(
            dict(
                id=self._filter_axis,
                label=self._filter_axis,
                kind="slider",
                options=options,
                all_label=f"All {self._filter_axis}",
            )
        )
        super().__init__(
            figure,
            groups,
            {"decoders": [names[0]], self._filter_axis: ["ALL"], "axes": []},
            [],
            "protocol",
        )
        self.chart.update_layout(margin_r=100)

    def _selection_changed(self, change=None):
        selected = self.controls.selected
        name = selected["decoders"][0]
        size = selected[self._filter_axis][0]
        probability, count = self._grids[name, size]
        swapped = bool(selected.get("axes"))
        x, y = self._axes[::-1] if swapped else self._axes
        labels = {"K": "K (message bits)", "Overhead": "Overhead", "SNR": "SNR: Es/N0 (dB)"}
        with self.chart.batch_update():
            self.chart.data[0].update(
                x=self._coordinates[x],
                y=self._coordinates[y],
                z=probability.T if swapped else probability,
                customdata=count.T if swapped else count,
                hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y}}"
                "<br>p(succ)=%{z:.3f}<br>Trials=%{customdata:.0f}<extra></extra>",
            )
            self.chart.update_layout(title=f"Message recovery: {x} vs {y}")
            for name, update in ((x, self.chart.update_xaxes), (y, self.chart.update_yaxes)):
                update(
                    title_text=labels[name],
                    type="category" if name == "K" else "linear",
                    categoryorder="array",
                    categoryarray=self._coordinates["K"],
                    autorange=True,
                )


def success(table, **kwargs):
    return ProtocolView(table, **kwargs)
