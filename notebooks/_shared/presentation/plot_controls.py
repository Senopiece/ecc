"""Selection-only frontend for the Python-owned comparison plots."""

from pathlib import Path

import anywidget
import traitlets as tl


class PlotControls(anywidget.AnyWidget):
    groups = tl.List().tag(sync=True)
    selected = tl.Dict().tag(sync=True)
    stylesheet = tl.Unicode().tag(sync=True)

    def __init__(self, groups, selected):
        assets = Path(__file__).parent / "components" / "plot_controls"
        super().__init__(
            _esm=(assets / "plot_controls.js").read_text(encoding="utf-8"),
            stylesheet=(assets / "plot_controls.css").read_text(encoding="utf-8"),
            groups=groups,
            selected=selected,
        )
        self.layout.margin = "0"
        self.layout.width = "100%"
