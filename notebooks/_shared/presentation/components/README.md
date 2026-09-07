# Notebook UI components

- `editor/`: static markup (`editor.html`), event handling and state rendering
  (`editor.js`), and styles (`editor.css`). Loaded by `_shared.presentation.widgets.LLREditor`.
- `plot_controls/`: dynamic legend markup and selection handling
  (`plot_controls.js`), and styles (`plot_controls.css`). Loaded by
  `_shared.presentation.plot_controls.PlotControls`. Groups are built from Python-provided data,
  so this component has no static HTML template.

Python reads these assets when creating each widget; no build step or web server
is required. Components handle input/output only. Numerical work stays in Python.
