# Shared notebook code

```text
core/
  lfsr.py                     Sparse binary encoding
  channel.py                  BPSK/AWGN batches with per-sample Es/N0
  generator.py                Sparse LFSR generator-column supports
  decoders/
    _adt.py                   Soft-XOR configuration variants
    _softxor.py               Shared soft-XOR dispatch and box-plus
    y/
      avail_softmajvote.py     Compiled codeword decoding; optional histories
    x/
      avail_softmajvote.py Recover x from fixed y LLRs via generator columns
application/
  comparison.py               Run decoders on independent observation copies
  metrics.py                  Diagnostics computed from histories and truth
  marginal_comparison.py      Paired process-pool experiments and SER distributions
presentation/
  widgets.py                  Interactive LLR editor
  plot_controls.py            Custom legend widget
  plots.py                    Plotly figures and comparison views
  marginal_plots.py            Conditional SER histogram, mean and timing table
  components/
    editor/                   HTML, JS, CSS
    plot_controls/            JS, CSS
```

`core` has no dependencies on application or presentation code. `application`
uses core definitions to analyze results. `presentation` renders those results
and handles notebook interaction; its components contain frontend assets only.
