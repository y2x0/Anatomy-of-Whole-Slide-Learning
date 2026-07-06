# Sequence Representations

A sequence representation treats a slide as an ordered list of patch embeddings:

```math
\mathcal{X}_i=(h_{i1},h_{i2},\ldots,h_{in_i}).
```

The order may come from:

```text
raster scan
space-filling curve
learned ordering
region traversal
Mamba/SSM reordering
attention-based ordering
```

The defining property is that order matters:

```math
f(h_1,h_2,\ldots,h_n)
\ne
f(h_{\pi(1)},h_{\pi(2)},\ldots,h_{\pi(n)})
```

in general.

## Files

- `01_slide_as_sequence.md`: ordering, positional structure, and symmetry.
- `02_sequence_operators.md`: recurrent, transformer, and state-space maps.
- `03_ordering_and_geometry.md`: how order encodes or distorts WSI geometry.
- `04_failure_modes.md`: arbitrary order, long-range loss, and order shortcuts.
