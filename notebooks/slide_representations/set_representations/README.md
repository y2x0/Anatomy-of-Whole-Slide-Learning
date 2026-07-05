# Set Representations

A set representation treats a slide as an unordered finite collection of patch
embeddings:

```math
\mathcal{X}_i
=
\{h_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}\in\mathbb{R}^{d}.
```

The defining symmetry is permutation invariance:

```math
f(\{h_1,\ldots,h_n\})
=
f(\{h_{\pi(1)},\ldots,h_{\pi(n)}\})
```

for every permutation $\pi$.

## Files

- `01_slide_as_set.md`: the mathematical object and symmetry.
- `02_deep_sets_and_mil.md`: Deep Sets, MIL, and invariant map approximation.
- `03_statistics_that_survive.md`: mean, max, attention, and distributional
  statistics.
- `04_failure_modes.md`: loss of geometry, sparse positives, and bag shortcuts.
