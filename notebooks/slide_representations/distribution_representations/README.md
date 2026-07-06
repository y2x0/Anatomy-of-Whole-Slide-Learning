# Distribution Representations

A distribution representation treats a slide as a probability measure over
morphology space.

The core question:

```text
What distribution of patch phenotypes does this slide contain?
```

Instead of representing the slide by an ordered or relational object, we write:

```math
\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\delta_{h_{ij}}.
```

Then we choose a finite statistic:

```math
z_i=T(\mu_i).
```

## Files

- `01_slide_as_distribution.md`: empirical measures and distributional
  equivalence.
- `02_moments_prototypes_mixtures.md`: moments, prototypes, and mixture
  summaries.
- `03_distances_and_transport.md`: comparing slides as distributions.
- `04_failure_modes.md`: identifiability, binning, and geometry loss.

## Anchor Methods

PANTHER is the cleanest modern pathology example: it summarizes slide patches
through mixture-model prototypes. SAMPLER-like CDF encodings are another
distributional view.

The unifying object is:

```math
\mu_i\in\mathcal{P}(\mathbb{R}^{d}),
```

not a sequence, graph, or attention map.
