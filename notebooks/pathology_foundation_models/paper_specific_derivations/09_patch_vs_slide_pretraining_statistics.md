# Patch Versus Slide Pretraining Statistics

References: [HIPT](https://arxiv.org/abs/2206.02647),
[Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w),
[TITAN](https://www.nature.com/articles/s41591-025-03982-3).

## 1. Patch Statistic

A patch foundation model learns

```math
h_j=f_\phi(x_j).
```

Its objective sees local views, local text, or local teacher targets.

## 2. Slide Statistic

A slide foundation model learns

```math
z_i=F_\psi(\{h_{ij},c_{ij}\}_{j=1}^{n_i}),
```

and can preserve co-occurrence, arrangement, and long-range context.

## 3. Non-equivalence

There is no general map from a strong patch encoder to an equally strong slide
encoder:

```math
\mathcal R_1(\{f_\phi(x_j)\})
\not\equiv
F_\psi(\{x_j,c_j\}).
```

The first may preserve only the marginal patch distribution; the second may
learn a joint spatial or multimodal statistic.

## 4. Sampling Unit

Patch-level objectives often weight slides by their number of sampled patches.
Slide-level objectives weight patients or slides directly. This changes the
effective training distribution and can bias rare disease or large-slide
representation quality.
