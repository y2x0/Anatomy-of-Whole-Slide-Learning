# CLAM Heatmap Resolution And Semantics

Primary anchor:

- Lu et al. "Data-Efficient and Weakly Supervised Computational Pathology on
  Whole-Slide Images." Nature Biomedical Engineering 2021.
  https://arxiv.org/abs/2004.09666

## Patch Lattice

Let patch centers be:

```math
c_k
\in
\mathbb{R}^{2}.
```

The model produces one score per sampled patch:

```math
a_k
=
a(c_k).
```

A displayed dense heatmap is a rendering operator:

```math
H_{\mathrm{display}}
=
\mathcal{I}
\left(
\left\{
(c_k,a_k)
\right\}_{k=1}^{N}
\right),
```

where `I` performs interpolation, overlap averaging, smoothing, or
normalization.

## Resolution Limit

If patch width is `p` and stride is `s`, the explanatory observation grid has
spacing `s`. Rendering at pixel resolution does not increase the information
resolution beyond patch sampling and receptive field.

```math
\text{display resolution}
\ne
\text{attribution resolution}.
```

## Overlap Averaging

For pixel location `u`, let covering patches be:

```math
\mathcal{K}(u)
=
\left\{
k:
u\in\mathrm{patch}_k
\right\}.
```

An averaged display is:

```math
H(u)
=
\frac{1}{|\mathcal{K}(u)|}
\sum_{k\in\mathcal{K}(u)}
a_k.
```

This smooths boundary variation and can make coarse patch scores appear like
fine segmentation.

## Within-Slide Normalization

If heatmaps are percentile normalized per slide:

```math
\widetilde a_k
=
\frac{
a_k-min_j a_j
}{
\max_j a_j-min_j a_j
},
```

equal colors across slides do not imply equal raw attention or equal model
evidence.

## Softmax Cardinality

Because:

```math
\sum_{k=1}^{N}a_k=1,
```

average attention is:

```math
\frac{1}{N}.
```

Slides with different patch counts have different reference scales. A raw
weight of fixed magnitude has different relative meaning across `N`.

## Heatmap Thresholding

Thresholded region:

```math
R_{\tau}
=
\left\{
k:a_k\ge\tau
\right\}.
```

Its area depends on temperature, bag cardinality, overlap, and display
normalization. Thresholds need calibration rather than visual selection.

## Localization Versus Segmentation

Attention can prioritize tumor-containing patches without delineating tumor
pixels. Patch-level localization is compatible with mixed-content patches and
cannot be evaluated as pixel-perfect segmentation without acknowledging this
coarsening.
