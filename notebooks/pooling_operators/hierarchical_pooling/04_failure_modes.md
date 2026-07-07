# Hierarchical Pooling Failure Modes

Hierarchical pooling is powerful because it respects scale. It fails when the
chosen hierarchy is wrong.

## 1. Premature Compression

If local pooling discards rare evidence:

```math
g_{ir}
=
\frac{1}{|\mathrm{Ch}(r)|}
\sum_{j\in\mathrm{Ch}(r)}u_{ij},
```

then the global stage cannot recover it.

## 2. Bad Region Boundaries

If a region mixes unrelated tissue:

```text
tumor + benign + artifact
```

then the region state may average away the relevant morphology.

## 3. Unequal Region Weighting

Two-stage means weight patches by:

```math
w_{ij}
=
\frac{1}{M_i|\mathrm{Ch}(r(j))|}.
```

This may overweight small regions and underweight large regions.

## 4. Scale Mismatch

Some labels depend on patch-level morphology. Others depend on tissue
architecture. A hierarchy fails if it compresses at the wrong scale:

```text
too fine:
    misses architecture

too coarse:
    misses rare cells or lesions
```

## 5. Interpretability Ambiguity

A high region weight:

```math
b_{ir}
```

does not mean every patch in the region is positive. It means the region state
was influential after local pooling.

## Dense Summary

Hierarchical pooling fails when:

```text
the parent map is wrong,
local pooling discards evidence,
or effective patch weights do not match the biological question.
```

The first diagnostic is always the effective fine-scale weight induced by the
hierarchy.

