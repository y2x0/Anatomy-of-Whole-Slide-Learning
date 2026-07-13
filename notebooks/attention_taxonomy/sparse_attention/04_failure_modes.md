# Sparse Attention Failure Modes

Sparse attention saves computation and can make support visually cleaner, but
it can remove evidence before the model has a chance to use it.

## Wrong Support

Let the true evidence indices be:

```math
E
\subset
\{1,\ldots,n\}.
```

Let the attention support be:

```math
\widehat E
=
\{j:a_j>0\}.
```

If:

```math
E\not\subseteq\widehat E,
```

then some evidence receives exactly zero value contribution:

```math
\sum_{j\in E\setminus\widehat E}a_jv_j
=
0.
```

This is worse than low attention. It is deletion.

## Boundary Instability

Sparsemax and top-k have support changes. A small score perturbation can switch:

```math
j\notin\widehat E
\quad
\to
\quad
j\in\widehat E.
```

The regularity depends on the mechanism. Sparsemax and entmax are continuous
normalizations with a changing Jacobian at support boundaries. Hard top-k changes
the selected index set discontinuously at score ties; if the exchanged values
differ, its readout can jump even when the score perturbation is arbitrarily
small.

Support stability should therefore be measured directly. For two perturbed
versions of the same bag, define:

```math
\mathrm{Jaccard}(\widehat E,\widehat E')
=
\frac{|\widehat E\cap\widehat E'|}
{|\widehat E\cup\widehat E'|}.
```

A sparse heat map with low support Jaccard is a brittle selector, regardless of
how clean the single displayed map appears.

## Rare Distributed Evidence

Sparse MIL is attractive when one small lesion determines the label. It is less
safe when evidence is weak and distributed:

```math
\text{risk}
\propto
\sum_{j\in E}b_j
```

with many small positive contributions. Sparse support may keep only the most
salient pieces and underestimate burden.

## Mask-Induced Geometry Error

If a sparse mask uses coordinates:

```math
M_{uv}
=
\mathbf{1}\{\|c_u-c_v\|\le r\},
```

then long-range morphologic relations are impossible in one layer. If a mask
uses feature nearest neighbors, physically adjacent context may disappear.

## Explanation Trap

Sparse maps look interpretable:

```text
few highlighted patches
```

but the support is an output of the model objective, not proof of causal
pathology. Sparsity increases visual confidence faster than it increases
identifiability.

## Dense Summary

Sparse attention is a support estimator:

```math
H
\to
\widehat E_\theta
\to
\sum_{j\in\widehat E_\theta}a_jv_j.
```

Its main failure mode is simple:

```text
the model cannot recover evidence it masked out
```
