# Set Representation Failure Modes

The set view is useful because it matches weak slide-level labels and variable
numbers of patches. Its failures follow from its symmetry.

## 1. Geometry Loss

If:

```math
\mathcal{X}_i=\{h_{ij}\}_{j=1}^{n_i},
```

then:

```math
\{h_{i1},h_{i2},h_{i3}\}
=
\{h_{i3},h_{i1},h_{i2}\}.
```

Any information carried only by arrangement is absent.

Examples:

```text
tumor-stroma interface
lymphocyte localization
gland architecture
necrosis relative to tumor
invasion front geometry
```

## 2. Sparse Positives

Mean pooling dilutes rare evidence:

```math
z_i
=
\frac{1}{n_i}
\left(
\sum_{j\in P_i}h_{ij}
+
\sum_{j\notin P_i}h_{ij}
\right).
```

If |P_i|\ll n_i, the positive morphology has small weight.

## 3. Attention Collapse

Attention can concentrate:

```math
a_{ij^\star}\approx1.
```

This helps sparse positives but can collapse to confounded patches:

```text
scanner artifacts
tissue folds
easy subtype correlates
background shortcuts
```

## 4. Bag Shortcut Learning

The model can learn slide-level shortcuts through patch distributions:

```text
tissue amount
stain intensity
scanner style
institution-specific preprocessing
tumor purity
```

These are valid set statistics but not necessarily biological mechanisms.

## 5. Loss Of Interaction Semantics

Set attention can model pairwise feature interactions, but without geometry it
cannot know whether two patches were adjacent unless coordinate features are
included.

Interaction:

```math
\psi(h_j,h_k)
```

does not imply spatial relation.

## Diagnostic Questions

1. Is the task driven by morphology prevalence or spatial arrangement?
2. Are coordinates included?
3. Is the positive signal sparse or diffuse?
4. Does attention remain stable across folds?
5. Can the same patch distribution with different layout change the label?

## Dense Summary

Set representations assume:

```math
f(H)=f(PH)
```

for every permutation
```math
P
```
.

If the label depends on anything not invariant to permutation, the set
representation is underspecified.
