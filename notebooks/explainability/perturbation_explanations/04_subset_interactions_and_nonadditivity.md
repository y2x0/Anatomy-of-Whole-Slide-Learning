# Subset Interactions And Nonadditivity

## Set Effect

For subset `S`:

```math
\Delta(S)
=
F(X)
-
F(X_{-S}).
```

Additivity would require:

```math
\Delta(S)
=
\sum_{i\in S}
\Delta(\{i\}).
```

This fails under interactions and normalization.

## Pair Interaction

Define removal interaction:

```math
I_{ij}
=
\Delta
\left(
\{i,j\}
\right)
-
\Delta(\{i\})
-
\Delta(\{j\}).
```

Positive `I_ij` indicates synergy under deletion; negative indicates redundancy
or overcounting.

## Inclusion-Exclusion Expansion

A set function can be decomposed into Moebius coefficients:

```math
m(S)
=
\sum_{T\subseteq S}
(-1)^{|S|-|T|}
v(T),
```

with inverse:

```math
v(S)
=
\sum_{T\subseteq S}
m(T).
```

Singleton heatmaps retain only first-order coefficients and omit higher-order
tissue interactions.

## Redundant Lesion Patches

If any one of `r` tumor patches is sufficient:

```math
F(X)
=
\mathbb{1}
\left\{
\sum_{i=1}^{r}y_i>0
\right\},
```

then deleting one patch has zero effect while deleting all has maximal effect.
Single-patch perturbation declares every redundant witness unimportant.

## Necessary Combination

If prediction requires two morphologies:

```math
F(X)
=
\mathbb{1}
\left\{
A\text{ present}
\right\}
\mathbb{1}
\left\{
B\text{ present}
\right\},
```

each component can have large removal effect, but neither is sufficient alone.

## Group Construction

Meaningful subsets can be defined by spatial connectedness, clustering,
annotations, prototypes, or concepts. Group explanations are conditional on
that partition:

```math
\mathcal{G}
=
\left\{
G_1,\ldots,G_m
\right\}.
```

## Complexity

All subset evaluations require:

```math
2^n
```

coalitions. Approximation is unavoidable at WSI scale, and the sampling law
determines which interactions are represented.
