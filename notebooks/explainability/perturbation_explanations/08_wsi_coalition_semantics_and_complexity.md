# WSI Coalition Semantics And Complexity

## Player Choice

Players can be:

```math
\text{patches},
\quad
\text{spatial regions},
\quad
\text{phenotype clusters},
\quad
\text{prototypes},
\quad
\text{concepts}.
```

Changing the player set changes the game and attribution.

## Marginal Coalition Game

```math
v_{\mathrm{marg}}(S)
=
\mathbb{E}_{X_{\bar S}}
\left[
F
\left(
x_S,X_{\bar S}
\right)
\right].
```

This breaks dependence between retained and missing tissue.

## Conditional Coalition Game

```math
v_{\mathrm{cond}}(S)
=
\mathbb{E}
\left[
F(X)
\mid
X_S=x_S
\right].
```

This preserves observational dependence but can credit a feature through
correlated tissue retained in the conditional distribution.

## Deletion Coalition Game

For variable-size MIL:

```math
v_{\mathrm{delete}}(S)
=
F(X_S).
```

This avoids explicit imputation but changes cardinality and tissue prevalence.

## Empty Bag

Many MIL models are undefined on:

```math
X_{\varnothing}
=
\varnothing.
```

One must define a baseline embedding, dummy patch, or model extension. This
choice sets:

```math
v(\varnothing)
```

and therefore shifts the efficiency decomposition.

## Duplicate Patches

Shapley symmetry divides credit among exchangeable duplicates. If one tissue
phenotype appears in `r` identical patches, each can receive roughly one-rth of
the phenotype's coalition credit. Low individual value does not imply low
phenotype importance.

## Computational Explosion

Exact evaluation requires:

```math
2^n
```

coalitions. Even polynomial sampling becomes expensive when each WSI forward
pass includes transformer or graph context.

## Hierarchical Approximation

One tractable strategy is:

```math
\text{patches}
\longrightarrow
\text{regions}
\longrightarrow
\text{region Shapley values}
\longrightarrow
\text{within-region allocation}.
```

This is not generally equal to patch-level Shapley values because cross-region
and within-region interactions are allocated differently.

## Cohort Dependence

Conditional or marginal references depend on the background cohort. Shapley
values can change when institution, organ, stain, or disease prevalence in the
reference set changes.
