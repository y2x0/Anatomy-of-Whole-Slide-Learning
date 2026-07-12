# Additive Credit And The Shapley Boundary

Primary anchor:

- Javed et al. "Additive MIL: Intrinsically Interpretable Multiple Instance
  Learning." NeurIPS 2022 Workshop.
  https://arxiv.org/abs/2206.01794

## Coalition Game

Let patch index set be:

```math
N
=
\left\{
1,\ldots,n
\right\}.
```

For coalition `S`:

```math
v(S)
=
F(X_S).
```

The Shapley value for patch `i` is:

```math
\phi_i
=
\sum_{S\subseteq N\setminus\{i\}}
\frac{
|S|!
\left(
n-|S|-1
\right)!
}{n!}
\left[
v(S\cup\{i\})-v(S)
\right].
```

## Additive Game

If:

```math
v(S)
=
b
+
\sum_{j\in S}e_j,
```

then every marginal contribution is:

```math
v(S\cup\{i\})-v(S)
=
e_i.
```

Therefore:

```math
\phi_i=e_i.
```

No coalition enumeration is required.

## Interaction Game

If:

```math
v(S)
=
\sum_{j\in S}e_j
+
\sum_{
j<k:
j,k\in S
}
e_{jk},
```

then:

```math
\phi_i
=
e_i
+
\frac{1}{2}
\sum_{j\ne i}
e_{ij}.
```

Any patch-only evidence map must decide how interaction credit is allocated.

## Missingness Semantics

The coalition value depends on how absent patches are represented:

```math
v(S)
=
F
\left(
\mathcal{T}(X,S,B)
\right).
```

Deleting instances, zeroing embeddings, or conditionally imputing tissue
defines different games and different Shapley values.

## Additive MIL Claim Boundary

The exact equivalence holds for the additive model's own coalition function.
It does not show that additive patch evidence equals Shapley values of an
arbitrary nonadditive MIL model.

## Explanatory Tradeoff

Additive architecture gives:

```math
\text{exactness},
\quad
\text{sign},
\quad
\text{class specificity},
\quad
\text{linear-time credit}.
```

The price is restriction of the score family. Whether that restriction harms
prediction depends on whether task-relevant interactions can be encoded within
individual patch representations or are truly cross-patch.
