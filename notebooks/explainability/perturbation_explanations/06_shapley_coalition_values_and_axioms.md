# Shapley Coalition Values And Axioms

Primary anchor:

- Lundberg, Lee. "A Unified Approach to Interpreting Model Predictions."
  NeurIPS 2017.

## Coalition Game

Let features or WSI groups be players:

```math
N
=
\left\{
1,\ldots,m
\right\}.
```

Coalition value is:

```math
v(S)
=
\mathbb{E}
\left[
F(X)
\mid
X_S=x_S
\right]
```

under conditional semantics, or another explicitly chosen missingness game.

## Shapley Value

```math
\phi_i
=
\sum_{S\subseteq N\setminus\{i\}}
\frac{
|S|!
\left(
m-|S|-1
\right)!
}{m!}
\left[
v(S\cup\{i\})-v(S)
\right].
```

It averages marginal contribution across all insertion orders.

## Efficiency

```math
\sum_{i=1}^{m}
\phi_i
=
v(N)-v(\varnothing).
```

The explained score difference is conserved.

## Dummy

If:

```math
v(S\cup\{i\})
=
v(S)
\qquad
\forall S,
```

then:

```math
\phi_i=0.
```

## Symmetry

If two players have identical marginal effects for every coalition:

```math
v(S\cup\{i\})
=
v(S\cup\{j\})
```

whenever neither is in `S`, then:

```math
\phi_i=\phi_j.
```

## Additivity

For games `v` and `u`:

```math
\phi_i(v+u)
=
\phi_i(v)
+
\phi_i(u).
```

## WSI Meaning

Shapley values are exact for the declared coalition game. They do not make the
game's patch-removal or imputation semantics biologically correct. Axiomatic
uniqueness is conditional on the game definition.
