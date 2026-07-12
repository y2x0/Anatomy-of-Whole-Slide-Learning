# Kernel SHAP Weighted Regression

Primary anchor:

- Lundberg, Lee. "A Unified Approach to Interpreting Model Predictions."
  NeurIPS 2017.

## Additive Explanation Model

```math
g(z')
=
\phi_0
+
\sum_{i=1}^{m}
\phi_i z_i'.
```

Kernel SHAP chooses a particular weighted regression so its coefficients equal
Shapley values when all coalitions are represented and the coalition game is
evaluated exactly.

## Shapley Kernel

For coalition mask `z'` with size:

```math
|z'|
=
\sum_i z_i',
```

the kernel is:

```math
\pi_{mathrm{SHAP}}(z')
=
\frac{
m-1
}{
\binom{m}{|z'|}
|z'|
\left(
m-|z'|
\right)
}.
```

Empty and full coalitions receive limiting infinite weight, enforcing baseline
and local accuracy constraints.

## Weighted Regression

```math
\widehat\phi
=
\arg\min_{\phi}
\sum_{z'}
\pi_{\mathrm{SHAP}}(z')
\left[
v(z')
-
\phi_0
-
\sum_i
\phi_i z_i'
\right]^2.
```

With sampled coalitions, estimates have Monte Carlo and regression error.

## Why It Differs From LIME

LIME chooses locality and complexity for a sparse local surrogate. Kernel SHAP
chooses its kernel and unregularized additive form to recover a cooperative-game
allocation. Similar software interfaces hide different estimands.

## Variance At WSI Scale

For thousands of patch players, most random masks concentrate near half the
features, while the Shapley kernel emphasizes small and large coalitions.
Importance sampling and grouping become necessary.

## Grouped SHAP

Partition patches into groups:

```math
\mathcal{G}
=
\left\{
G_1,\ldots,G_m
\right\}.
```

Kernel SHAP then allocates group-game value:

```math
v_{\mathcal{G}}(S)
=
F
\left(
\bigcup_{r\in S}G_r
\right).
```

Group Shapley values cannot be uniquely disaggregated to patches without an
additional within-group rule.

## Regularization Boundary

Adding sparse feature selection changes the estimator and can violate exact
Shapley recovery. Approximate Kernel SHAP output should not be described as an
axiomatically exact value without convergence checks.
