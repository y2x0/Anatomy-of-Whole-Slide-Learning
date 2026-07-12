# LIME Local Surrogate Geometry

Primary anchor:

- Ribeiro, Singh, Guestrin. "Why Should I Trust You? Explaining the Predictions
  of Any Classifier." KDD 2016.

## Interpretable Representation

Let original WSI representation be `x` and interpretable binary mask be:

```math
z'
\in
\left\{
0,1
\right\}^{m}.
```

An interpretable-to-model map creates:

```math
z
=
h_x(z').
```

For pathology, coordinates of `z'` can denote patch groups, tissue regions, or
concept clusters.

## Local Objective

LIME solves:

```math
g^{\star}
=
\arg\min_{g\in\mathcal{G}}
\mathcal{L}
\left(
f,g,\pi_x
\right)
+
\Omega(g).
```

For weighted squared error:

```math
\mathcal{L}
\left(
f,g,\pi_x
\right)
=
\sum_{z'\in\mathcal{Z}}
\pi_x(z)
\left[
f(z)-g(z')
\right]^2.
```

The locality kernel in the paper has exponential form:

```math
\pi_x(z)
=
\exp
\left(
-D(x,z)^2/\sigma^2
\right).
```

## Sparse Linear Explanation

```math
g(z')
=
\beta_0
+
\sum_{j=1}^{m}
\beta_jz_j'.
```

Complexity penalty or explicit feature count limits the displayed terms.

## Weighted Least Squares

Without sparsity penalty:

```math
\widehat\beta
=
\left(
Z^{\top}WZ
\right)^{-1}
Z^{\top}Wy.
```

The explanation depends on sampled masks `Z`, kernel weights `W`, and
perturbed predictions `y`.

## Locality Is Mask-Space Dependent

Two masks with equal Hamming distance can remove one contiguous tumor region or
scattered benign patches. A mask-space metric need not reflect histologic or
model-space locality.

## Surrogate Fidelity

Weighted local fidelity is:

```math
R_{\mathrm{local}}^2
=
1-
\frac{
\sum_r w_r
\left(
y_r-g(z_r')
\right)^2
}{
\sum_r w_r
\left(
y_r-\bar y_w
\right)^2
}.
```

Coefficients should not be interpreted without reporting whether the sparse
linear surrogate actually approximates the model locally.

## Nonuniqueness

Correlated region masks make `Z^T W Z` ill-conditioned. Different samples or
regularization can yield different coefficients with similar local fidelity.
