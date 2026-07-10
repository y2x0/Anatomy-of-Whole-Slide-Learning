# SwAV Swapped Prediction And Gradients

SwAV does not directly attract two feature vectors. It computes a prototype
code for each view and asks the other view's feature to predict that code.

Primary anchor:

- Caron et al. "Unsupervised Learning of Visual Features by Contrasting Cluster
  Assignments." NeurIPS 2020. https://arxiv.org/abs/2006.09882

## View Features

For two transformations:

```math
x_t
=
t(x),
\qquad
x_s
=
s(x),
```

the encoder produces unit-normalized features:

```math
z_t
=
\frac{
f_{\theta}(x_t)
}{
\left\lVert
f_{\theta}(x_t)
\right\rVert_2
},
\qquad
z_s
=
\frac{
f_{\theta}(x_s)
}{
\left\lVert
f_{\theta}(x_s)
\right\rVert_2
}.
```

Let trainable prototypes be:

```math
C
=
\begin{bmatrix}
c_1&
\cdots&
c_K
\end{bmatrix}
\in
\mathbb{R}^{d\times K}.
```

The released training pseudocode normalizes prototype columns after each
update:

```math
c_k
\leftarrow
\frac{
c_k
}{
\lVert c_k\rVert_2
}.
```

## Prototype Prediction Distribution

For feature `z_t`:

```math
p_t^{(k)}
=
\frac{
\exp
\left(
z_t^{\top}c_k/\tau
\right)
}{
\sum_{r=1}^{K}
\exp
\left(
z_t^{\top}c_r/\tau
\right)
}.
```

In vector form:

```math
p_t
=
\mathrm{Softmax}
\left(
\frac{
C^{\top}z_t
}{
\tau
}
\right).
```

The same map gives `p_s`.

## Assignment Codes

Balanced assignment produces:

```math
q_t,
q_s
\in
\Delta^{K-1}.
```

These are not ordinary per-sample softmax probabilities. They are computed
jointly across a minibatch under prototype-usage constraints.

## One Direction

The cross-entropy predicting the code of view `s` from feature
`z_t` is:

```math
\ell
\left(
z_t,q_s
\right)
=
-
\sum_{k=1}^{K}
q_s^{(k)}
\log
p_t^{(k)}.
```

Equivalently:

```math
\ell
\left(
z_t,q_s
\right)
=
-
\frac{
1
}{
\tau
}
z_t^{\top}Cq_s
+
\log
\sum_{k=1}^{K}
\exp
\left(
z_t^{\top}c_k/\tau
\right).
```

The first term attracts the feature to the code-weighted prototype barycenter:

```math
\overline c(q_s)
=
Cq_s.
```

The log-partition term compares it against all prototypes.

## Swapped Objective

For two views:

```math
\mathcal{L}
\left(
z_t,z_s
\right)
=
\ell
\left(
z_t,q_s
\right)
+
\ell
\left(
z_s,q_t
\right).
```

The target crosses the view boundary:

```math
z_t
\longrightarrow
q_s,
\qquad
z_s
\longrightarrow
q_t.
```

No term directly minimizes:

```math
\left\lVert
z_t-z_s
\right\rVert_2^2.
```

The two features can differ while inducing mutually predictable prototype
coordinates.

## Stop-Gradient Assignment

In the SwAV pseudocode, Sinkhorn assignment is evaluated under no-gradient
context. During the prediction update:

```math
\frac{
\partial q_s
}{
\partial z_s
}
=
0,
\qquad
\frac{
\partial q_t
}{
\partial z_t
}
=
0.
```

Codes are recomputed on the next forward pass, but are fixed targets within the
current backward pass.

## Exact Score Gradient

Let:

```math
r_t
=
\frac{
C^{\top}z_t
}{
\tau
}.
```

Then:

```math
\nabla_{r_t}
\ell
\left(
z_t,q_s
\right)
=
p_t-q_s.
```

The feature gradient is:

```math
\nabla_{z_t}
\ell
\left(
z_t,q_s
\right)
=
\frac{1}{\tau}
C
\left(
p_t-q_s
\right).
```

Write:

```math
\overline c(p_t)
=
Cp_t.
```

Then:

```math
\nabla_{z_t}
\ell
=
\frac{1}{\tau}
\left[
\overline c(p_t)
-
\overline c(q_s)
\right].
```

The update aligns the predicted prototype barycenter with the assignment
barycenter from the other view.

## Prototype Gradient

For prototype `c_k`, ignoring the later normalization projection:

```math
\nabla_{c_k}
\ell
\left(
z_t,q_s
\right)
=
\frac{
p_t^{(k)}
-
q_s^{(k)}
}{
\tau
}
z_t.
```

Prototypes receiving more predicted than target mass move away under gradient
descent; prototypes receiving more target than predicted mass move toward the
feature.

The full swapped objective sums contributions from both directions and all
examples.

## Feature Hessian

For fixed codes and prototypes:

```math
\nabla_{z_t}^{2}
\ell
=
\frac{
1
}{
\tau^2
}
C
\left[
\mathrm{Diag}(p_t)
-
p_t p_t^{\top}
\right]
C^{\top}.
```

Since:

```math
\mathrm{Diag}(p_t)
-
p_t p_t^{\top}
\succeq
0,
```

the directional loss is convex in `z_t` when prototypes and code are
fixed. The complete neural optimization remains nonconvex because
`z_t=f_{\theta}(x_t)`, prototypes move, and codes are recomputed.

## Fixed-Point Condition

For unconstrained logits, the directional cross-entropy is minimized when:

```math
p_t
=
q_s.
```

The swapped pair seeks:

```math
p_t
=
q_s,
\qquad
p_s
=
q_t.
```

This is code consistency across views, not equality of features.

## Comparison With Pairwise InfoNCE

Pairwise contrast uses:

```math
z_t^{\top}z_s
```

and many feature-feature candidate scores.

SwAV uses:

```math
C^{\top}z_t,
\qquad
C^{\top}z_s,
```

and compares each feature to `K` prototypes. Pairwise sample interaction
enters through the batch-coupled assignment `Q`, not through a dense
feature-feature denominator.

## Complexity

Prototype score computation for `B` features costs:

```math
\Theta
\left(
BKd
\right).
```

By contrast, a dense pairwise feature matrix costs:

```math
\Theta
\left(
B^2d
\right).
```

Sinkhorn adds repeated scaling of a `K\times B` matrix:

```math
\Theta
\left(
IBK
\right)
```

for `I` iterations.

## C/R/G/S Placement

```text
\mathcal{G}:
    two views coupled through a batch-balanced prototype assignment

\mathcal{C}:
    shared encoder, trainable prototypes, and online Sinkhorn assignment

\mathcal{R}:
    soft code q and prototype probability p

\mathcal{S}:
    cross-entropy prediction of each view's code from the other view's feature
```

## Surviving Statistic

SwAV preserves a view-consistent coordinate distribution over learned
prototypes. The representation is identified through prototype prediction, not
direct feature equality.
