# GRACE Node-Identity Objective

Primary anchor:

- Zhu et al. "Deep Graph Contrastive Representation Learning." 2020.
  https://arxiv.org/abs/2006.04131

## Corresponding Node Views

GRACE generates two corrupted views of one graph:

```math
\widetilde G^{(1)}
=
\left(
\widetilde X^{(1)},
\widetilde A^{(1)}
\right),
\qquad
\widetilde G^{(2)}
=
\left(
\widetilde X^{(2)},
\widetilde A^{(2)}
\right).
```

A shared GNN returns node matrices:

```math
U
=
f_{\theta}
\left(
\widetilde X^{(1)},
\widetilde A^{(1)}
\right),
\qquad
V
=
f_{\theta}
\left(
\widetilde X^{(2)},
\widetilde A^{(2)}
\right).
```

Node indices are retained, so `(u_i,v_i)` is positive.

## Projected Critic

The critic is:

```math
\theta(u,v)
=
\mathrm{sim}
\left(
g_{\phi}(u),
g_{\phi}(v)
\right),
```

where `g` is a two-layer MLP and `sim` is cosine similarity.

## Exact Directed Objective

GRACE defines:

```math
\ell(u_i,v_i)
=
\log
\frac{
\exp
\left(
\theta(u_i,v_i)/\tau
\right)
}{
\exp
\left(
\theta(u_i,v_i)/\tau
\right)
+
\sum_{k\ne i}
\exp
\left(
\theta(u_i,v_k)/\tau
\right)
+
\sum_{k\ne i}
\exp
\left(
\theta(u_i,u_k)/\tau
\right)
}.
```

The first negative sum is inter-view; the second is intra-view. The paper
maximizes the symmetric average:

```math
\mathcal{J}_{\mathrm{GRACE}}
=
\frac{1}{2N}
\sum_{i=1}^{N}
\left[
\ell(u_i,v_i)
+
\ell(v_i,u_i)
\right].
```

## Candidate Set Geometry

For anchor `u_i`, define:

```math
\mathcal{K}_i
=
\left\{
v_i
\right\}
\cup
\left\{
v_k:k\ne i
\right\}
\cup
\left\{
u_k:k\ne i
\right\}.
```

It has:

```math
|\mathcal{K}_i|
=
2N-1.
```

Unlike cross-view-only InfoNCE, GRACE explicitly repels different nodes inside
the anchor's own view.

## Exact Similarity Gradients

Let `p_{i,+}`, `p_{i,k}^{\mathrm{inter}}`, and
`p_{i,k}^{\mathrm{intra}}` be softmax probabilities over the denominator. For
the minimized loss `L_i=-\ell(u_i,v_i)`:

```math
\frac{\partial L_i}
{\partial\theta(u_i,v_i)}
=
\frac{1}{\tau}
\left(
p_{i,+}-1
\right),
```

```math
\frac{\partial L_i}
{\partial\theta(u_i,v_k)}
=
\frac{1}{\tau}
p_{i,k}^{\mathrm{inter}},
\qquad
k\ne i,
```

```math
\frac{\partial L_i}
{\partial\theta(u_i,u_k)}
=
\frac{1}{\tau}
p_{i,k}^{\mathrm{intra}},
\qquad
k\ne i.
```

Hard structurally similar nodes receive the largest repulsive gradients.

## Computational Scale

The full score calculation is quadratic:

```math
\mathrm{cost}
=
\Theta
\left(
N^2d
\right),
\qquad
\mathrm{memory}
=
\Theta
\left(
N^2
\right)
```

if all pairwise similarities are materialized. A WSI graph with tens of
thousands of patches cannot use this full node contrast naively.
