# Graph False Negatives, Automorphisms, And Collisions

Primary anchors:

- You et al. "Graph Contrastive Learning with Augmentations." NeurIPS 2020.
  https://arxiv.org/abs/2010.13902
- Zhu et al. "Deep Graph Contrastive Representation Learning." 2020.
  https://arxiv.org/abs/2006.04131

## Node Identity Is Stronger Than Structural Identity

GRACE labels only `(u_i,v_i)` positive. Suppose nodes `i` and `j` are exchanged
by graph automorphism `\pi`:

```math
A_{\pi(i)\pi(j)}
=
A_{ij},
\qquad
X_{\pi(i)}
=
X_i.
```

For a permutation-equivariant GNN:

```math
h_{\pi(i)}
=
h_i.
```

If `j=\pi(i)`, GRACE simultaneously has:

```math
\theta(u_i,v_i)
=
\theta(u_i,v_j)
```

while treating the first as positive and the second as negative.

## Irreducible Node Loss

If `m` candidate nodes are embedding-indistinguishable from the designated
positive, then its softmax probability is at most:

```math
p_{i,+}
\le
\frac{1}{m}.
```

Therefore:

```math
-\ell(u_i,v_i)
\ge
\log m.
```

The contradiction is created by index identity within an exchangeable
structural class.

## Repeated Tissue Motifs

Let `C(i)` denote a latent morphologic role. GRACE's observed relation is:

```math
R_{ij}
=
\mathbb{1}\{i=j\},
```

while a semantic relation may be:

```math
R_{ij}^{\star}
=
\mathbb{1}
\left\{
C(i)=C(j)
\right\}.
```

Every repeated gland, lymphocyte cluster, or stromal motif can become a false
negative under the semantic relation.

## Graph-Level False Negatives

GraphCL treats every other source graph as negative:

```math
R_{ab}
=
\mathbb{1}\{a=b\}.
```

If patient slides `a` and `b` share diagnostic state:

```math
Y_a=Y_b,
\qquad
a\ne b,
```

then graph identity and task identity disagree. The loss preserves
instance-discriminative features that may include stain, scanner, tissue area,
or institution.

## Collision Probability Under Readout

Let graph representation be:

```math
h_G
=
\mathcal{R}
\left(
\left\{
h_i
\right\}_{i\in V}
\right).
```

For non-injective `R`, define collision event:

```math
\mathcal{E}_{ab}
=
\left\{
G_a\ne G_b,
\quad
h_{G_a}=h_{G_b}
\right\}.
```

Graph-level contrast cannot satisfy instance discrimination on collided
representations. Increasing encoder capacity does not repair information lost
by a fixed readout after the collision occurs.

## Spatial Coordinate Escape

Adding absolute coordinates can break automorphisms:

```math
x_i'
=
\left[
x_i;
c_i
\right].
```

This makes node identity easier, but can introduce slide-frame shortcuts. If
coordinates encode scanner cropping or tissue placement patterns, contrastive
success need not reflect morphology.

## Design Principle

The negative relation should match the intended equivalence class:

```math
\text{index identity}
\ne
\text{structural role}
\ne
\text{morphologic class}
\ne
\text{clinical label}.
```

Choosing among them is supervision design, not a minor sampling detail.
