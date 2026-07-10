# GraphCL Graph-Level Objective

Primary anchor:

- You et al. "Graph Contrastive Learning with Augmentations." NeurIPS 2020.
  https://arxiv.org/abs/2010.13902

## Forward Map

For graph `i` and view `a`:

```math
\widehat G_i^{(a)}
\sim
q_a
\left(
\cdot\mid G_i
\right),
```

```math
h_i^{(a)}
=
f_{\theta}
\left(
\widehat G_i^{(a)}
\right),
```

```math
z_i^{(a)}
=
g_{\phi}
\left(
h_i^{(a)}
\right).
```

Here `f` includes GNN encoding and graph-level readout; `g` is a two-layer
projection head used for contrast.

## Printed Equation 3

The paper prints the directed loss for graph `n` as:

```math
\ell_n^{\mathrm{printed}}
=
-\log
\frac{
\exp
\left(
\mathrm{sim}
\left(
z_{n}^{(1)},z_{n}^{(2)}
\right)/\tau
\right)
}{
\sum_{n'\ne n}
\exp
\left(
\mathrm{sim}
\left(
z_{n}^{(1)},z_{n'}^{(2)}
\right)/\tau
\right)
}.
```

The positive term is excluded from the printed denominator. Consequently the
ratio is not necessarily a probability and the loss can be negative.

## Conventional Normalized Form

The usual cross-view InfoNCE normalization would be:

```math
\ell_n^{\mathrm{NCE}}
=
-\log
\frac{
\exp
\left(
s_{nn}/\tau
\right)
}{
\sum_{k=1}^{B}
\exp
\left(
s_{nk}/\tau
\right)
},
```

where:

```math
s_{nk}
=
\mathrm{sim}
\left(
z_n^{(1)},z_k^{(2)}
\right).
```

These expressions must not be silently conflated. The paper names its
criterion NT-Xent, while its displayed Equation 3 has the exclusion above.

## Gradient Of The Normalized Cross-View Form

Define:

```math
p_{nk}
=
\frac{
\exp(s_{nk}/\tau)
}{
\sum_j\exp(s_{nj}/\tau)
}.
```

Then:

```math
\frac{\partial\ell_n^{\mathrm{NCE}}}
{\partial s_{nk}}
=
\frac{1}{\tau}
\left(
p_{nk}-
\mathbb{1}\{k=n\}
\right).
```

For normalized vectors, the positive view attracts while every other graph in
the batch repels in proportion to its current softmax confusion.

## Negative Distribution

Other graphs in the minibatch define negatives:

```math
G_k
\sim
p_{\mathrm{batch}}(G),
\qquad
k\ne n.
```

The candidate task estimates same-source identity relative to this proposal.
If two WSIs share morphology or diagnosis, they remain negatives unless the
batch relation is modified.

## Graph-Level Bottleneck

Let node embeddings of a view be `H`. The graph encoder contains a readout:

```math
h
=
\mathcal{R}_{G}(H).
```

The objective can constrain only differences surviving `R_G`:

```math
\mathcal{R}_{G}(H_a)
=
\mathcal{R}_{G}(H_b)
\Longrightarrow
\ell
\text{ cannot distinguish }H_a,H_b
\text{ through the anchor score}.
```

Graph contrast is therefore not automatically sensitive to node-level
arrangement or rare-node evidence.

## Projection-Head Boundary

Training constrains:

```math
z
=
g_{\phi}(h),
```

while downstream transfer commonly uses `h`. If `g` is non-injective:

```math
h_a\ne h_b,
\qquad
g(h_a)=g(h_b),
```

the loss permits differences in `h` that it cannot observe. The contrastive
geometry and transferred representation geometry are related but not
identical.
