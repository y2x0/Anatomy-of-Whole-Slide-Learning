# GraphCL Augmentation Kernels

Primary anchor:

- You et al. "Graph Contrastive Learning with Augmentations." NeurIPS 2020.
  https://arxiv.org/abs/2010.13902

GraphCL defines graph-level positives by sampling two transformations of one
source graph. The augmentation kernels encode the semantic prior.

## Two-View Sampling

For source graph `G_i`:

```math
\widehat G_i^{(1)}
\sim
q_1
\left(
\cdot\mid G_i
\right),
\qquad
\widehat G_i^{(2)}
\sim
q_2
\left(
\cdot\mid G_i
\right).
```

The positive joint is:

```math
p_{+}
\left(
\widehat G^{(1)},
\widehat G^{(2)}
\right)
=
\int
p(G)
q_1
\left(
\widehat G^{(1)}\mid G
\right)
q_2
\left(
\widehat G^{(2)}\mid G
\right)
\,\mathrm{d}G.
```

## Node Dropping

Let node-retention indicators be:

```math
r_i
\sim
\mathrm{Bernoulli}
\left(
1-p_v
\right).
```

With retained index set:

```math
V'
=
\left\{
i:r_i=1
\right\},
```

the induced view is:

```math
A'
=
A[V',V'],
\qquad
X'
=
X[V',:].
```

The implicit assumption is that deleting sampled vertices and all incident
edges preserves graph semantics.

## Edge Perturbation

Let deletion and addition masks act on present and absent edges:

```math
D_{ij}
\sim
\mathrm{Bernoulli}(p_{-})
\quad
\text{when }A_{ij}=1,
```

```math
B_{ij}
\sim
\mathrm{Bernoulli}(p_{+})
\quad
\text{when }A_{ij}=0.
```

Then:

```math
A'_{ij}
=
A_{ij}(1-D_{ij})
+
(1-A_{ij})B_{ij}.
```

This declares some connectivity changes nuisance variation. In spatial WSI
graphs, random edge addition can create anatomically impossible neighbors.

## Attribute Masking

For feature mask `m`:

```math
m_k
\sim
\mathrm{Bernoulli}
\left(
1-p_x
\right),
```

```math
X'
=
X
\mathrm{diag}(m).
```

GraphCL describes masking partial vertex attributes. The induced invariance is
to information removed along masked feature coordinates.

## Subgraph Sampling

Let a random walk or related sampler choose `V'`:

```math
V'
\sim
q_{\mathrm{sub}}
\left(
\cdot\mid G
\right).
```

The augmented graph is the induced subgraph:

```math
G'
=
G[V'].
```

The prior is:

```math
Y(G[V'])
\approx
Y(G).
```

For sparse-positive MIL-style pathology labels, this can fail whenever the
sampled subgraph excludes the diagnostic region.

## Survival Probability Of Rare Evidence

Suppose `r` of `N` nodes carry essential evidence and each node survives
independently with probability `1-p_v`. The probability that at least one
essential node survives is:

```math
p_{\mathrm{survive}}
=
1-p_v^r.
```

The probability that all essential nodes survive is:

```math
p_{\mathrm{all}}
=
\left(
1-p_v
\right)^r.
```

If one unique focus is essential, semantic preservation probability is only:

```math
p_{\mathrm{survive}}
=
1-p_v.
```

Random augmentation strength is therefore a direct prior on how dispensable
rare pathology evidence is.
