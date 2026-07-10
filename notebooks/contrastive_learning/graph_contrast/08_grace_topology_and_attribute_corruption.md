# GRACE Topology And Attribute Corruption

Primary anchor:

- Zhu et al. "Deep Graph Contrastive Representation Learning." 2020.
  https://arxiv.org/abs/2006.04131

## Edge Removal

For each existing edge, GRACE samples:

```math
R_{ij}^{(a)}
\sim
\mathrm{Bernoulli}
\left(
1-p_{r,a}
\right)
\qquad
\text{when }A_{ij}=1.
```

The corrupted adjacency is:

```math
\widetilde A^{(a)}
=
A\odot R^{(a)}.
```

No non-edge is added in the stated GRACE corruption. This differs from
GraphCL's edge perturbation family, which allows addition and deletion.

## Shared-Dimension Feature Masking

For view `a`, sample a feature-dimension mask:

```math
m_k^{(a)}
\sim
\mathrm{Bernoulli}
\left(
1-p_{m,a}
\right),
\qquad
k=1,\ldots,F.
```

Every node receives the same coordinate mask:

```math
\widetilde X^{(a)}
=
X
\mathrm{diag}
\left(
m^{(a)}
\right).
```

This is not independent elementwise masking. A removed feature coordinate is
absent from every node in that view.

## Node-Receptive-Field Change

For an `L`-layer GNN, define the rooted receptive field after corruption:

```math
\widetilde G^{(a)}[i,L].
```

The positive pair is:

```math
u_i
=
F_{\theta}^{(L)}
\left(
\widetilde G^{(1)}[i,L]
\right),
\qquad
v_i
=
F_{\theta}^{(L)}
\left(
\widetilde G^{(2)}[i,L]
\right).
```

GRACE preserves root identity while perturbing the context used to encode that
root.

## Receptive-Field Isolation

For node `i` with degree `d_i`, the probability all incident edges are removed
in view `a` is:

```math
p_{\mathrm{iso},i}^{(a)}
=
p_{r,a}^{d_i}.
```

For low-degree WSI boundary nodes, this can be appreciable. Positive alignment
then asks an isolated-node representation to agree with a context-rich one.

## Expected Degree

Under independent removal:

```math
\mathbb{E}
\left[
\widetilde d_i^{(a)}
\mid
d_i
\right]
=
\left(
1-p_{r,a}
\right)d_i.
```

The degree variance is:

```math
\mathrm{Var}
\left(
\widetilde d_i^{(a)}
\mid
d_i
\right)
=
d_i p_{r,a}
\left(
1-p_{r,a}
\right).
```

Thus corruption noise depends on original degree.

## Information Intersection

If the two views independently mask feature coordinate `k`, it survives both
with probability:

```math
p_{\mathrm{both},k}
=
\left(
1-p_{m,1}
\right)
\left(
1-p_{m,2}
\right).
```

It appears in exactly one view with probability:

```math
p_{\mathrm{one},k}
=
\left(
1-p_{m,1}
\right)p_{m,2}
+
p_{m,1}
\left(
1-p_{m,2}
\right).
```

Strong alignment preferentially preserves information predictable from the
intersection of the two corrupted contexts.

## Spatial WSI Consequence

If an edge represents a genuine tissue adjacency, removal is a missing-edge
intervention. GRACE teaches robustness to incomplete spatial context. It does
not teach robustness to false adjacency because its stated view generator does
not add edges.
