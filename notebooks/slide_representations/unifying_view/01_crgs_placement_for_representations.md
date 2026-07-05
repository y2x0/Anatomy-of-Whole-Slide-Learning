# C/R/G/S Placement For Representations

The slide representation chooses the mathematical object that the model sees.

Start with patch embeddings:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}\in\mathbb{R}^{d}.
```

A representation augments or quotients this collection:

```math
\mathcal{X}_i
=
(H_i,\Omega_i),
```

where $\Omega_i$ is the structure assigned to the slide.

The full forward map is:

```math
\widetilde{H}_i
=
\mathcal{C}_\theta(H_i;\Omega_i),
\qquad
z_i
=
\mathcal{R}_\theta(\widetilde{H}_i;\Omega_i),
\qquad
\widehat{y}_i
=
\mathcal{H}_\theta(z_i).
```

This gives the decomposition:

```text
C:
    context operator

R:
    readout operator

G:
    geometry or graph structure

S:
    supervision and task signal
```

The important point is that $\Omega_i$ changes which functions are natural,
which symmetries are valid, and which failures are expected.

## Set Representation

The set view chooses:

```math
\Omega_i=\varnothing.
```

The slide is an unordered empirical measure:

```math
\mu_i
=
\frac{1}{n_i}\sum_{j=1}^{n_i}\delta_{h_{ij}}.
```

The required symmetry is:

```math
f(H_i)=f(PH_i)
\qquad
\text{for every permutation }P.
```

A set model usually has:

```math
\widetilde{H}_i
=
\mathcal{C}_{\operatorname{set}}(H_i),
\qquad
z_i
=
\mathcal{R}_{\operatorname{inv}}(\widetilde{H}_i).
```

If there is no interaction before readout:

```math
\mathcal{C}_{\operatorname{set}}(H_i)=H_i.
```

Then the readout determines almost everything:

```math
z_i
=
\frac{1}{n_i}\sum_j h_{ij},
\qquad
z_i
=
\sum_j a_{ij}v(h_{ij}),
\qquad
z_i
=
\max_j g(h_{ij}).
```

The surviving statistic is a functional of $\mu_i$.

## Sequence Representation

The sequence view chooses a total order:

```math
\Omega_i=\sigma_i.
```

The slide object is:

```math
\mathcal{X}_i
=
(h_{i\sigma_i(1)},\ldots,h_{i\sigma_i(n_i)}).
```

A sequence context operator is order-sensitive:

```math
u_{it}
=
\mathcal{C}_{\operatorname{seq}}
(h_{i\sigma_i(1)},\ldots,h_{i\sigma_i(n_i)})_t.
```

Then:

```math
z_i
=
\mathcal{R}(u_{i1},\ldots,u_{in_i}).
```

The order $\sigma_i$ is part of the representation. Changing $\sigma_i$ changes
the hypothesis class unless the model explicitly averages over orders or
restores permutation invariance.

For RNNs and state-space models, the order induces a path:

```math
\sigma_i(1)\to\sigma_i(2)\to\cdots\to\sigma_i(n_i).
```

For full self-attention with positional information, the graph of interactions
is not this path. It is a complete graph with positional or coordinate features.

## Graph Representation

The graph view chooses:

```math
\Omega_i=(G_i,E_i),
\qquad
G_i=(V_i,A_i).
```

Node features are patch, cell, region, or tissue-unit features:

```math
H_i=\{h_v:v\in V_i\}.
```

Message passing computes:

```math
\widetilde{H}_i
=
\operatorname{GNN}_\theta(H_i,A_i,E_i).
```

Readout gives:

```math
z_i
=
\mathcal{R}_{\operatorname{graph}}(G_i,\widetilde{H}_i).
```

The valid symmetry is graph relabeling equivariance:

```math
\operatorname{GNN}(PH_i,PA_iP^\top)
=
P\operatorname{GNN}(H_i,A_i).
```

The graph is not neutral preprocessing. It defines possible information flow.

## Placement Table

| Representation | $\Omega_i$ | $\mathcal{C}$ | $\mathcal{R}$ | Surviving Statistic |
|---|---:|---:|---:|---:|
| Set mean MIL | none | identity | mean | first moment of $\mu_i$ |
| Attention MIL | none | instance scoring | weighted mean | learned first moment |
| Set Transformer | complete learned relations | set attention | PMA or pool | interaction-aware set summary |
| Sequence SSM | order $\sigma_i$ | scan | final state or pool | compressed trajectory |
| Spatial GNN | adjacency $A_i$ | message passing | graph pool | contextualized node statistic |
| Heterogeneous graph | typed nodes and edges | typed message passing | multitype pool | typed relational summary |

## What This Decomposition Buys

The same readout means different things under different context operators:

```math
\frac{1}{n}\sum_j h_j
```

is a raw first moment if $\mathcal{C}$ is identity, but:

```math
\frac{1}{n}\sum_j h_j^{(L)}
```

is a first moment of contextualized states if $\mathcal{C}$ is a GNN,
Transformer, or sequence operator.

Thus the question is not only:

```text
What readout is used?
```

but:

```text
What information was already mixed into the states before readout?
```
