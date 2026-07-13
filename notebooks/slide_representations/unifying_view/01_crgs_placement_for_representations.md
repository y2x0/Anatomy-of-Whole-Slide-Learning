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

where
```math
\Omega_i
```
is the structure assigned to the slide.

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

The important point is that
```math
\Omega_i
```
changes which functions are natural,
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
\mathcal{C}_{\text{set}}(H_i),
\qquad
z_i
=
\mathcal{R}_{\text{inv}}(\widetilde{H}_i).
```

If there is no interaction before readout:

```math
\mathcal{C}_{\text{set}}(H_i)=H_i.
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

The surviving statistic is a functional of
```math
\mu_i
```
.

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
\mathcal{C}_{\text{seq}}
(h_{i\sigma_i(1)},\ldots,h_{i\sigma_i(n_i)})_t.
```

Then:

```math
z_i
=
\mathcal{R}(u_{i1},\ldots,u_{in_i}).
```

The order
```math
\sigma_i
```
 is part of the representation. Changing
```math
\sigma_i
```
changes
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
\mathrm{GNN}_\theta(H_i,A_i,E_i).
```

Readout gives:

```math
z_i
=
\mathcal{R}_{\text{graph}}(G_i,\widetilde{H}_i).
```

The valid symmetry is graph relabeling equivariance:

```math
\mathrm{GNN}(PH_i,PA_iP^\top)
=
P\mathrm{GNN}(H_i,A_i).
```

The graph is not neutral preprocessing. It defines possible information flow.

## Hierarchy Representation

The hierarchy view chooses nested levels:

```math
\Omega_i
=
\left(
\{V_i^{(\ell)}\}_{\ell=0}^{L},
\{\pi_i^{(\ell)}\}_{\ell=0}^{L-1}
\right).
```

The context operator has scale-aware pieces:

```math
\mathcal{C}
=
\mathcal{C}_{\text{bottom-up}}
\circ
\mathcal{C}_{\text{lateral}}
\circ
\mathcal{C}_{\text{top-down}}.
```

Readout may use only the top token:

```math
z_i=h_{\text{slide}}^{(L)}.
```

or several scales:

```math
z_i
=
\psi_\theta
\left(
\mathcal{R}^{(0)}(H_i^{(0)}),
\ldots,
\mathcal{R}^{(L)}(H_i^{(L)})
\right).
```

The hierarchy assumes that fine morphology composes into larger tissue units.

## Distribution Representation

The distribution view chooses:

```math
\Omega_i=\mu_i,
\qquad
\mu_i
=
\frac{1}{n_i}
\sum_j\delta_{h_{ij}}.
```

The readout is a statistic of a measure:

```math
z_i=T(\mu_i).
```

Examples:

```math
T(\mu_i)
=
\int \phi(h)\,d\mu_i(h),
```

```math
T(\mu_i)
=
\left(
\int q_1(h)\,d\mu_i(h),
\ldots,
\int q_M(h)\,d\mu_i(h)
\right).
```

The distribution view assumes morphology prevalence or distribution shape is the
right object.

## Retrieval Memory Representation

The retrieval view chooses an external memory:

```math
\Omega_i=\mathcal{M}
=
\{(k_r,v_r)\}_{r=1}^{N}.
```

The slide first becomes a query:

```math
q_i=Q(S_i).
```

Then memory context is:

```math
\mathcal{M}(q_i)
=
\{(k_r,v_r):r\in\mathcal{N}_K(i)\}.
```

The representation is:

```math
\mathcal{X}_i=(q_i,\mathcal{M}(q_i)).
```

Retrieval makes the database part of the mathematical object.

## Foundation Latent Representation

The foundation-latent view chooses a pretrained map:

```math
\Omega_i=F_{\text{FM}}.
```

The slide object is:

```math
\mathcal{X}_i
=
F_{\text{FM}}(S_i)
```

or patch tokens in pretrained latent space:

```math
\mathcal{X}_i
=
\{E_{\text{FM}}(x_{ij})\}_{j=1}^{n_i}.
```

The inherited geometry is:

```math
d_{\text{FM}}(a,b)
=
\|F_{\text{FM}}(a)-F_{\text{FM}}(b)\|.
```

This geometry is shaped by pretraining, not by the downstream task alone.

## Placement Table

| Representation | \Omega_i | \mathcal{C} | \mathcal{R} | Surviving Statistic |
|---|---:|---:|---:|---:|
| Set mean MIL | none | identity | mean | first moment of \mu_i |
| Attention MIL | none | instance scoring | weighted mean | learned first moment |
| Set Transformer | complete learned relations | set attention | PMA or pool | interaction-aware set summary |
| Sequence SSM | order \sigma_i | scan | final state or pool | compressed trajectory |
| Spatial GNN | adjacency A_i | message passing | graph pool | contextualized node statistic |
| Heterogeneous graph | typed nodes and edges | typed message passing | multitype pool | typed relational summary |
| Hierarchy | parent maps \pi_i | multiscale context | top or multiscale readout | scale-composed summary |
| Distribution | empirical measure \mu_i | optional statistic map | T(\mu_i) | measure statistic |
| Retrieval memory | external memory \mathcal{M} | nearest-neighbor context | query plus retrieved values | archive-conditioned statistic |
| Foundation latent | pretrained map F_{\text{FM}} | frozen or adapted encoder | probe or slide head | pretrained latent coordinate |

## What This Decomposition Buys

The same readout means different things under different context operators:

```math
\frac{1}{n}\sum_j h_j
```

is a raw first moment if
```math
\mathcal{C}
```
is identity, but:

```math
\frac{1}{n}\sum_j h_j^{(L)}
```

is a first moment of contextualized states if
```math
\mathcal{C}
```
is a GNN,
Transformer, or sequence operator.

Thus the question is not only:

```text
What readout is used?
```

but:

```text
What information was already mixed into the states before readout?
```
