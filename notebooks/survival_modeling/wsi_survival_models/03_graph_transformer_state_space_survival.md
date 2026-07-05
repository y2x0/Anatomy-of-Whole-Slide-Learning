# Graph, Transformer, And State-Space Survival

Context-heavy WSI survival models modify patch embeddings before readout:

```math
\widetilde{H}_i=\mathcal{C}(H_i;G_i).
```

The survival head still sees:

```math
z_i=\mathcal{R}(\widetilde{H}_i).
```

## Graph Survival

Represent the slide as:

```math
G_i=(V_i,E_i),
\qquad
V_i=\{1,\ldots,n_i\}.
```

Node states:

```math
h_{ij}^{(0)}=h_{ij}.
```

Message passing:

```math
h_{ij}^{(\ell+1)}
=
\phi_\ell
\left(
h_{ij}^{(\ell)},
\operatorname*{AGG}_{k\in\mathcal{N}(j)}
\psi_\ell(h_{ij}^{(\ell)},h_{ik}^{(\ell)},e_{jk})
\right).
```

Readout:

```math
z_i=\mathcal{R}(\{h_{ij}^{(L)}\}_{j\in V_i}).
```

Survival:

```math
\eta_i=f(z_i)
\quad\text{or}\quad
h_i=f(z_i).
```

Patch-GCN fits this family: WSI patches become nodes in a spatial graph, and
survival risk is predicted from context-aware patch representations.

## Transformer Survival

Transformer context:

```math
\widetilde{H}_i
=
\operatorname{Transformer}(H_i+P_i).
```

Readout:

```math
z_i=\widetilde{h}_{i,\operatorname{cls}}
\quad\text{or}\quad
z_i=\frac{1}{n_i}\sum_j\widetilde{h}_{ij}.
```

Self-attention can model long-range patch interactions:

```math
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V.
```

For survival, the transformer changes the context operator $\mathcal{C}$. The
risk object is still defined by the survival head.

## State-Space Survival

If patches are ordered into a sequence:

```math
(h_{i1},\ldots,h_{in_i}),
```

a state-space model computes:

```math
s_{j+1}=A_js_j+B_jh_{ij},
\qquad
\widetilde{h}_{ij}=C_js_j+D_jh_{ij}.
```

Selective SSMs make $A_j,B_j,C_j$ input-dependent.

Survival readout:

```math
z_i=\mathcal{R}(\{\widetilde{h}_{ij}\}_{j=1}^{n_i}).
```

The ordering becomes part of the model. If the order is arbitrary, the state
space model may learn ordering artifacts.

## Context Does Not Remove Readout Bottleneck

Even with graph or transformer context:

```math
\eta_i=w^\top z_i
```

collapses the slide to one scalar.

If:

```math
z_i=\frac{1}{n_i}\sum_jh_{ij}^{(L)},
```

then the final statistic is still a first moment, but over contextualized nodes.

## Dense Summary

```math
H_i
\xrightarrow{\mathcal{C}_{\operatorname{graph/attn/ssm}}}
\widetilde{H}_i
\xrightarrow{\mathcal{R}}
z_i
\xrightarrow{\mathcal{H}_{\operatorname{surv}}}
\eta_i,h_i,\lambda_i(t).
```

Context operators decide which interactions can be represented. Readout decides
which of those interactions survive into the survival loss.
