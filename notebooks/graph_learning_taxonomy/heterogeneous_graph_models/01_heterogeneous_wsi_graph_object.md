# Heterogeneous WSI Graph Object

Chan et al. represent a WSI as a heterogeneous graph:

```math
G_i
=
\left(
V_i,
E_i,
\tau_i,
\phi_i
\right).
```

Here:

```text
V_i:
    patch nodes

E_i:
    feature-kNN patch-patch relation

tau_i:
    node-type assignment into the pseudo-label vocabulary

phi_i:
    edge-attribute assignment into the edge-attribute space
```

The type and edge-attribute maps are:

```math
\tau_i:
V_i
\to
\mathcal{T},
\qquad
\phi_i:
E_i
\to
\mathcal{B}.
```

Each node has a feature:

```math
h_{iv}^{(0)}
\in
\mathbb{R}^{d}.
```

So the WSI object is:

```math
\left(
\{h_{iv}^{(0)}\}_{v\in V_i},
\{\tau_i(v)\}_{v\in V_i},
\{e=(s,t)\in E_i\},
\{\phi_i(e)\}_{e\in E_i}
\right).
```

## Node Ontology

The nodes are patches:

```math
v
\leftrightarrow
\text{patch }x_{iv}.
```

This is not a cell graph. A patch may contain zero, one, or many nuclei.

The node type is derived by:

```text
HoVer-Net nuclei detection and classification
majority vote over nuclei inside the patch
patch-level pseudo-label
```

Thus:

```math
\tau_i(v)
=
\mathrm{MajorityVote}
\left(
\{\text{nucleus types inside patch }v\}
\right).
```

The figure in the paper illustrates types such as:

```text
no label
neoplastic
inflammatory
connective
dead
non-neoplastic epithelial
```

The mathematical role of `tau` is that the message-passing layer can use
node-type-specific projections.

## Edge Ontology

Edges are not coordinate adjacency edges in the main construction.

The paper uses feature-space kNN. For the message-passing formulas in these
notes, orient the relation as source-neighbor to target:

```math
E_i
=
\left\{
(u,v):
u\in\mathcal{K}_k(v)
\right\},
```

where:

```math
\mathcal{K}_k(v)
```

contains nodes whose node features are among the `k` most similar to node `v`.
The paper text does not make edge orientation central. If an implementation
stores query-to-neighbor edges `(v,u)`, the message-passing convention above is
the transposed adjacency.

This is different from Patch-GCN:

```text
Patch-GCN:
    coordinate kNN

Chan et al.:
    feature-similarity kNN
```

The edge attribute is computed from the endpoint node features using
Pearson correlation:

```math
\phi_i(u,v)
=
\rho_{\mathrm{Pearson}}
\left(
h_{iu}^{(0)},
h_{iv}^{(0)}
\right).
```

The paper uses this as encoder-space similarity between patch nodes.

## Heterogeneity

There are two heterogeneity sources:

```text
node heterogeneity:
    tau(v) is a pseudo-label nucleus type

edge heterogeneity:
    phi(e) is a continuous encoder-similarity attribute
```

This differs from standard relational GNNs where edge types are usually finite
and discrete:

```math
r(e)
\in
\{1,\dots,R\}.
```

Here:

```math
\phi(e)
\in
\mathbb{R}
```

for the Pearson-correlation construction, while the HEAT formulation allows
continuous or higher-dimensional edge attributes.

## Factorization

The model factors through the constructed graph:

```math
\Gamma_{\mathrm{hetero}}(X_i)
=
G_i,
```

```math
\widehat y_i
=
f_\theta(G_i).
```

The construction map contains:

```text
OTSU tissue filtering
sliding-window patch extraction
background removal
KimiaNet patch feature extraction
HoVer-Net nucleus-type pseudo-labeling
feature-space kNN graph support
Pearson-correlation edge attributes
```

The graph learner does not receive the raw WSI directly after this map. HEAT is
the context operator applied after this heterogeneous graph is constructed; it
is not the WSI-to-graph construction map.

## C/R/G/S Placement

```text
\mathcal{G}:
    typed patch graph with continuous edge attributes

\mathcal{C}:
    HEAT message passing

\mathcal{R}:
    PL pooling, then graph-level readout

\mathcal{S}:
    slide-level label for cancer staging, classification, or typing
```
