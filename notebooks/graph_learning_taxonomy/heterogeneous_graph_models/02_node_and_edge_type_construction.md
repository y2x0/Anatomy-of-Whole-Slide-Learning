# Node And Edge Type Construction

The construction map is:

```math
X_i
\xrightarrow{\Gamma_{\mathrm{hetero}}}
G_i.
```

It has four main stages:

```text
patch extraction
node feature extraction
node pseudo-labeling
edge and edge-attribute construction
```

## Patch Nodes

The WSI is first filtered and tiled:

```math
X_i
\longmapsto
\{x_{iv}\}_{v=1}^{N_i}.
```

The paper uses:

```text
OTSU thresholding
sliding-window non-overlapping patches
background patch removal
```

Each remaining patch becomes one graph node:

```math
V_i
=
\{1,\dots,N_i\}.
```

For TCGA datasets, the paper reports around:

```math
N_i
\approx
300
```

patch nodes per WSI on average. For Camelyon16, it reports around:

```math
N_i
\approx
5000.
```

These are sampled patch graphs, not exhaustive pixel-level WSI graphs.

## Node Features

Each patch is encoded by a pretrained feature encoder, KimiaNet:

```math
h_{iv}^{(0)}
=
f_{\mathrm{KimiaNet}}(x_{iv})
\in
\mathbb{R}^{d}.
```

The node feature matrix is:

```math
H_i^{(0)}
=
\begin{bmatrix}
h_{i1}^{(0)}\\
\vdots\\
h_{iN_i}^{(0)}
\end{bmatrix}
\in
\mathbb{R}^{N_i\times d}.
```

The graph construction and edge attributes depend on these embeddings.

## Node Pseudo-Labels

For each patch, HoVer-Net detects nuclei and assigns nucleus types. The paper
uses HoVer-Net pretrained on PanNuke.

Let:

```math
\mathcal{N}_{iv}^{\mathrm{nuc}}
```

be the set of detected nuclei inside patch `v`. Let:

```math
c_{ivu}^{\mathrm{nuc}}
\in
\mathcal{T}
```

be the predicted nucleus type for nucleus `u` in patch `v`.

The patch node type is:

```math
\tau_i(v)
=
\mathrm{mode}
\left(
\{c_{ivu}^{\mathrm{nuc}}:
u\in\mathcal{N}_{iv}^{\mathrm{nuc}}\}
\right).
```

This is majority-vote compression. If the patch contains mixed nuclei, only the
dominant type survives as the node type.

The paper's figure includes a `no label` category. A mathematical convention
consistent with that figure is:

```math
\tau_i(v)
=
\mathrm{no\ label}
```

for patches without detected nuclei.

## Feature-Space kNN Edges

For each query node `v`, define:

```math
\mathcal{K}_k(v)
=
\text{the }k\text{ nodes most similar to }v
\text{ under the implementation's feature similarity}.
```

For the message-passing formulas in these notes, orient feature-kNN support as
source-neighbor to target:

```math
E_i
=
\{(u,v):u\in\mathcal{K}_k(v)\}.
```

The paper says these are the nodes with the most similar features. This is not
spatial coordinate kNN. If an implementation stores the opposite directed edge,
the HEAT formulas use the transposed adjacency.

So a patch can be connected to a distant patch if their features are similar:

```math
\|c_{iv}^{\mathrm{coord}}-c_{iu}^{\mathrm{coord}}\|_2
\text{ can be large, while }
(u,v)\in E_i.
```

This is a major difference from coordinate graph methods.

The main paper separately defines Pearson correlation as the edge attribute.
It does not require us to identify the neighbor-search similarity and the edge
attribute as the same mathematical object. These notes keep them distinct:

```text
kNN support:
    chosen by feature similarity in the implementation

edge attribute:
    Pearson correlation between endpoint features
```

## Pearson Edge Attribute

For edge:

```math
e=(u,v),
```

the paper computes Pearson correlation between endpoint node features:

```math
r_{iuv}
=
\rho_{\mathrm{Pearson}}
\left(
h_{iu}^{(0)},
h_{iv}^{(0)}
\right).
```

Explicitly:

```math
\rho_{\mathrm{Pearson}}(u,v)
=
\frac{
\sum_{m=1}^{d}(u_m-\bar u)(v_m-\bar v)
}{
\sqrt{\sum_{m=1}^{d}(u_m-\bar u)^2}
\sqrt{\sum_{m=1}^{d}(v_m-\bar v)^2}
}.
```

Thus:

```math
\phi_i(u,v)
=
r_{iuv}.
```

The HEAT layer then transforms this edge attribute:

```math
b_{iuv}^{(0)}
=
\mathrm{Embed}_{\mathrm{edge}}
\left(
\phi_i(u,v)
\right).
```

For a scalar Pearson edge attribute:

```math
b_{iuv}^{(0)}
\in
\mathbb{R}^{d_{e,0}}.
```

The HEAT architecture itself is written more generally and can handle
continuous or high-dimensional edge attributes.

## The Full Constructed Object

The final graph object is:

```math
G_i
=
\left(
H_i^{(0)},
B_i^{(0)},
E_i,
\tau_i,
\phi_i
\right).
```

Expanded:

```math
G_i
=
\left(
\{h_{iv}^{(0)}\}_{v=1}^{N_i},
\{b_{iuv}^{(0)}\}_{(u,v)\in E_i},
\{(s,t)\in E_i\},
\{\tau_i(v)\}_{v=1}^{N_i},
\{r_{iuv}\}_{(u,v)\in E_i}
\right).
```

## What Is Already Lost

Before HEAT message passing starts, the construction has already discarded:

```text
raw pixels
within-patch minority nucleus types
exact nucleus counts per type, unless encoded elsewhere
spatial adjacency if it disagrees with feature-space similarity
edge semantics beyond Pearson feature correlation
```

The graph is therefore not simply "the WSI." It is a typed, feature-neighborhood
summary of sampled WSI patches.
