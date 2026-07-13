# Region-Graph Integrated Gradients

The mapped interpretable region-graph work builds a graph whose nodes have
clinically motivated texture, morphology, and nuclear features, then applies a
GAT and integrated-gradients visualization.

## 1. Region Construction

Start with a region-adjacency graph

```math
G^0=(V^0,E^0).
```

Adjacent regions with embedding similarity above threshold `tau` are merged
agglomeratively:

```math
s_{ij}
=\frac{h_i^{\top}h_j}{\|h_i\|_2\|h_j\|_2},
\qquad
(i,j)\in E^0.
```

The final region graph preserves spatial contiguity while reducing node count.

## 2. Interpretable Node Features

Each final region receives a feature vector

```math
x_v
=\left[
x_v^{\mathrm{texture}},
x_v^{\mathrm{morphology}},
x_v^{\mathrm{nuclear}}
\right].
```

The explanation is therefore grounded at the feature definition stage, before
GAT context is added.

## 3. Integrated Gradients

For scalar class score `F_c(X)` and baseline `X_0`, the node-feature attribution
for feature `r` is

```math
\mathrm{IG}_{v,r}
=(X_{v,r}-X_{0,v,r})
\int_0^1
\frac{\partial F_c
(X_0+\alpha(X-X_0))}
{\partial X_{v,r}}
d\alpha.
```

Completeness gives

```math
\sum_{v,r}\mathrm{IG}_{v,r}
=F_c(X)-F_c(X_0)
```

under the usual differentiability and path assumptions.

## 4. What It Explains

Summing `IG_vr` across features gives a signed region score. The region is
interpretable by construction, while the attribution remains baseline- and
path-dependent. A region heatmap can be more semantically grounded than a raw
patch attention map, but it still explains the trained graph score rather than
clinical causality.

