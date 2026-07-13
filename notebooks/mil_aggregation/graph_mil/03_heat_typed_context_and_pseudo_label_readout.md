# HEAT Typed Context And Pseudo-Label Readout

Source:

```text
Chan et al., Histopathology Whole Slide Image Analysis With Heterogeneous
Graph Representation Learning, CVPR 2023.
https://arxiv.org/abs/2307.04189
```

## 1. Heterogeneous Patch Graph

For slide `i`, Chan et al. construct:

```math
G_i
=
\left(
V_i,E_i,\tau_i,\phi_i,H_i^{(0)}
\right).
```

The node type map is:

```math
\tau_i:V_i\to\mathcal{T},
```

where patch-level pseudo-types are obtained from HoVer-Net nuclear predictions
and a majority-vote construction. The edge support is feature-space kNN:

```math
E_i
=
\left\{
(u,v):u\in\mathcal{K}_k(v)
\right\}.
```

The edge attribute map is:

```math
\phi_i:E_i\to\mathcal{B},
```

with the main scalar attribute derived from Pearson correlation of endpoint
KimiaNet embeddings:

```math
\phi_i(u,v)
=
\rho_{\mathrm{Pearson}}
\left(
h_{iu}^{(0)},h_{iv}^{(0)}
\right).
```

This is a typed, attributed graph. It is not the same geometry as Patch-GCN:

```text
Patch-GCN:
    coordinate kNN support

HEAT:
    feature kNN support plus pseudo-types and Pearson edge attributes
```

## 2. HEAT Context

At layer `ell`, node `v` has state `h_iv^(ell-1)` and edge `e=(u,v)` has state
`b_ive^(ell-1)`. For head `m`, type-specific projections are:

```math
q_{iv}^{m}
=
h_{iv}^{(\ell-1)}W_{Q,\tau_i(v)}^{m},
```

```math
k_{iu}^{m}
=
h_{iu}^{(\ell-1)}W_{K,\tau_i(u)}^{m},
\qquad
v_{iu}^{m}
=
h_{iu}^{(\ell-1)}W_{V,\tau_i(u)}^{m}.
```

An edge projection can be written:

```math
g_{iuv}^{m}
=
b_{iuv}^{(\ell-1)}W_E^{m}.
```

The dimension-safe invariant is that the edge state modulates source-target
compatibility. One realization is:

```math
a_{iuv}^{m}
=
\frac{
\left(k_{iu}^{m}\odot g_{iuv}^{m}\right)
\left(q_{iv}^{m}\right)^{\top}
}{\sqrt{d_h}}.
```

The neighbor distribution is normalized per target:

```math
\alpha_{iuv}^{m}
=
\frac{\exp(a_{iuv}^{m})}
{\sum_{r\in\mathcal{N}_i(v)}\exp(a_{irv}^{m})}.
```

The head message is:

```math
m_{iv}^{m}
=
\sum_{u\in\mathcal{N}_i(v)}
\alpha_{iuv}^{m}v_{iu}^{m}.
```

Heads are concatenated and mapped to the next node state. Edge states may also
be updated by an edge transformer. The paper-level point is not an arbitrary
choice of tensor contraction; it is the coupling:

```text
node pseudo-type chooses projection family;
edge attribute changes compatibility;
feature-kNN determines support.
```

## 3. Pseudo-Label Pooling

After `L` HEAT layers:

```math
\widetilde H_i
=
\left\{
\widetilde h_{iv}:v\in V_i
\right\}.
```

For type `a`, define:

```math
V_{ia}
=
\left\{
v\in V_i:\tau_i(v)=a
\right\}.
```

The type-wise readout is:

```math
z_{ia}
=
\mathcal{R}_a
\left(
\left\{
\widetilde h_{iv}:v\in V_{ia}\right\}
\right).
```

Stacking in a fixed teacher-vocabulary order gives:

```math
Z_i^{\mathrm{PL}}
=
\begin{bmatrix}
z_{ia_1}^{\top}\\
\vdots\\
z_{ia_{|\mathcal{T}|}}^{\top}
\end{bmatrix}
\in
\mathbb{R}^{|\mathcal{T}|\times d_{\mathrm{PL}}}.
```

An empty type requires a mask or explicit convention:

```math
m_{ia}
=
\mathbf{1}\left\{V_{ia}\neq\varnothing\right\}.
```

The final graph vector is some readout of the typed rows and mask:

```math
z_i
=
\mathcal{R}_{\mathrm{graph}}
\left(
Z_i^{\mathrm{PL}},m_i
\right).
```

The row identity is stable across slides because row `a` means the same
teacher pseudo-label vocabulary entry. This removes learned-cluster label
switching, but it does not preserve within-type spatial arrangement or the
uncertainty of the teacher.

## 4. Why This Is A Distinct MIL Readout

Ordinary attention pooling produces one vector directly:

```math
z_i
=
\sum_{v\in V_i}\alpha_{iv}\widetilde h_{iv}.
```

PL pooling first partitions by an external semantic index and produces a
structured statistic:

```math
\left\{
\mathcal{R}_a
\left(
\widetilde H_{ia}
\right)
\right\}_{a\in\mathcal{T}}.
```

The surviving statistic is therefore:

```text
type-conditioned summaries plus the fixed type coordinate system.
```

It is not a full distribution over patches. Two slides can have the same typed
rows while differing in the internal arrangement of patches within each type.

## 5. C/R/G/S Placement

```text
C:
    HEAT typed edge-attribute transformer over feature-kNN neighbors

R:
    pseudo-label type-wise pooling followed by graph-level readout

G:
    feature similarity, node pseudo-types, and Pearson edge attributes

S:
    slide classification/staging labels; HoVer-Net pseudo-types are upstream
    teacher supervision rather than slide labels
```

The main failure boundary is teacher error. A semantically stable row index is
not the same as a correct biological label.
