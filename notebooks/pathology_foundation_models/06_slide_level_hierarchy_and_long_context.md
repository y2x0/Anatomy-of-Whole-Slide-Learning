# Slide-Level Hierarchy and Long Context

## 1. Patch-to-Slide Composition

Prov-GigaPath and related slide foundation models factor the slide map into a
tile encoder and a slide encoder:

```math
h_j=f_{\phi}(x_j),
\qquad
z=F_{\psi}(\{h_j,c_j\}_{j=1}^{n}).
```

The slide encoder can use coordinates, tiling order, regional grouping, or a
long-context sequence operator.

## 2. Hierarchical Approximation

Partition patches into regions `r`:

```math
h_r=F_{\mathrm{region}}(\{h_j:j\in r\}),
\qquad
z=F_{\mathrm{slide}}(\{h_r\}_r).
```

This reduces quadratic patch interactions but inserts a region bottleneck. Two
slides that differ within a region can become identical after region pooling.

## 3. Coordinate Dependence

If the slide encoder uses coordinates, permutation invariance is intentionally
broken:

```math
F_{\psi}(\{h_j,c_j\})
\ne
F_{\psi}(\{h_{\pi(j)},c_j\})
```

for arbitrary permutation `pi`. The learned geometry must be tested against
coordinate shifts, missing tiles, and different tessellations.

## 4. Long Context Is Not Universal Context

Even a long-context encoder only transmits information through its attention,
state, or hierarchy. Complexity and memory impose a practical support. A slide
representation should state which patch pairs can interact before readout.

