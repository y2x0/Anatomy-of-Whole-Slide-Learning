# Spatial Rendering and Topology Counterfactuals

## 1. Coordinate Lift

Let node `v` correspond to region `R_v` with coordinates `c_v` and explanation
score `s_v`. A heatmap is a rendering operator:

```math
\mathcal H(x,y)
=\sum_{v\in V}s_vK((x,y),R_v),
```

where `K` spreads a region score over pixels. Rendering can smooth, overlap, or
clip scores; it does not alter the graph computation that produced `s_v`.

## 2. Spatial Equivariance Test

If a model uses coordinates only through an equivariant graph construction, a
rigid transformation `T` should satisfy

```math
F(TX,TG)=F(X,G)
```

up to any task-relevant coordinate effects. A heatmap should transform as

```math
\mathcal H_{TX}(Tx,Ty)
=\mathcal H_X(x,y).
```

Violation can indicate coordinate shortcuts, padding artifacts, or graph
construction instability.

## 3. Topology Intervention

For fixed node features, compare two graph supports:

```math
\Delta_{A\to A'}
=F(X,A)-F(X,A').
```

Examples include spatial adjacency versus feature k-nearest neighbors, random
rewiring with preserved degree, or removal of a cross-region edge family.

This tests dependence on a topology hypothesis, not the importance of one
highlighted node.

## 4. Same Multiset, Different Layout

Construct two slides with identical node features but different coordinates and
adjacency:

```math
\{x_v\}_{v\in V_1}=\{x_v\}_{v\in V_2},
\qquad
A_1\ne A_2.
```

A set model must be invariant; a graph model may change. This is the cleanest
sanity check for whether spatial geometry survives the representation.

