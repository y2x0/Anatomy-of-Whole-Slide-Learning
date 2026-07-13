# Hierarchical Graph Attribution

HACT-Net represents a slide with cell-level graphs nested inside tissue-region
graphs.

## 1. Assignment Operator

Let `B` assign cells to tissue regions:

```math
B_{rc}
=\mathbf 1\{\text{cell }c\text{ belongs to region }r\}.
```

Cell states `H_cell` transfer to region states through a normalized operator:

```math
H_{\mathrm{reg}}^{(0)}
=D_B^{-1}BH_{\mathrm{cell}}.
```

The assignment is an information bottleneck before region-level message
passing.

## 2. Cell-to-Region Credit

For a scalar region score, a local derivative path is

```math
\frac{\partial F}{\partial h_c}
=\sum_r
\frac{\partial F}{\partial h_r^{(0)}}
\frac{\partial h_r^{(0)}}{\partial h_c}.
```

For fixed mean assignment, the direct coefficient is `B_rc / deg(r)`. It is not
the full cell importance after graph context and nonlinear region pooling.

## 3. Region-to-Slide Credit

Let `s_r` be a region-level removal or attribution score. A cell heatmap induced
by region explanation is

```math
s_c^{\mathrm{lift}}
=\sum_rB_{rc}s_r.
```

All cells in one region inherit the same region score. This is an intentional
resolution choice, not evidence that each cell contributed equally.

## 4. Hierarchical Counterfactual

Deleting one cell can affect the region state; deleting a region removes all its
cells and changes the region graph. The two effects should be reported
separately:

```math
\Delta_c^{\mathrm{cell}}
=F(G)-F(G\setminus c),
\qquad
\Delta_r^{\mathrm{region}}
=F(G)-F(G\setminus r).
```

## 5. Failure Mode

A bad cell-to-region assignment can make a biologically meaningful cell
unexplainable at slide level. Better region-level attribution cannot recover
information discarded by the assignment operator.

