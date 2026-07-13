# Graph Spatial C/R/G/S and Failure Matrix

## 1. Unified Placement

```math
G_i=\mathcal G(X_i,C_i;S_i),
\qquad
\widetilde H_i=\mathcal C(H_i;G_i),
\qquad
z_i=\mathcal R(\widetilde H_i;G_i),
\qquad
F_i=\mathcal H(z_i).
```

Graph explanations can target construction, context, readout, or the final
head:

| Object | Primary quantity | What it supports |
|---|---|---|
| node | removal loss or node attribution | region dependence |
| edge | edge deletion or interaction score | relation dependence |
| path | message derivative | route through context |
| type | type deletion or pooled block | entity-family dependence |
| topology | support intervention | geometry hypothesis |
| feature | integrated gradients | interpretable region descriptor |

## 2. Failure Matrix

| Failure | Mathematical cause | Diagnostic |
|---|---|---|
| attention-credit confusion | routing weight treated as signed score effect | node/edge intervention |
| feature-edge confusion | feature kNN called spatial adjacency | coordinate-preserving topology swap |
| fixed/rebuilt ambiguity | deletion changes support in one implementation only | report both graph operators |
| pseudo-label shortcut | pooled type is wrong or label-correlated | type perturbation and expert audit |
| bridge-node artifact | deleting a structural bridge disconnects graph | connectivity-preserving deletion |
| oversmoothing | deep message passing erases node distinctions | layer-wise similarity and deletion stability |
| rendering illusion | interpolation creates smooth map not present in graph | inspect raw node scores |
| hierarchy bottleneck | cell information lost in assignment | cell versus region attribution |
| stain shortcut | attention follows acquisition/stain rather than biology | stain-balanced topology and counterfactual swap |
| topology nonidentifiability | different graphs yield same slide score | graph ensemble and support perturbation |

## 3. Minimum Reporting Rule

Every graph heatmap should state:

```text
node definition; edge semantics; graph construction; whether support was fixed;
target score or loss; signed convention; message layers included; pooling
operator; coordinate rendering; and whether the claim is model-relative or
biological.
```

Without these fields, a graph heatmap hides the very geometry it claims to
explain.

## Fixed Versus Rebuilt Graph Interventions

Let the predictor be

```math
F(H,A)
=
\mathcal H\left(
\mathcal R\left(\mathcal C(H,A),A\right)
\right).
```

Deleting node j has at least two distinct meanings. A fixed-support intervention
masks the node while retaining the original adjacency:

```math
\Delta_j^{\mathrm{fixed}}
=
F(H,A)-F(H^{(-j)},A).
```

A rebuilt-support intervention removes the node and reconstructs the graph:

```math
A^{(-j)}
=
\mathrm{Build}(X_{\setminus j},C_{\setminus j};S),
\qquad
\Delta_j^{\mathrm{rebuild}}
=
F(H,A)-F(H^{(-j)},A^{(-j)}).
```

Their difference measures topology dependence induced by the graph constructor:

```math
\Delta_j^{\mathrm{topo}}
=
\Delta_j^{\mathrm{rebuild}}
-
\Delta_j^{\mathrm{fixed}}.
```

Reporting only one of these effects hides whether the explanation is about the
node feature, its messages, or the fact that its removal changes the support.

## Edge, Path, And Type Effects

For an edge e, an edge intervention changes support or edge attributes while
holding node features fixed:

```math
\Delta_e
=
F(H,A)-F(H,A^{(-e)}).
```

For a message path p from source u to target v, a chain-rule attribution has the
form

```math
\frac{\partial F}{\partial h_u}\bigg|_{p}
=
\prod_{(r\to s)\in p}
\frac{\partial h_s^{(\ell+1)}}{\partial h_r^{(\ell)}}.
```

This is a local sensitivity along a computational route, not automatically a
unique causal path. For a node type t, compare a type-specific readout with a
masked type intervention:

```math
\Delta_t
=
F(H,A)-F(H_{\setminus t},A_{\setminus t}).
```

The support, message operator, readout, and target must all be held or changed
according to the stated intervention semantics.

## Topology Nonidentifiability

Different graphs can produce the same slide prediction:

```math
F(H,A)=F(H,A')
\quad\text{for}\quad A\ne A'.
```

Therefore a graph recovered from slide-level supervision is not identified by the
prediction loss alone. A topology claim needs support perturbations, external
geometry, or node/edge-level supervision that distinguishes the competing graphs.
