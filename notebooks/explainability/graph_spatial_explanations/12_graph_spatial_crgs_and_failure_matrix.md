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
