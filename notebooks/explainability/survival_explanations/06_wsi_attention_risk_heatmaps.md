# WSI Attention Risk Heatmaps

## 1. Attention Readout

For patch states `r_ij`,

```math
a_{ij}
=\frac{\exp(q_{ij})}{\sum_\ell\exp(q_{i\ell})},
\qquad
z_i=\sum_j a_{ij}r_{ij}.
```

An attention heatmap visualizes `a_ij` over patch coordinates.

## 2. Exact Head Credit

For a scalar Cox head, the signed term is

```math
\gamma_{ij}=a_{ij}w^{\top}r_{ij}.
```

For a nonlinear survival head, attention alone cannot determine score credit.
Use gradient, integrated-gradient, or deletion analyses targeted at the output
object.

## 3. Attention Collapse

If one patch receives nearly all mass,

```math
\max_j a_{ij}\approx1,
```

the risk statistic becomes approximately the state of one patch. This may be
appropriate for sparse positive morphology or may reflect a stain/artifact
shortcut. Entropy and seed stability are necessary diagnostics.

## 4. Spatial Claim

Mapping attention to coordinates is meaningful only when patch coordinates are
retained and the graph or set model has not discarded their geometry. A smooth
overlay is a visualization of weights, not evidence of spatial reasoning.

## 5. Survival-Specific Reporting

State whether the map explains relative risk, a hazard bin, a survival horizon,
or a cause-specific incidence target. The colorbar should name the quantity and
sign.

