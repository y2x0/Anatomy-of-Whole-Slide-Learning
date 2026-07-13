# C/R/G/S Survival Decomposition

Use:

```math
\widetilde{H}_i=\mathcal{C}(H_i;G_i,S_i),
\qquad
z_i=\mathcal{R}(\widetilde{H}_i),
\qquad
\widehat{Y}_i=\mathcal{H}(z_i).
```

For survival,
```math
S_i
```
in the operator arguments should be read as supervision
structure, not survival function.

## C: Context Operator

Examples:

```text
none
attention
transformer
graph
state-space
co-attention
causal graph filtering
```

Mathematical role:

```math
H_i\mapsto\widetilde{H}_i.
```

## R: Readout Operator

Examples:

```text
mean
attention pooling
CLS token
graph readout
prototype distribution
time-conditioned attention
cause-conditioned attention
```

Mathematical role:

```math
\widetilde{H}_i\mapsto z_i
```

or:

```math
\widetilde{H}_i\mapsto z_i(t),z_{ic},z_{ikc}.
```

## G: Geometry

Examples:

```text
no geometry
coordinates
grid
graph
sequence order
hierarchy
hyperbolic space
topology
```

Geometry defines which interactions are easy or possible.

## S: Supervision And Censoring

Observed:

```math
X_i=\min(T_i,C_i),
\qquad
\delta_i=\mathbf{1}[T_i\le C_i].
```

Loss:

```math
\ell(\widehat{Y}_i,X_i,\delta_i).
```

For competing risks:

```math
\ell(\widehat{Y}_i,X_i,\Delta_i).
```

## Dense Map

```math
\boxed{
H_i
\xrightarrow{\mathcal{C}(G)}
\widetilde{H}_i
\xrightarrow{\mathcal{R}}
z_i
\xrightarrow{\mathcal{H}}
\mathcal{Y}_i^{\text{surv}}
\xrightarrow{\ell(S)}
\mathcal{L}
}
```

The architecture determines the statistic. The survival loss determines what
that statistic is trained to mean.
