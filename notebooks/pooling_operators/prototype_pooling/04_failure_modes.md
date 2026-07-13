# Prototype Pooling Failure Modes

Prototype pooling is clean because it exposes a finite morphology dictionary.
It fails when that dictionary or its assignments do not match the task.

## 1. Prototype Collision

If two biologically different morphologies map to the same prototype:

```math
q_m(h_a)\approx 1,
\qquad
q_m(h_b)\approx 1,
```

then prevalence cannot distinguish them:

```math
p_{im}
\ \text{mixes both morphologies}.
```

The slide statistic is interpretable only if prototypes are semantically
coherent.

## 2. Too Many Prototypes

If
```math
M
```
is very large, assignments become sparse:

```math
p_{im}
\approx
0
```

for many
```math
m
```
. Estimates become noisy, and downstream heads can overfit rare
prototype coordinates.

## 3. Too Few Prototypes

If
```math
M
```
is too small:

```math
q_m(h)
```

must absorb heterogeneous patches. The representation becomes a coarse histogram
that may erase diagnostic subtypes.

## 4. Lost Spatial Layout

Two slides can have the same prototype prevalence:

```math
p_i=p_k
```

but different spatial arrangement:

```text
slide i:
    prototype A clustered inside tumor

slide k:
    prototype A scattered across benign tissue
```

Prototype pooling cannot distinguish them unless coordinates, graph context, or
region-wise summaries are added before readout.

## 5. Prototype Drift Across Cohorts

If prototypes are learned on a source cohort:

```math
\{c_m\}_{m=1}^{M}
\sim
\mathcal{D}_{\text{source}},
```

then target-cohort patches may assign poorly:

```math
h\sim\mathcal{D}_{\text{target}}
\quad
\Rightarrow
\quad
\max_m q_m(h)\ \text{small or misleading}.
```

The model may still output a simplex vector, but the coordinates no longer mean
the same morphology.

## 6. Residual Noise

Residual statistics:

```math
\bar r_{im}
=
\frac{1}{n_i}
\sum_j\gamma_{ijm}\Sigma_m^{-1/2}(h_{ij}-\mu_m)
```

are informative only when enough patch mass is assigned to component
```math
m
```
. If:

```math
\sum_j\gamma_{ijm}
\approx
0,
```

then residual estimates are unstable.

## Dense Summary

Prototype pooling assumes:

```text
1. recurring morphology can be discretized
2. prototype assignments are stable
3. prevalence or residuals are task-relevant
4. layout is not essential unless added elsewhere
```

It is strongest for diffuse composition-like phenotypes. It is weakest for rare
events, spatial motifs, and cohort shifts that move morphology away from the
learned dictionary.
