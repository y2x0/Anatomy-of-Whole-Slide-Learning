# Paper Placement Matrix

This note places the pooling operators in the same C/R/G/S language.

## Matrix

| Method | $G$ | $\mathcal{C}$ | $\mathcal{R}$ | $S$ | Surviving Statistic |
|---|---|---|---|---|---|
| Mean pooling | none unless encoded upstream | identity or upstream context | normalized average | slide label or survival loss | first moment |
| Max pooling | none unless encoded upstream | instance scoring | maximum or smooth maximum | positive-instance bag label | extreme score |
| ABMIL | none in the readout | instance score and value maps | softmax-weighted sum | slide label | learned weighted first moment |
| DSMIL | none unless features encode it | instance classifier plus critical query | max score plus critical-instance weighted sum | slide label and pretraining signal | extreme score plus critical-centered mean |
| CLAM | none unless externally supplied | class-specific scoring plus pseudo-instance shaping | class-specific weighted sum | slide label plus top/bottom-k constraints | class-conditioned weighted first moment |
| PANTHER | prototype geometry or mixture covariance | GMM/prototype assignment | prevalence and residual statistics | unsupervised prototype learning plus task head | morphology distribution summary |

## Mean Pooling

```text
G:
    ignored by R

C:
    identity or upstream context

R:
    z_i = n_i^{-1} sum_j u_ij

S:
    downstream loss only
```

Mathematically:

```math
z_i
=
\int u\,d\mu_i(u).
```

Mean pooling preserves the first moment and discards sparse extremes,
multimodality, and layout.

## Max Pooling

```text
G:
    ignored by R

C:
    instance scoring map g_theta

R:
    max_j g_theta(u_ij)

S:
    positive-instance MIL assumption
```

Mathematically:

```math
z_i
=
\sup_{u\in\mathrm{supp}(\mu_i)}g_\theta(u).
```

Max pooling preserves strongest evidence and discards prevalence.

## ABMIL

```text
G:
    none in the attention readout

C:
    s_theta(h_ij), v_theta(h_ij)

R:
    sum_j softmax_j(s_theta(h_ij)) v_theta(h_ij)

S:
    bag-level class or survival loss
```

Mathematically:

```math
a_{ij}
=
\frac{\exp(s_\theta(h_{ij}))}
{\sum_\ell\exp(s_\theta(h_{i\ell}))},
\qquad
z_i
=
\sum_j a_{ij}v_\theta(h_{ij}).
```

ABMIL preserves a label-trained weighted first moment.

## DSMIL

```text
G:
    none unless encoded in h_ij

C:
    instance classifier selects a critical instance

R:
    max branch plus attention around the critical query

S:
    bag label and feature pretraining
```

Mathematically:

```math
j_i^\star(c)
=
\arg\max_j s_{ijc},
```

```math
a_{ij}^{(c)}
\propto
\exp(q_{i,j_i^\star(c)}^\top q_{ij}),
\qquad
z_i^{(c)}
=
\sum_j a_{ij}^{(c)}v_{ij}.
```

DSMIL preserves both an extreme instance score and a critical-instance-centered
weighted first moment.

## CLAM

```text
G:
    none unless coordinates or regions are added externally

C:
    class-specific attention and instance-level feature shaping

R:
    class-specific attention readout

S:
    slide label plus top-k and bottom-k pseudo-instance constraints
```

Mathematically:

```math
a_{ij}^{(c)}
=
\mathrm{softmax}_j s_c(h_{ij}),
\qquad
z_i^{(c)}
=
\sum_j a_{ij}^{(c)}h_{ij}.
```

The auxiliary loss shapes high-attention and low-attention patches, but the
inference statistic is still a class-conditioned weighted first moment.

## PANTHER

```text
G:
    prototype geometry, mixture covariance, or transport cost

C:
    assignment to morphology components

R:
    average responsibilities and residuals

S:
    unsupervised prototype learning plus downstream task supervision
```

Mathematically:

```math
\gamma_{ijm}
=
p(m\mid h_{ij}),
\qquad
\widehat\pi_{im}
=
\frac{1}{n_i}\sum_j\gamma_{ijm}.
```

PANTHER preserves morphology component usage and optional deviations from the
component centers.

## Dense Summary

```text
Pooling papers are different answers to R.
Their real differences become clear only after C, G, and S are named.
```

