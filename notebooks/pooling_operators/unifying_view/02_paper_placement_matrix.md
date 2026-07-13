# Paper Placement Matrix

This note places pooling papers and pooling families in the same C/R/G/S
language used across the repository.

The point is not to make a catalog. The point is to expose which mathematical
statistic reaches the slide-level head.

## Matrix

| Family or paper anchor | G | \mathcal{C} | \mathcal{R} | S | Surviving Statistic |
|---|---|---|---|---|---|
| Deep Sets / learned moment pooling | ignored unless encoded upstream | instance feature map \phi | canonical sum, or normalized mean as a distinct variant | bag label | empirical sum/mean of \phi(h); cardinality is retained only by sum or an explicit count channel |
| Mean pooling | ignored by readout | identity or upstream context | normalized average | downstream slide loss | first moment |
| Additive evidence pooling | ignored unless burden is spatially stratified | patch evidence map e_\theta | unnormalized sum or exposure-normalized sum | slide label or survival loss | total evidence burden |
| Max pooling | ignored unless scores encode it | instance scoring map | maximum or smooth maximum | positive-instance bag label | extreme score |
| Top-k / quantile pooling | ignored by readout | instance scoring map | order statistic or trimmed extreme mean | bag label | upper-tail statistic |
| Noisy-or MIL | ignored unless probabilities are local-context dependent | instance probability map | probability that at least one instance is positive | bag label under OR assumption | complement product of negative probabilities |
| ABMIL | none in readout | attention score and value maps | softmax-weighted sum | slide label | learned weighted first moment |
| DSMIL | none unless features encode it | instance classifier plus critical query | max branch plus critical-instance attention | bag label and representation pretraining | extreme score plus critical-centered mean |
| CLAM | none unless supplied externally | class-specific scoring and pseudo-instance shaping | class-specific attention readout | bag label plus top/bottom-k constraints | class-conditioned weighted first moment |
| Set Transformer / PMA | optional positional or structural tokens | self-attention blocks, often with inducing points | attention from learned seed queries | task loss through seed outputs | learned query-dependent set statistics |
| Prototype / PANTHER | prototype geometry, mixture covariance, transport cost | assignment to morphology components | prevalence, residual, and mixture summaries | prototype learning plus task loss | quantized morphology distribution |
| Distribution pooling | metric or kernel geometry | optional embedding map before empirical measure | CDF, histogram, MMD, OT, quantile, or sketch | task loss and sometimes unsupervised fitting | distribution summary beyond one moment |
| Robust pooling | usually ignored by readout | scoring or feature map with outliers present | trimmed, winsorized, median, M-estimator, or clipped attention | robust task objective or ordinary task loss | bounded-influence statistic |
| Hierarchical pooling / HIPT-style pooling | region tree, scale index, tissue hierarchy | local context within regions or scales | nested region-to-slide readout | slide label, survival loss, or pretraining | multiscale composition of local statistics |

## Moment Pooling And Deep Sets

```text
G:
    ignored unless coordinates or graph context have already been injected

C:
    instance map phi_theta

R:
    sum_j phi_theta(h_ij) or n_i^{-1} sum_j phi_theta(h_ij)

S:
    slide-level loss through rho_theta(R)
```

Mathematically:

```math
z_i
=
\frac{1}{n_i}\sum_{j=1}^{n_i}\phi_\theta(h_{ij}),
\qquad
\widehat y_i
=
\rho_\theta(z_i).
```

This is the Deep Sets shape: the readout is a moment of a learned feature map.
If
```math
\phi_\theta
```
is rich enough, the statistic can encode many distributional
features; if
```math
\phi_\theta
```
is weak, the slide collapses to a small number of
low-order moments.

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
multimodality, and layout unless those features have already been encoded into
the instance states
```math
u_{ij}
```
.

## Additive Evidence Pooling

```text
G:
    ignored by a flat sum, optionally used by regional sums

C:
    patch evidence map e_theta

R:
    total evidence or exposure-normalized evidence

S:
    slide label, count-like label, or survival objective
```

Mathematically:

```math
E_i
=
\sum_{j=1}^{n_i} e_\theta(h_{ij}),
\qquad
\widehat y_i
=
\sigma(b+E_i).
```

Additive pooling assumes evidence accumulates. This is a different inductive
bias from mean pooling: duplicating a positive region changes the logit under a
sum, but not under a normalized mean.

## Max And Upper-Tail Pooling

```text
G:
    ignored by R

C:
    instance scoring map g_theta

R:
    max, top-k mean, quantile, generalized mean, or log-sum-exp

S:
    usually the positive-instance MIL assumption
```

Mathematically:

```math
z_i^{\max}
=
\max_j g_\theta(h_{ij}),
```

```math
z_i^{\mathrm{top}k}
=
\frac{1}{k}\sum_{r=1}^{k}s_{i(r)},
\qquad
s_{i(1)}\ge s_{i(2)}\ge\cdots\ge s_{i(n_i)}.
```

Max preserves the single strongest score. Top-k and quantile pooling preserve
upper-tail behavior, which is often a better match when a phenotype occupies a
small but nonzero tissue fraction.

## Noisy-Or MIL

```text
G:
    ignored unless p_ij comes from contextual states

C:
    p_ij = sigmoid(g_theta(h_ij))

R:
    P(Y_i = 1) = 1 - prod_j (1 - p_ij)

S:
    bag label under the at-least-one-positive assumption
```

Mathematically:

```math
P(Y_i=0\mid H_i)
=
\prod_{j=1}^{n_i}(1-p_{ij}),
\qquad
P(Y_i=1\mid H_i)
=
1-\prod_{j=1}^{n_i}(1-p_{ij}).
```

Noisy-or is not just a smooth max. It is an independence model for latent patch
events. Its statistic is the product of patch-level non-event probabilities,
which makes calibration and bag size central.

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
{\sum_{\ell=1}^{n_i}\exp(s_\theta(h_{i\ell}))},
\qquad
z_i
=
\sum_{j=1}^{n_i}a_{ij}v_\theta(h_{ij}).
```

ABMIL preserves a label-trained weighted first moment. The attention weights are
not automatically instance probabilities; they are coefficients in a slide
statistic.

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
\sum_{j=1}^{n_i}a_{ij}^{(c)}v_{ij}.
```

DSMIL preserves both an extreme instance score and a weighted first moment
centered around the critical instance.

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
\sum_{j=1}^{n_i}a_{ij}^{(c)}h_{ij}.
```

The auxiliary loss changes the geometry of patch embeddings, but the inference
statistic is still a class-conditioned weighted first moment.

## Set Transformer Pooling

```text
G:
    optional positional, scale, or structural tokens

C:
    permutation-equivariant self-attention blocks

R:
    pooling by multihead attention from learned seed queries

S:
    downstream task loss through the seed outputs
```

Mathematically:

```math
U_i
=
\mathcal{C}_{\theta}(H_i),
\qquad
Z_i
=
\mathrm{MHA}(Q=S, K=U_i, V=U_i),
```

where
```math
S\in\mathbb{R}^{k\times d}
```
is a learned seed matrix. Set Transformer
pooling preserves
```math
k
```
learned summaries, not one scalar attention distribution.
When
```math
k>1
```
, different seed queries can specialize to different slide-level
statistics.

## Prototype And PANTHER-Style Pooling

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
\frac{1}{n_i}\sum_{j=1}^{n_i}\gamma_{ijm}.
```

PANTHER-like pooling preserves morphology component usage and optional
component-conditioned deviations. The slide is represented as a structured
summary of a patch distribution rather than as one pooled embedding.

## Distribution Pooling

```text
G:
    metric, kernel, transport geometry, or learned embedding geometry

C:
    optional map psi_theta before the empirical measure is summarized

R:
    histogram, CDF, quantile, MMD, OT, or sketch

S:
    downstream task loss, sometimes with unsupervised distribution fitting
```

Mathematically:

```math
\widehat\mu_i
=
\frac{1}{n_i}\sum_{j=1}^{n_i}\delta_{\psi_\theta(h_{ij})}.
```

Distribution pooling applies a finite statistic
```math
T
```
 to
```math
\widehat\mu_i
```
:

```math
z_i
=
T(\widehat\mu_i).
```

It differs from ordinary set pooling only when
```math
T
```
is chosen to preserve
distributional information beyond a single learned first moment, such as
quantiles, tail mass, kernel comparisons, or transport geometry.

## Robust Pooling

```text
G:
    usually ignored by R

C:
    feature or score map possibly contaminated by artifacts

R:
    trimmed mean, winsorized mean, median, geometric median, M-estimator, or clipped attention

S:
    ordinary task loss or explicit robust objective
```

Mathematically, an M-estimator readout can be written as:

```math
z_i
\in
\arg\min_z
\sum_{j=1}^{n_i}\rho(\|u_{ij}-z\|).
```

Mean pooling uses
```math
\rho(t)=t^2
```
. Robust pooling uses a loss whose influence does
not grow indefinitely. The surviving statistic is deliberately less sensitive
to outliers, which is useful for artifacts but dangerous when the rare outlier
is the diagnostic signal.

## Hierarchical Pooling

```text
G:
    region tree, tissue hierarchy, scale index, or block partition

C:
    local context inside regions and sometimes cross-region context

R:
    patch-to-region pooling followed by region-to-slide pooling

S:
    slide label, survival target, region pseudo-label, or pretraining task
```

Mathematically:

```math
r_{im}
=
\mathcal{R}_{\mathrm{local}}(\{h_{ij}:j\in B_{im}\}),
\qquad
z_i
=
\mathcal{R}_{\mathrm{global}}(\{r_{im}\}_{m=1}^{M_i}).
```

The surviving statistic is a composition of local statistics. This reduces
quadratic cost and exposes multiscale structure, but premature local pooling
can erase patch-level evidence before global reasoning sees it.

## Dense Summary

```text
Pooling papers are different answers to R.
Their real differences become clear only after C, G, and S are named.
The main question is always: which statistic survives, and what equivalence
class of slides does that statistic create?
```
