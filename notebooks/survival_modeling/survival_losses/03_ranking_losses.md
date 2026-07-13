# Ranking Losses

Ranking losses optimize the order of risk scores rather than the probability of
event times.

A pair
```math
(i,j)
```
is comparable when:

```math
X_i<X_j,
\qquad
\delta_i=1.
```

Subject
```math
i
```
 had an observed event before subject
```math
j
```
left observation.

## Generic Pairwise Ranking

Let:

```math
r_i=\rho(\widehat{S}_i,\widehat{F}_i,\widehat{\lambda}_i)
```

be a scalar risk summary. A pairwise loss is:

```math
\mathcal{L}_{\text{rank}}
=
\sum_{i,j}
A_{ij}\phi(r_j-r_i),
```

where:

```math
A_{ij}=\mathbf{1}[X_i<X_j,\delta_i=1].
```

The function
```math
\phi
```
 is decreasing when
```math
r_i>r_j
```
.

Examples:

```math
\phi_{\text{log}}(u)=\log(1+\exp(u/\sigma)),
```

```math
\phi_{\text{hinge}}(u)=\max(0,1+u),
```

```math
\phi_{\text{exp}}(u)=\exp(u/\sigma).
```

## Gradient

For logistic pairwise loss:

```math
\phi(u)=\log(1+\exp(u/\sigma)).
```

Then:

```math
\frac{\partial\phi}{\partial r_i}
=
-
\frac{1}{\sigma}
\sigma_{\text{logistic}}((r_j-r_i)/\sigma),
```

```math
\frac{\partial\phi}{\partial r_j}
=
\frac{1}{\sigma}
\sigma_{\text{logistic}}((r_j-r_i)/\sigma).
```

The loss pushes the earlier event subject to higher risk.

## Risk Functional Problem

If the model outputs a survival curve, the ranking loss still needs a scalar:

```math
r_i=\rho[\widehat{S}_i].
```

Common choices:

```math
r_i(t)=1-\widehat{S}_i(t),
```

```math
r_i=-\widehat{\mathbb{E}}[T_i],
```

```math
r_i=\sum_k\widehat{h}_{ik},
```

```math
r_i=\widehat{F}_i(t^\star).
```

Different
```math
\rho
```
can produce different rankings. The ranking loss optimizes
the chosen functional, not the whole curve.

## Ranking For Competing Risks

For cause
```math
c
```
:

```math
A_{ij}^{(c)}
=
\mathbf{1}[X_i<X_j,\Delta_i=c].
```

Use cause-specific cumulative incidence:

```math
r_{ic}(t)=\widehat{F}_{ic}(t).
```

A DeepHit-style ranking term is:

```math
\mathcal{L}_{\text{rank}}
=
\sum_c\sum_{i,j}
A_{ij}^{(c)}
\phi(\widehat{F}_{jc}(X_i)-\widehat{F}_{ic}(X_i)).
```

This ranks cause-specific CIFs at the event time of the earlier subject.

## WSI Consequence

For WSI:

```math
r_i=\rho(\mathcal{H}(\mathcal{R}(\mathcal{C}(H_i)))).
```

The ranking loss sees only
```math
r_i
```
. If the aggregator discards morphology that
matters for calibration but not ranking, the loss will not recover it.

This is why ranking can produce high C-index but poor survival curves.

## Dense Summary

```math
\begin{aligned}
A_{ij}&=\mathbf{1}[X_i<X_j,\delta_i=1],\\
\mathcal{L}_{\text{rank}}
&=
\sum_{i,j}A_{ij}\phi(r_j-r_i),\\
r_i&=\rho[\text{predicted risk object}_i].
\end{aligned}
```

Ranking losses are useful, but they train a scalar ordering functional. They are
not probabilistic survival likelihoods.
