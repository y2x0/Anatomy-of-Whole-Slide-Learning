# WSI Continuous Risk And Failure Modes

For whole-slide survival, a continuous-time model can be written:

```math
z_i=\mathcal R(\mathcal C(H_i;G_i)),
\qquad
\lambda_i(t)=\lambda_\theta(t,z_i).
```

The key difference from Cox is that morphology can interact with time:

```math
\lambda_\theta(t,z_i)
\ne
\lambda_0(t)\exp(f_\theta(z_i)).
```

## Time-Conditioned Slide Readout

The simplest version uses one slide embedding for all times:

```math
z_i=\mathcal R(H_i),
\qquad
\lambda_i(t)=\rho(g_\theta(t,z_i)).
```

A more expressive version makes the readout itself time-conditioned:

```math
a_{ij}(t)
=
\operatorname{softmax}_j(q(t)^\top \phi(h_{ij})),
```

```math
z_i(t)
=
\sum_j a_{ij}(t)h_{ij},
```

```math
\lambda_i(t)
=
\rho(g_\theta(t,z_i(t))).
```

Now the model can attend to different morphology at different time horizons.

## Prototype Continuous Survival

If a slide is represented by prototype proportions:

```math
p_i=(p_{i1},\ldots,p_{iM}),
```

then:

```math
\lambda_i(t)
=
\rho\left(
b(t)+\sum_{m=1}^{M}p_{im}r_m(t)
\right).
```

Each prototype has a time-varying risk contribution \(r_m(t)\).

This is a clean mathematical bridge between prototype MIL and continuous
survival.

## Graph Continuous Survival

For graph WSI models:

```math
G_i=(V_i,E_i),
\qquad
H_i^{(L)}=\operatorname{GNN}(G_i,H_i^{(0)}).
```

Then:

```math
z_i(t)
=
\operatorname{READOUT}_t(H_i^{(L)}),
\qquad
\lambda_i(t)=\rho(g_\theta(t,z_i(t))).
```

The model can learn time-specific graph summaries, for example early risk from
tumor regions and late risk from tumor-stroma-immune interactions.

## Failure Mode 1: Integration Error

The loss includes:

```math
\int_0^{X_i}\lambda_\theta(u,z_i)\,du.
```

If this integral is approximated poorly, training optimizes the wrong objective.

This matters when hazards are sharp, multimodal, or highly time-varying.

## Failure Mode 2: Unstable Hazards

If:

```math
\lambda_\theta(t,z)=\exp(g_\theta(t,z)),
```

large values of \(g_\theta\) create exploding hazards and gradients. Softplus
usually behaves better, but can still produce poorly calibrated tails.

## Failure Mode 3: Tail Extrapolation

Censoring limits observed follow-up. The model may output a smooth curve beyond
the observed support, but that does not mean the tail is learned.

For pathology cohorts with short follow-up:

```text
late survival is often extrapolation.
```

## Failure Mode 4: Shared Embedding Bottleneck

Even a continuous-time head cannot recover information discarded by the slide
readout:

```math
H_i \to z_i \to \lambda_i(t).
```

If \(z_i\) collapses rare morphology, the hazard function cannot use it.

## Failure Mode 5: Overfitting Time

Continuous-time models can fit flexible time functions with limited events.
This creates smooth-looking but unsupported risk curves.

The danger is aesthetic calibration: curves look plausible but are not
statistically grounded.

## Diagnostic Questions

1. Is the hazard integral computed accurately?
2. Does the model learn non-proportional effects, or only imitate Cox?
3. Are predictions evaluated at observed time horizons?
4. Does the slide representation preserve time-specific morphology?
5. Are survival curves calibrated, not just ranked?

## Design Escape Routes

```text
time-conditioned attention
prototype-specific hazard functions
mixture survival models
regularized/spline hazards
monotonic cumulative hazard parameterization
calibration checks at fixed horizons
```

## Key Takeaway

Continuous-time WSI survival is the most expressive of the first three families:

```text
slide morphology can change its meaning as a function of time.
```

But the price is numerical, statistical, and interpretive complexity.
