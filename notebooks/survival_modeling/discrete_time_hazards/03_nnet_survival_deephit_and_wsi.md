# Nnet-Survival, DeepHit, And WSI

Nnet-survival style models use a neural network to output interval hazards:

```math
g_i=f_\theta(z_i)\in\mathbb{R}^K,
\qquad
h_{ik}=\sigma(g_{ik}).
```

The loss is the discrete hazard likelihood.

This family predicts a survival curve:

```math
S_i(\tau_k)
=
\prod_{\ell=1}^{k}(1-h_{i\ell}).
```

## DeepHit-Style Event Distribution

DeepHit predicts a discrete probability mass function over event times, and for
competing risks over event-time/cause pairs.

For a single event type:

```math
p_{ik}
=
\Pr(T_i\in I_k\mid z_i),
\qquad
\sum_{k=1}^{K}p_{ik}\le 1.
```

Then:

```math
S_i(\tau_k)
=
1-\sum_{\ell=1}^{k}p_{i\ell}.
```

This is subtly different from predicting hazards. A hazard is conditional on
survival; a mass
```math
p_{ik}
```
is unconditional.

The two connect by:

```math
p_{ik}
=
h_{ik}\prod_{\ell=1}^{k-1}(1-h_{i\ell}).
```

## Ranking Terms

Many discrete survival models combine likelihood with a ranking term. A generic
ranking penalty compares patients whose event order is known:

```math
\mathcal{L}_{\mathrm{rank}}
=
\sum_{i,j}
\mathbf{1}[X_i<X_j,\delta_i=1]
\phi(r_i,r_j).
```

Here
```math
r_i
```
may be a scalar summary of the predicted event distribution, such
as cumulative incidence at a clinically chosen time.

This improves concordance-like behavior but can weaken calibration if it
dominates the likelihood.

## WSI Head

For whole-slide features:

```math
z_i=\mathcal{R}(\mathcal{C}(H_i)).
```

A simple discrete hazard head is:

```math
h_i
=
\sigma(Wz_i+b),
\qquad
W\in\mathbb{R}^{K\times d}.
```

Each row
```math
w_k^\top
```
asks a different time-specific question of the slide:

```math
g_{ik}=w_k^\top z_i+b_k.
```

If the slide representation is attention pooled:

```math
z_i=\sum_j a_{ij}h_{ij},
```

then:

```math
g_{ik}
=
\sum_j a_{ij}w_k^\top h_{ij}+b_k.
```

The same attention weights are reused across time unless the model makes
attention time-dependent.

## Time-Dependent Attention

A more expressive WSI design is:

```math
a_{ijk}
=
\mathrm{softmax}_{j}(q_k^\top \phi(h_{ij})),
\qquad
z_{ik}=\sum_j a_{ijk}h_{ij},
```

then:

```math
h_{ik}=\sigma(w_k^\top z_{ik}+b_k).
```

Now early and late hazards may attend to different morphology.

This is a useful design axis:

```text
shared slide representation for all times
versus
time-specific slide representation
```

## What Survives Aggregation

Mean MIL:

```text
time-specific heads see the same first moment
```

Attention MIL:

```text
time-specific heads see one weighted first moment unless attention is time-indexed
```

Transformer/graph MIL:

```text
time-specific heads see contextualized slide embedding
```

Prototype/distribution pooling:

```text
time-specific heads can use prevalence of morphological phenotypes
```

## Key Takeaway

Discrete hazard models are the first survival family where WSI models can
represent:

```text
early-risk morphology != late-risk morphology.
```

But this only happens if the slide representation or head has enough
time-specific capacity.
