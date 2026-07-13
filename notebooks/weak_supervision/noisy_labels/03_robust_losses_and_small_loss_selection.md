# Robust Losses And Small-Loss Selection

When the noise channel is unknown, robust learning changes the training
dynamics rather than explicitly inverting
```math
T
```
.

## Memorization Assumption

Deep networks often fit clean labels earlier than random noise. This motivates
small-loss selection:

```math
\mathcal{I}_t
=
\left\{
i:
\ell_i(\theta_t)
\le
\mathrm{Quantile}_{q_t}
(\{\ell_m(\theta_t)\}_{m=1}^{N})
\right\}.
```

Training uses:

```math
\mathcal{L}_t
=
\frac{1}{|\mathcal{I}_t|}
\sum_{i\in\mathcal{I}_t}
\ell_i(\theta_t).
```

The assumption is:

```math
P(Y_i=\widetilde Y_i\mid \ell_i\ \text{small})
>
P(Y_i=\widetilde Y_i).
```

## Co-Teaching View

Two models select small-loss examples for each other:

```math
\mathcal{I}_t^{(1)}
=
\mathrm{SmallLoss}(\theta_t^{(1)}),
\qquad
\mathcal{I}_t^{(2)}
=
\mathrm{SmallLoss}(\theta_t^{(2)}).
```

Updates are crossed:

```math
\theta^{(1)}
\leftarrow
\theta^{(1)}
-
\eta
\nabla_{\theta^{(1)}}
\sum_{i\in\mathcal{I}_t^{(2)}}\ell_i(\theta^{(1)}),
```

```math
\theta^{(2)}
\leftarrow
\theta^{(2)}
-
\eta
\nabla_{\theta^{(2)}}
\sum_{i\in\mathcal{I}_t^{(1)}}\ell_i(\theta^{(2)}).
```

The hope is that the two models make different errors early enough to filter
noise.

## Generalized Cross Entropy

For predicted probability
```math
p_y
```
, generalized cross entropy is:

```math
\ell_q(p_y)
=
\frac{1-p_y^q}{q},
\qquad
0<q\le 1.
```

As
```math
q\to 0
```
:

```math
\ell_q(p_y)
\to
-\log p_y.
```

For larger
```math
q
```
, the loss is less sensitive to very small
```math
p_y
```
, reducing the
effect of hard mislabeled samples.

## Symmetric Cross Entropy

One can combine ordinary CE with a reverse term:

```math
\mathcal{L}_{\mathrm{SCE}}
=
\alpha\mathcal{L}_{\mathrm{CE}}
+
\beta\mathcal{L}_{\mathrm{RCE}}.
```

The goal is to reduce overconfidence on noisy labels.

## WSI-Specific Issue

In WSI, a high-loss slide may be:

```text
mislabeled
hard but correctly labeled
rare morphology
poorly sampled
out-of-distribution
```

Small-loss filtering can remove rare but important pathology.

## C/R/G/S Placement

```text
G:
    hard examples may cluster by site or tissue geometry

C:
    representation learning is shaped by selected examples

R:
    high-loss bags may be excluded before the readout learns rare evidence

S:
    noisy labels are handled by robust sample weighting
```

## Dense Summary

Robust losses replace:

```math
\text{trust every observed label equally}
```

with:

```math
\text{downweight or filter labels likely to be corrupted}.
```

The risk is filtering out true hard cases.
