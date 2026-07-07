# Self-Training As Latent-Variable Optimization

Pseudo-labeling can be written as approximate latent-variable optimization.

Let $Z_i$ be hidden instance labels and $Y_i$ observed bag labels.

The ideal objective is:

```math
\max_\theta
\sum_i
\log
P_\theta(Y_i\mid H_i)
=
\max_\theta
\sum_i
\log
\sum_{z_i}
P_\theta(Y_i,z_i\mid H_i).
```

The sum over $z_i$ is often hard or poorly identified. Pseudo-labeling replaces
the posterior with a point estimate:

```math
\widehat z_i
=
\Psi_\theta(H_i,Y_i).
```

Then it optimizes:

```math
\max_\theta
\sum_i
\log
P_\theta(Y_i,\widehat z_i\mid H_i).
```

This is hard EM in spirit.

## EM View

The exact evidence lower bound is:

```math
\log P_\theta(Y_i\mid H_i)
\ge
\mathbb{E}_{q_i(z)}
\left[
\log P_\theta(Y_i,z\mid H_i)
-
\log q_i(z)
\right].
```

The E-step would choose:

```math
q_i(z)
=
P_{\theta^{old}}(z\mid Y_i,H_i).
```

Hard pseudo-labeling chooses:

```math
q_i(z)
=
\delta_{\widehat z_i}(z).
```

The M-step then treats $\widehat z_i$ as truth.

## Confidence Thresholding

A pseudo-label rule may use:

```math
\widehat z_{ij}
=
\mathbf{1}\{p_\theta(Z_{ij}=1\mid h_{ij})\ge\tau\}.
```

Only confident predictions are used:

```math
M_{ij}
=
\mathbf{1}
\left\{
\max_c p_\theta(Z_{ij}=c\mid h_{ij})
\ge
\tau
\right\}.
```

The pseudo-label loss is:

```math
\mathcal{L}_{\mathrm{pseudo}}
=
-
\sum_{i,j}
M_{ij}
\log
p_\theta(\widehat z_{ij}\mid h_{ij}).
```

## Coupled Target Problem

Pseudo-labels depend on parameters:

```math
\widehat z
=
\Psi_{\theta}(H).
```

The training target changes as the model changes:

```math
\mathcal{L}(\theta)
=
\ell(\theta,\Psi_{\theta}(H)).
```

Most implementations stop gradients through $\Psi_\theta$:

```math
\nabla_\theta
\ell(\theta,\mathrm{stopgrad}(\Psi_{\theta}(H))).
```

This is not optimizing an ordinary fixed supervised loss.

## C/R/G/S Placement

```text
G:
    pseudo-labels may use spatial neighborhoods, clusters, or graph context

C:
    pseudo-instance labels directly shape feature space

R:
    pseudo-label rules often select which instances influence readout

S:
    observed weak labels plus generated latent assignments
```

## Dense Summary

Pseudo-labeling replaces posterior uncertainty:

```math
P(Z\mid Y,H)
```

with a point estimate:

```math
\widehat Z.
```

It gains stronger gradients by pretending an inferred latent variable is
observed. The danger is that the inferred variable may be wrong.
