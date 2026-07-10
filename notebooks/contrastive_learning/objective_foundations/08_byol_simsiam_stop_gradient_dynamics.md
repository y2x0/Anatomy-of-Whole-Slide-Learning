# BYOL And SimSiam Stop-Gradient Dynamics

BYOL and SimSiam remove explicit negative candidates. Both use asymmetric
prediction and stop-gradient targets, but their target-network dynamics are
different.

Primary sources:

- Grill et al. "Bootstrap Your Own Latent." NeurIPS 2020.
  https://arxiv.org/abs/2006.07733
- Chen and He. "Exploring Simple Siamese Representation Learning." CVPR 2021.
  https://arxiv.org/abs/2011.10566

## Normalized Regression Loss

For prediction `p` and target `z`, define normalized vectors:

```math
\overline p
=
\frac{p}{\lVert p\rVert_2},
\qquad
\overline z
=
\frac{z}{\lVert z\rVert_2}.
```

Squared normalized error is:

```math
\lVert
\overline p-\overline z
\rVert_2^2
=
2
-
2\overline p^{\top}\overline z.
```

Thus normalized MSE and negative cosine similarity differ only by affine
scaling.

## Stop-Gradient Operator

Define the automatic-differentiation primitive:

```math
\mathrm{sg}(z)
=
z
```

in the forward pass, but:

```math
\frac{\partial\mathrm{sg}(z)}
{\partial z}
=
0
```

in the backward pass.

This is a computational-graph rule, not an ordinary differentiable function:
the classical identity map cannot simultaneously have zero Jacobian. Any
mathematical description of BYOL or SimSiam must therefore specify both forward
values and blocked gradient paths.

## BYOL Online And Target Branches

For view `v_1`, the online branch computes:

```math
y_{\theta}^{(1)}
=
f_{\theta}(v_1),
```

```math
z_{\theta}^{(1)}
=
g_{\theta}
\left(
y_{\theta}^{(1)}
\right),
```

```math
p_{\theta}^{(1)}
=
q_{\theta}
\left(
z_{\theta}^{(1)}
\right).
```

For view `v_2`, the target branch computes:

```math
z_{\xi}^{(2)}
=
g_{\xi}
\left(
f_{\xi}(v_2)
\right).
```

The directional BYOL loss is:

```math
\mathcal{L}_{1\to2}^{\mathrm{BYOL}}
=
\left\lVert
\overline p_{\theta}^{(1)}
-
\mathrm{sg}
\left(
\overline z_{\xi}^{(2)}
\right)
\right\rVert_2^2.
```

The total objective is symmetrized:

```math
\mathcal{L}^{\mathrm{BYOL}}
=
\mathcal{L}_{1\to2}^{\mathrm{BYOL}}
+
\mathcal{L}_{2\to1}^{\mathrm{BYOL}}.
```

## BYOL Parameter Dynamics

Online parameters follow the loss gradient:

```math
\theta_{t+1}
=
\mathrm{Optimizer}
\left(
\theta_t,
\nabla_{\theta}
\mathcal{L}^{\mathrm{BYOL}}
\right).
```

Target parameters follow an exponential moving average:

```math
\xi_{t+1}
=
\beta_t\xi_t
+
(1-\beta_t)\theta_{t+1}.
```

They do not follow:

```math
-\nabla_{\xi}
\mathcal{L}^{\mathrm{BYOL}}.
```

BYOL is therefore a coupled optimization-and-tracking system, not joint
gradient descent on both branches.

## SimSiam Shared Encoder

SimSiam uses one shared encoder-projector `f_theta` and a predictor `h_phi`:

```math
z_1
=
f_{\theta}(v_1),
\qquad
z_2
=
f_{\theta}(v_2),
```

```math
p_1
=
h_{\phi}(z_1),
\qquad
p_2
=
h_{\phi}(z_2).
```

Define negative cosine similarity:

```math
D(p,z)
=
-
\frac{p^{\top}z}
{\lVert p\rVert_2\lVert z\rVert_2}.
```

The SimSiam loss is:

```math
\mathcal{L}^{\mathrm{SimSiam}}
=
\frac{1}{2}
D
\left(
p_1,
\mathrm{sg}(z_2)
\right)
+
\frac{1}{2}
D
\left(
p_2,
\mathrm{sg}(z_1)
\right).
```

Each view is a target in one term and an online prediction path in the other.

## Stop-Gradient Changes The Vector Field

For one directional term:

```math
\nabla_{z_2}
D
\left(
p_1,
\mathrm{sg}(z_2)
\right)
=
0.
```

But:

```math
\nabla_{p_1}
D
\left(
p_1,
\mathrm{sg}(z_2)
\right)
\ne
0.
```

The numerical forward loss is symmetric after adding both directions, but the
gradient path inside each direction is asymmetric.

## Predictor Is Not A Cosmetic Head

If the predictor is removed so that:

```math
p_1=z_1,
\qquad
p_2=z_2,
```

the two stopped directional gradients combine like a scaled gradient of the
ordinary symmetric similarity objective. SimSiam reports collapse in this
configuration.

The predictor lets the online branch adapt to a target representation without
requiring the target branch to move along the same instantaneous gradient.

## Collapse Remains A Feasible Forward Solution

Suppose every input maps to one normalized vector `c` and predictors output the
same vector:

```math
\overline p(x)
=
\overline z(x)
=
c.
```

Then:

```math
\mathcal{L}^{\mathrm{BYOL}}
=
0,
```

and SimSiam reaches its minimum negative-cosine value.

Therefore the forward objective alone does not algebraically exclude constant
representations. Collapse avoidance is a property of the asymmetric training
dynamics and architecture. BYOL explicitly notes that undesirable equilibria
remain possible; SimSiam presents stop-gradient as empirically essential.

## No Explicit Negative Does Not Mean No Coupling

BYOL couples examples through shared network parameters, normalization layers,
and a moving target encoder. SimSiam couples them through shared parameters and
batch-normalized networks.

The objectives do not contain a candidate denominator, but training examples
are not independent scalar regressions in parameter space.

## C/R/G/S Placement

```text
\mathcal{G}:
    paired-view relation and online-target direction

\mathcal{C}:
    online encoder plus EMA target encoder, or shared SimSiam encoder

\mathcal{R}:
    projector, predictor, and L2 normalization

\mathcal{S}:
    stopped target embedding rather than a negative candidate index
```

## Dense Summary

The anti-collapse mechanism is dynamical:

```text
BYOL:
    predictor + stopped EMA target

SimSiam:
    predictor + stopped opposite view
```

Neither method proves that constant representations are impossible from the
forward loss alone.
