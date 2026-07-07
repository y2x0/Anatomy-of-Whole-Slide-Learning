# Assumption Theorem Template

Every weak-supervision method should be described with the same theorem-like
block.

The point is not to pretend every method has a formal proof. The point is to
force the assumptions, target, loss, and non-identifiable variables into the
open.

## Observed Data

```math
\mathcal{D}
=
\{(H_i,G_i,S_i^{\mathrm{obs}})\}_{i=1}^{N}.
```

## Latent Target

```math
U_i
=
(Y_i,Z_i,B_i,A_i,R_i).
```

State which target is needed:

```math
T(U_i)
\in
\{Y_i,Z_i,B_i,A_i,R_i\}.
```

## Observation Channel

```math
S_i^{\mathrm{obs}}
\sim
Q_\alpha(S\mid U_i,H_i,G_i).
```

If the method creates pseudo-labels, write:

```math
\widehat U_{i,t}
=
\Psi_t(H_i,G_i,S_i^{\mathrm{obs}},\theta_t,\mathcal{D})
```

instead of treating $\widehat U_{i,t}$ as observed data.

## Surrogate Loss

```math
\mathcal{L}_{\mathrm{sur}}(\theta)
=
\mathbb{E}
\left[
\ell(F_\theta(H,G),S^{\mathrm{obs}})
\right].
```

If generated targets are used:

```math
\mathcal{L}_{\mathrm{gen}}(\theta)
=
\mathbb{E}
\left[
\ell_{\mathrm{gen}}(F_\theta(H,G),\widehat U_t)
\right].
```

## Assumptions

State the assumptions explicitly:

```text
bag map:
    Y = Gamma(Z)

noise:
    Q_alpha is known, estimated, or ignored

missingness:
    M is MCAR, MAR, SAR, SCAR, or MNAR

pseudo-label:
    error rate of Psi is small enough

contrastive:
    positive pairs approximate latent equivalence
```

## Identifiable Quantity

State what the observed signal identifies:

```math
P(Y\mid H,G),
\qquad
P(\Gamma(Z)\mid H,G),
\qquad
P(S^{\mathrm{obs}}\mid H,G).
```

## Non-Identifiable Quantity

State what remains hidden:

```math
P(Z_j\mid H,G),
\qquad
A_j,
\qquad
\text{true witness index}.
```

## Counterexample

Give two latent worlds:

```math
U
\ne
U'
```

that imply the same observed supervision:

```math
Q(S^{\mathrm{obs}}\mid U,H,G)
=
Q(S^{\mathrm{obs}}\mid U',H,G).
```

If the method cannot distinguish them, say so.

## Dense Summary

The reporting unit is:

```math
(U,\ Q,\ S^{\mathrm{obs}},\ \Psi,\ \mathcal{L},\ \text{identifiable target}).
```

Without this block, weak-supervision claims become architectural storytelling.
