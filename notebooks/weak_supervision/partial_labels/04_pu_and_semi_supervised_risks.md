# PU And Semi-Supervised Risks

Partial labels often produce positive-unlabeled or semi-supervised instance
learning.

Let labeled positives be:

```math
\mathcal{P}
=
\{(i,j):Z_{ij}=1\text{ observed}\}.
```

Let unlabeled instances be:

```math
\mathcal{U}
=
\{(i,j):Z_{ij}\text{ unobserved}\}.
```

Unlabeled does not mean negative.

## PU Risk

Let:

```math
\pi
=
P(Z=1).
```

The true risk is:

```math
R(g)
=
\pi R_1^{+}(g)
+
(1-\pi)R_0^{-}(g),
```

where:

```math
R_1^{+}(g)
=
\mathbb{E}_{h\sim P_1}[\ell(g(h),1)],
```

and:

```math
R_0^{-}(g)
=
\mathbb{E}_{h\sim P_0}[\ell(g(h),0)].
```

The unlabeled distribution is:

```math
P_U
=
\pi P_1+(1-\pi)P_0.
```

Therefore:

```math
R_U^{-}(g)
=
\pi R_1^{-}(g)
+
(1-\pi)R_0^{-}(g).
```

Solving for the negative risk:

```math
(1-\pi)R_0^{-}(g)
=
R_U^{-}(g)-\pi R_1^{-}(g).
```

Thus:

```math
R_{\mathrm{PU}}(g)
=
\pi R_1^{+}(g)
+
R_U^{-}(g)
-
\pi R_1^{-}(g).
```

This is the basic PU correction.

## Consistency Regularization

For unlabeled instances, a semi-supervised objective can impose:

```math
g_\theta(t_1(h))
\approx
g_\theta(t_2(h)),
```

where $t_1,t_2$ are augmentations.

The loss is:

```math
\mathcal{L}_{\mathrm{cons}}
=
\sum_{(i,j)\in\mathcal{U}}
\left\|
p_\theta(t_1(h_{ij}))
-
p_\theta(t_2(h_{ij}))
\right\|_2^2.
```

This assumes augmentations preserve the latent label.

## Joint Objective

A partial-label objective can be:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{sup}}
+
\lambda_{\mathrm{PU}}\mathcal{L}_{\mathrm{PU}}
+
\lambda_{\mathrm{cons}}\mathcal{L}_{\mathrm{cons}}
+
\lambda_{\mathrm{bag}}\mathcal{L}_{\mathrm{bag}}.
```

Each term encodes a different supervision channel.

## C/R/G/S Placement

```text
G:
    augmentations and local neighborhoods define label-preserving transformations

C:
    representation is shaped by labeled positives and unlabeled consistency

R:
    bag constraints can coexist with instance PU risk

S:
    labeled positive set plus unlabeled pool
```

## Dense Summary

Partial labels often produce:

```math
\text{positive}
+
\text{unlabeled},
```

not:

```math
\text{positive}
+
\text{negative}.
```

The distinction changes the risk estimator.
