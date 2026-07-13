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

## Assumptions Behind The Correction

The algebra above is not enough. PU risk correction also needs a sampling
model for why positives are labeled.

Let
```math
L\in\{0,1\}
```
indicate whether an instance receives a positive annotation.
Under SCAR:

```math
P(L=1\mid Z=1,H,G)
=
c,
```

and:

```math
P(L=1\mid Z=0,H,G)
=
0.
```

Then labeled positives are representative of all positives:

```math
P(H\mid L=1,Z=1)
=
P(H\mid Z=1).
```

Under SAR, labeling can depend on observed covariates:

```math
P(L=1\mid Z=1,H,G)
=
c(H,G).
```

Under MNAR, labeling depends on unobserved morphology or difficulty:

```math
P(L=1\mid Z=1,H,G,U_{\mathrm{hidden}})
```

and the standard PU estimator is generally biased.

## Class Prior And Non-Negative Correction

The correction requires the class prior:

```math
\pi
=
P(Z=1).
```

If
```math
\pi
```
is wrong, the negative-risk estimate:

```math
R_U^{-}(g)-\pi R_1^{-}(g)
```

can become negative in finite samples. Non-negative PU methods replace this
term by a constrained version:

```math
\max
\left\{
0,
R_U^{-}(g)-\pi R_1^{-}(g)
\right\}.
```

This is not just numerical hygiene. It acknowledges that the unbiased estimator
can overfit by driving the estimated negative risk below zero.

## Consistency Regularization

For unlabeled instances, a semi-supervised objective can impose:

```math
g_\theta(t_1(h))
\approx
g_\theta(t_2(h)),
```

where
```math
t_1,t_2
```
are augmentations.

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
