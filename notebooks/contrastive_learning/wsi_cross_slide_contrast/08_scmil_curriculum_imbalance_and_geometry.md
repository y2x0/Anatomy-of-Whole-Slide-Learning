# SC-MIL Curriculum, Imbalance, And Geometry

Primary anchor:

- Juyal et al. "SC-MIL: Supervised Contrastive Multiple Instance Learning for
  Imbalanced Classification in Digital Pathology." WACV 2024.
  https://openaccess.thecvf.com/content/WACV2024/html/Juyal_SC-MIL_Supervised_Contrastive_Multiple_Instance_Learning_for_Imbalanced_Classification_in_WACV_2024_paper.html

## Curriculum Objective

```math
\mathcal{L}_{\mathrm{SC\mbox{-}MIL}}^{(t)}
=
\beta_t\mathcal{L}_{\mathrm{SCL}}
+
\left(
1-\beta_t
\right)
\mathcal{L}_{\mathrm{CE}}.
```

The contrastive weight decays, moving from representation geometry toward
decision-boundary fitting.

```math
\nabla_{\theta}\mathcal{L}^{(t)}
=
\beta_t
\nabla_{\theta}\mathcal{L}_{\mathrm{SCL}}
+
\left(
1-\beta_t
\right)
\nabla_{\theta}\mathcal{L}_{\mathrm{CE}}.
```

This is not equivalent to freezing a contrastive encoder before classifier
training because both branches reshape shared features during the schedule.

## Minority Positive Count

```math
N_c
\sim
\mathrm{Binomial}
\left(
B,\pi_c
\right).
```

The probability that a class-c anchor has no same-class partner is:

```math
p_{0,c}
=
\left(
1-\pi_c
\right)^{B-1}.
```

It approaches one for rare classes and small WSI batches.

## Reweighted Sampling

Oversampling changes the contrastive risk from empirical class prior `pi` to a
proposal `tilde pi`:

```math
\mathbb{E}_{Y\sim\widetilde\pi}
\mathbb{E}
\left[
\mathcal{L}_{\mathrm{SCL}}
\mid
Y
\right].
```

This supplies minority positives but learns geometry under a reweighted law.

## Within-Class Compression

```math
\Sigma_c
=
\mathbb{E}
\left[
\left(
z-\mu_c
\right)
\left(
z-\mu_c
\right)^{\top}
\mid
Y=c
\right].
```

Same-class attraction can suppress grade, molecular subtype, or prognosis
directions not needed for current classification.

## Projection Escape

If the projection head is non-injective:

```math
b_a\ne b_b,
\qquad
g(b_a)=g(b_b),
```

the contrastive space can compress variation that remains in the transferred
bag embedding.
