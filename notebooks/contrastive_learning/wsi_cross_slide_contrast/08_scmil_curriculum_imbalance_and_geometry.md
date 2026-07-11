# SC-MIL Curriculum, Imbalance, And Geometry

Primary anchor:

- Juyal et al. "SC-MIL: Supervised Contrastive Multiple Instance Learning for
  Imbalanced Classification in Digital Pathology." WACV 2024.
  https://openaccess.thecvf.com/content/WACV2024/html/Juyal_SC-MIL_Supervised_Contrastive_Multiple_Instance_Learning_for_Imbalanced_Classification_in_WACV_2024_paper.html

## Curriculum Objective

SC-MIL combines bag contrast with classification:

```math
\mathcal{L}_{\mathrm{SC\mbox{-}MIL}}^{(t)}
=
\beta_t
\mathcal{L}_{\mathrm{SCL}}
+
\left(
1-\beta_t
\right)
\mathcal{L}_{\mathrm{CE}}.
```

The schedule decays the contrastive weight, transitioning from representation geometry to
decision-boundary fitting.

## Gradient Mixture

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

This is not equivalent to sequentially freezing a contrastive encoder and then
training a classifier. Both branches jointly reshape the shared bag encoder
while their weights change.

## Minority Positive Count

For a class with fixed prevalence, random batches give:

```math
N_c
\sim
\mathrm{Binomial}
\left(
B,\pi_c
\right).
```

Probability that a class-`c` anchor has no same-class partner is:

```math
p_{0,c}
=
\left(
1-\pi_c
\right)^{B-1}.
```

This approaches one for rare classes and small bag batches.

## Class-Balanced Sampling Changes The Proposal

Oversampling can replace the empirical class prior with a reweighted proposal.
The contrastive risk becomes:

```math
\mathbb{E}_{Y\sim\widetilde\pi}
\mathbb{E}
\left[
\mathcal{L}_{\mathrm{SCL}}
\mid
Y
\right].
```

This improves minority positive availability but learns geometry under a
reweighted class law.

## Within-Class Compression

For class `c`, define covariance of normalized bag representations:

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

Same-class attraction tends to reduce within-class covariance directions that do not
help separate labels. Those directions may contain grade, molecular subtype,
or prognosis information required by later tasks.

## Projection-Head Escape

Because contrast acts on:

```math
z_i
=
g(b_i),
```

a non-injective `g` can compress same-class variation while `b_i` retains some
of it. Downstream use of `b_i` and `z_i` therefore has different information
content.
