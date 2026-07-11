# SCL-WC Exact WSCL Objective And Gradients

Primary anchor:

- Wang et al. "SCL-WC: Cross-Slide Contrastive Learning for Weakly-Supervised
  Whole-Slide Image Classification." NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Paper-Conference.pdf

## Exact Paper Objective

For an anchor patch in an anchor bag, its positive bags, and the total
comparison collection, the paper defines:

```math
\mathcal{L}_{\mathrm{WSCL}}
=
\sum_{z_i\in\mathcal{B}}
-\frac{1}{k|\mathcal{P}|}
\sum_{p_r\in\mathcal{P}}
\sum_{z_j\in p_r}
\log
\frac{
\exp
\left(
z_i^{\top}z_j/\tau
\right)
}{
\sum_{a\in\mathcal{Q}}
\exp
\left(
z_i^{\top}z_a/\tau
\right)
}.
```

The denominator notation treats `a` as elements drawn from the total bag
collection; operationally the comparison units are stored patch features.

## Multi-Positive Softmax

For anchor `i`, define candidate probability:

```math
q_{ia}
=
\frac{
\exp
\left(
z_i^{\top}z_a/\tau
\right)
}{
\sum_{b\in\mathcal{Q}}
\exp
\left(
z_i^{\top}z_b/\tau
\right)
}.
```

Let positive count be:

```math
M_i^{+}
=
k|\mathcal{P}|.
```

Then anchor loss is an average of log probabilities for every positive patch.

## Exact Logit Gradient

For similarity:

```math
s_{ia}
=
z_i^{\top}z_a,
```

the derivative is:

```math
\frac{\partial\mathcal{L}_i}
{\partial s_{ia}}
=
\frac{1}{\tau}
\left[
q_{ia}
-
\frac{
\mathbb{1}
\left\{
a\in\mathcal{P}_i^{\mathrm{patch}}
\right\}
}{M_i^{+}}
\right].
```

Each positive target receives uniform mass `1/M_i+`, while predicted mass is
similarity-adaptive.

## Anchor Gradient

```math
\nabla_{z_i}
\mathcal{L}_i
=
\frac{1}{\tau}
\left[
\sum_{a\in\mathcal{Q}}
q_{ia}z_a
-
\frac{1}{M_i^{+}}
\sum_{p\in\mathcal{P}_i^{\mathrm{patch}}}
z_p
\right].
```

The anchor moves away from the softmax-weighted candidate mean and toward the
uniform positive mean.

## Positive Heterogeneity Penalty

If same-class positives form multiple distant modes, the target mean can lie
between real morphologies:

```math
\bar z_i^{+}
=
\frac{1}{M_i^{+}}
\sum_{p\in\mathcal{P}_i^{\mathrm{patch}}}
z_p.
```

The objective encourages class compactness even when disease-positive tissue
has legitimate subtype or grade structure.

## Full Objective

```math
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{CDA}}
+
\beta
\mathcal{L}_{\mathrm{PNM}}
+
\gamma
\mathcal{L}_{\mathrm{WSCL}}.
```

The contrastive support is produced by parameters also optimized through CDA
and PNM, making the objective support-dependent and nonstationary.
