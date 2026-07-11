# SCL-WC Exact WSCL Objective And Gradients

Primary anchor:

- Wang et al. "SCL-WC: Cross-Slide Contrastive Learning for Weakly-Supervised
  Whole-Slide Image Classification." NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Paper-Conference.pdf

## Exact Objective

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

The denominator notation treats the total bag collection as a collection of
stored patch features.

## Multi-Positive Probabilities

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

Let:

```math
M_i^{+}
=
k|\mathcal{P}|.
```

For similarity `s_ia`:

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

The anchor gradient is:

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

Each positive receives uniform target mass while predicted mass is
similarity-adaptive.

## Positive Heterogeneity

```math
\bar z_i^{+}
=
\frac{1}{M_i^{+}}
\sum_{p\in\mathcal{P}_i^{\mathrm{patch}}}
z_p.
```

If same-class positives form distant modes, this mean lies between legitimate
morphologies and the loss compresses subtype or grade variation.

## Full Loss

```math
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{CDA}}
+
\beta\mathcal{L}_{\mathrm{PNM}}
+
\gamma\mathcal{L}_{\mathrm{WSCL}}.
```

Its support is itself parameter-dependent and nonstationary.
