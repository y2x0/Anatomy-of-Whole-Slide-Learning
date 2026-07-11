# SCL-WC Positive-Negative Subspaces

Primary anchor:

- Wang et al. "SCL-WC: Cross-Slide Contrastive Learning for Weakly-Supervised
  Whole-Slide Image Classification." NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Paper-Conference.pdf

## Class-Agnostic Gate

```math
\widetilde A_n^l
=
\sigma
\left(
\theta_4
\left[
\mathrm{ReLU}
\left(
\theta_3f_{n,l}^{\top}
\right)
\right]
\right).
```

## Complementary Readouts

```math
F_n^{\mathrm{pos}}
=
\frac{1}{L_n}
\sum_{l=1}^{L_n}
\widetilde A_n^l f_{n,l},
```

```math
F_n^{\mathrm{neg}}
=
\frac{1}{L_n}
\sum_{l=1}^{L_n}
\left(
1-\widetilde A_n^l
\right)
f_{n,l}.
```

Their sum is the unweighted first moment:

```math
F_n^{\mathrm{pos}}
+
F_n^{\mathrm{neg}}
=
\frac{1}{L_n}
\sum_{l=1}^{L_n}
f_{n,l}.
```

The module partitions mean feature mass rather than preserving interactions
between positive and negative regions.

## Gate Gradient

```math
\frac{\partial F_n^{\mathrm{pos}}}
{\partial\widetilde A_n^l}
=
\frac{f_{n,l}}{L_n},
\qquad
\frac{\partial F_n^{\mathrm{neg}}}
{\partial\widetilde A_n^l}
=
-\frac{f_{n,l}}{L_n}.
```

The readout is not normalized by gate mass:

```math
F_n^{\mathrm{pos}}
\ne
\frac{
\sum_l\widetilde A_n^l f_{n,l}
}{
\sum_l\widetilde A_n^l
}.
```

Its norm therefore mixes morphology with positive gate prevalence.

## Printed PNM Expression

```math
\mathcal{L}_{\mathrm{PNM}}^{\mathrm{printed}}
=
-\frac{1}{B}
\sum_{n=1}^{B}
\left(
\log p_n^{\mathrm{pos}}
-
\log p_n^{\mathrm{neg}}
\right).
```

The paper describes this as a sum of positive-aware and negative-aware
cross-entropies. The displayed probability sign is ambiguous unless the
negative term denotes the opposite target consistently. We preserve the
printed expression instead of inventing an unreported correction.

## Surviving Statistic

The PNM retains two complementary weighted first moments. Any patch multisets
with the same gated sums are indistinguishable to it.
