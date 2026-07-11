# SCL-WC Positive-Negative Subspaces

Primary anchor:

- Wang et al. "SCL-WC: Cross-Slide Contrastive Learning for Weakly-Supervised
  Whole-Slide Image Classification." NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Paper-Conference.pdf

## Class-Agnostic Gate

SCL-WC computes:

```math
\widetilde A_{n}^{l}
=
\sigma
\left(
\theta_4
\left[
\mathrm{ReLU}
\left(
\theta_3 f_{n,l}^{\top}
\right)
\right]
\right),
```

with:

```math
0
\le
\widetilde A_n^l
\le
1.
```

## Complementary Readouts

Positive-aware feature:

```math
F_n^{\mathrm{pos}}
=
\frac{1}{L_n}
\sum_{l=1}^{L_n}
\widetilde A_n^l f_{n,l}.
```

Negative-aware feature:

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

Their sum is exactly the unweighted first moment:

```math
F_n^{\mathrm{pos}}
+
F_n^{\mathrm{neg}}
=
\frac{1}{L_n}
\sum_{l=1}^{L_n}
f_{n,l}.
```

The module partitions mean feature mass; it does not preserve interactions
between positive and negative regions.

## Gate Gradient

```math
\frac{\partial F_n^{\mathrm{pos}}}
{\partial\widetilde A_n^l}
=
\frac{f_{n,l}}{L_n},
```

```math
\frac{\partial F_n^{\mathrm{neg}}}
{\partial\widetilde A_n^l}
=
-\frac{f_{n,l}}{L_n}.
```

Increasing a patch's positive mass removes the same vector mass from the
negative readout.

## Scale Is Not Normalized Attention

Because the denominator is `L_n`, not the sum of gates:

```math
F_n^{\mathrm{pos}}
\ne
\frac{
\sum_l
\widetilde A_n^l f_{n,l}
}{
\sum_l
\widetilde A_n^l
}.
```

Its norm encodes both selected morphology and total gate mass. A slide with
few positive patches can have smaller positive-feature magnitude even when
their conditional mean is identical.

## Printed PNM Loss Boundary

The paper displays:

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

It describes this as a sum of positive-aware and negative-aware
cross-entropies. The notation is ambiguous unless `p_neg` denotes probability
of the opposite target in a way consistent with that sign. This note preserves
the printed expression rather than inventing an unreported correction.

## Surviving Statistic

The PNM retains two complementary weighted first moments:

```math
\left(
F_n^{\mathrm{pos}},
F_n^{\mathrm{neg}}
\right).
```

Any two patch multisets with the same gated sums are indistinguishable to this
module.
