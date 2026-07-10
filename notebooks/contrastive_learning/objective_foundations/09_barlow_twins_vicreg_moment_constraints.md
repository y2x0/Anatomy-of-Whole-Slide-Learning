# Barlow Twins And VICReg Moment Constraints

Barlow Twins and VICReg avoid explicit negative candidates by constraining
batch-level moments. Their anti-collapse mechanisms are visible directly in
their objectives.

Primary sources:

- Zbontar et al. "Barlow Twins: Self-Supervised Learning via Redundancy
  Reduction." ICML 2021. https://arxiv.org/abs/2103.03230
- Bardes, Ponce, and LeCun. "VICReg: Variance-Invariance-Covariance
  Regularization for Self-Supervised Learning." ICLR 2022.
  https://arxiv.org/abs/2105.04906

## Paired Batch Embeddings

For `B` source examples and two views, let:

```math
Z^{(1)},Z^{(2)}
\in
\mathbb{R}^{B\times d}.
```

Row `b` contains the embedding of source example `b`; column `r` contains one
embedding coordinate across the batch.

## Barlow Twins Cross-Correlation

After centering each coordinate over the batch, define:

```math
C_{rs}
=
\frac{
\sum_{b=1}^{B}
Z_{br}^{(1)}Z_{bs}^{(2)}
}{
\sqrt{
\sum_{b=1}^{B}
\left(Z_{br}^{(1)}\right)^2
}
\sqrt{
\sum_{b=1}^{B}
\left(Z_{bs}^{(2)}\right)^2
}
}.
```

Barlow Twins minimizes:

```math
\mathcal{L}_{\mathrm{BT}}
=
\sum_{r=1}^{d}
\left(
1-C_{rr}
\right)^2
+
\lambda
\sum_{r=1}^{d}
\sum_{s\ne r}
C_{rs}^2.
```

Diagonal term:

```math
C_{rr}
\to
1
```

aligns corresponding coordinates across views.

Off-diagonal term:

```math
C_{rs}
\to
0,
\qquad
r\ne s
```

reduces cross-coordinate redundancy.

## Identity As A Full-Rank Target

The target is:

```math
C
\to
I_d.
```

An identity matrix has rank `d`. A rank-one or constant embedding field cannot
match this target when coordinate variances and correlations are well-defined.

Barlow Twins prevents collapse through batch normalization and a full-rank
cross-correlation target, not through pairwise negative repulsion.

## Finite-Batch Rank Constraint

After centering over `B` examples, each view matrix has rank at most:

```math
B-1.
```

Therefore:

```math
\mathrm{rank}(C)
\le
\min(B-1,d).
```

If:

```math
d
>
B-1,
```

then:

```math
C
=
I_d
```

is impossible for an empirical batch because the target has rank `d`. The loss
can approach the identity coordinatewise but cannot attain zero exactly under
that rank constraint.

## VICReg Invariance Term

VICReg first aligns paired rows:

```math
s
\left(
Z^{(1)},Z^{(2)}
\right)
=
\frac{1}{B}
\sum_{b=1}^{B}
\left\lVert
z_b^{(1)}-z_b^{(2)}
\right\rVert_2^2.
```

This term alone is minimized by constant embeddings and therefore cannot
prevent collapse.

## VICReg Variance Term

For coordinate `r`, define regularized standard deviation:

```math
\sigma_r(Z)
=
\sqrt{
\mathrm{Var}
\left(
Z_{:r}
\right)
+
\varepsilon
}.
```

The variance hinge is:

```math
v(Z)
=
\frac{1}{d}
\sum_{r=1}^{d}
\max
\left(
0,
\gamma-\sigma_r(Z)
\right).
```

Coordinates below target spread `gamma` receive pressure to increase their
batch variance.

## VICReg Covariance Term

Let centered embeddings be:

```math
\widetilde Z
=
Z
-
\mathbf{1}_B\overline z^{\top},
```

where:

```math
\overline z
=
\frac{1}{B}
Z^{\top}\mathbf{1}_B.
```

The empirical covariance is:

```math
\Sigma(Z)
=
\frac{1}{B-1}
\widetilde Z^{\top}\widetilde Z.
```

VICReg penalizes off-diagonal covariance:

```math
c(Z)
=
\frac{1}{d}
\sum_{r\ne s}
\Sigma(Z)_{rs}^2.
```

This discourages redundant embedding coordinates. It does not by itself prevent
all embeddings from becoming constant because a collapsed covariance matrix is
already diagonal and zero.

## Full VICReg Objective

The objective is:

```math
\mathcal{L}_{\mathrm{VICReg}}
=
\lambda
s
\left(
Z^{(1)},Z^{(2)}
\right)
+
\mu
\left[
v(Z^{(1)})+v(Z^{(2)})
\right]
+
\nu
\left[
c(Z^{(1)})+c(Z^{(2)})
\right].
```

The three terms have separate jobs:

```text
invariance:
    align paired examples

variance:
    prevent dimensional collapse

covariance:
    reduce informational redundancy across coordinates
```

## Constant-Embedding Counterexample

Suppose:

```math
z_b^{(1)}
=
z_b^{(2)}
=
c
```

for every batch element. Then:

```math
s
\left(
Z^{(1)},Z^{(2)}
\right)
=
0,
```

```math
c(Z^{(1)})
=
c(Z^{(2)})
=
0.
```

But:

```math
\sigma_r(Z)
=
\sqrt{\varepsilon}.
```

If:

```math
\gamma
>
\sqrt{\varepsilon},
```

then:

```math
v(Z)
=
\gamma-\sqrt{\varepsilon}
>
0.
```

The variance term explicitly assigns positive loss to the constant solution.

## Batch Dependence

Neither method uses explicit negative pairs, but both couple different source
examples through batch statistics:

```math
C,
\qquad
\sigma_r(Z),
\qquad
\Sigma(Z).
```

Small, correlated, or site-homogeneous batches produce noisy or biased moment
estimates. Negative-free does not mean sample-independent.

The empirical covariance also satisfies:

```math
\mathrm{rank}
\left(
\Sigma(Z)
\right)
\le
\min(B-1,d).
```

When embedding dimension exceeds centered batch rank, covariance regularization
cannot observe all population directions independently in one batch.

## Barlow Twins Versus VICReg

Barlow Twins constrains a cross-view normalized correlation matrix:

```math
C
\approx
I_d.
```

VICReg separates three constraints:

```math
Z^{(1)}
\approx
Z^{(2)},
```

```math
\sigma_r(Z^{(b)})
\ge
\gamma,
```

```math
\Sigma(Z^{(b)})_{rs}
\approx
0
\quad
r\ne s.
```

VICReg makes collapse prevention explicit through a per-branch variance floor;
Barlow Twins encodes invariance and redundancy reduction jointly in its
cross-correlation target.

## C/R/G/S Placement

```text
\mathcal{G}:
    paired rows and batch-coordinate geometry

\mathcal{C}:
    twin encoders

\mathcal{R}:
    embedding matrices and their empirical moments

\mathcal{S}:
    identity cross-correlation target or invariance/variance/covariance targets
```

## Dense Summary

Explicit negatives are replaced by moment constraints:

```math
\text{pair alignment}
+
\text{nonzero spread}
+
\text{nonredundant coordinates}.
```

The learned similarity is therefore inseparable from how batch moments are
estimated.
