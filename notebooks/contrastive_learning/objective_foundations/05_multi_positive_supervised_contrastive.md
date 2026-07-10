# Multi-Positive Supervised Contrastive Learning

Supervised contrastive learning replaces one augmentation positive with a set
of label-defined positives. Where the positive average is placed relative to
the logarithm changes the loss and its gradient.

Primary source:

- Khosla et al. "Supervised Contrastive Learning." NeurIPS 2020.
  https://arxiv.org/abs/2004.11362

## Multiview Batch

Let a source batch contain `B` labeled samples. Two augmentations produce an
index set:

```math
I
=
\{1,\ldots,2B\}.
```

For anchor `i`, define all non-anchor candidates:

```math
A(i)
=
I\setminus\{i\}.
```

Define label-positive indices:

```math
P(i)
=
\left\{
p\in A(i):
y_p=y_i
\right\}.
```

The augmented view from the same source sample belongs to `P(i)`, so the set is
nonempty under the two-view construction.

## Outside-Log SupCon

For normalized projection vectors `z_i`, define:

```math
Z_i
=
\sum_{a\in A(i)}
\exp
\left(
\frac{z_i^{\top}z_a}{\tau}
\right).
```

The outside-log loss used by SupCon is:

```math
\mathcal{L}_{\mathrm{out},i}
=
-
\frac{1}{|P(i)|}
\sum_{p\in P(i)}
\log
\frac{
\exp(z_i^{\top}z_p/\tau)
}{Z_i}.
```

Because the denominator is shared across positives:

```math
\mathcal{L}_{\mathrm{out},i}
=
\log Z_i
-
\frac{1}{|P(i)|\tau}
\sum_{p\in P(i)}
z_i^{\top}z_p.
```

Define the positive centroid:

```math
\overline z_{P(i)}
=
\frac{1}{|P(i)|}
\sum_{p\in P(i)}z_p.
```

Then:

```math
\mathcal{L}_{\mathrm{out},i}
=
\log Z_i
-
\frac{
z_i^{\top}\overline z_{P(i)}
}{\tau}.
```

The attractive target is the equal-weight positive centroid.

## Inside-Log Alternative

An alternative places the positive average inside the logarithm:

```math
\mathcal{L}_{\mathrm{in},i}
=
-
\log
\left[
\frac{1}{|P(i)|}
\sum_{p\in P(i)}
\frac{
\exp(z_i^{\top}z_p/\tau)
}{Z_i}
\right].
```

Equivalently:

```math
\mathcal{L}_{\mathrm{in},i}
=
\log Z_i
-
\log
\left[
\frac{1}{|P(i)|}
\sum_{p\in P(i)}
\exp(z_i^{\top}z_p/\tau)
\right].
```

## Jensen Relation

By convexity of log-sum-exp:

```math
\log
\left[
\frac{1}{|P(i)|}
\sum_{p\in P(i)}
\exp(z_i^{\top}z_p/\tau)
\right]
\ge
\frac{1}{|P(i)|\tau}
\sum_{p\in P(i)}
z_i^{\top}z_p.
```

Therefore:

```math
\mathcal{L}_{\mathrm{in},i}
\le
\mathcal{L}_{\mathrm{out},i}.
```

The inequality does not imply that the lower objective learns better
representations. The two functions weight positives differently.

## Outside-Log Gradient

Define the candidate softmax:

```math
\pi_{ia}
=
\frac{
\exp(z_i^{\top}z_a/\tau)
}{Z_i}.
```

Holding candidate vectors fixed:

```math
\nabla_{z_i}
\mathcal{L}_{\mathrm{out},i}
=
\frac{1}{\tau}
\left(
\sum_{a\in A(i)}
\pi_{ia}z_a
-
\overline z_{P(i)}
\right).
```

Each positive contributes equally to the target centroid before the
normalization Jacobian is applied.

## Inside-Log Gradient

Define a positive-only softmax:

```math
\rho_{ip}
=
\frac{
\exp(z_i^{\top}z_p/\tau)
}{
\sum_{r\in P(i)}
\exp(z_i^{\top}z_r/\tau)
}.
```

Then:

```math
\nabla_{z_i}
\mathcal{L}_{\mathrm{in},i}
=
\frac{1}{\tau}
\left(
\sum_{a\in A(i)}
\pi_{ia}z_a
-
\sum_{p\in P(i)}
\rho_{ip}z_p
\right).
```

The inside-log objective gives larger positive target weight to positives that
are already more similar. The outside-log objective does not make this
softmax-weighted substitution.

## Self-Supervised Limit

If there is exactly one positive:

```math
|P(i)|
=
1,
```

then:

```math
\mathcal{L}_{\mathrm{out},i}
=
\mathcal{L}_{\mathrm{in},i}
=
\mathcal{L}_{\mathrm{NT-Xent},i}.
```

The distinction appears only when multiple positives are available.

## Class Frequency And Candidate Geometry

The factor:

```math
\frac{1}{|P(i)|}
```

normalizes positive attraction per anchor, but class frequency still changes:

```text
how often a class supplies anchors
how many same-class samples enter denominators
which negatives compete with each class
the variance of the positive centroid
```

SupCon does not by itself make the minibatch class distribution irrelevant.

## Pathology Interpretation Boundary

Using slide labels to define `P(i)` imposes:

```math
y_p=y_i
\quad\Longrightarrow\quad
\text{representations should align}.
```

This can collapse morphology that differs within the same diagnosis, grade, or
survival stratum. The loss is correct for its declared equivalence relation;
the biological question is whether that equivalence is appropriate.

## C/R/G/S Placement

```text
\mathcal{G}:
    class-induced positive relation inside the multiview batch

\mathcal{C}:
    shared encoder

\mathcal{R}:
    normalized projected representation

\mathcal{S}:
    positive set P(i), candidate set A(i), and outside-log cross-entropy
```

## Dense Summary

SupCon replaces one target key with a positive centroid:

```math
\nabla_{z_i}\mathcal{L}_{\mathrm{out},i}
=
\frac{
\text{candidate Gibbs mean}
-
\text{equal-weight positive mean}
}{\tau}.
```

Moving the positive sum inside the logarithm changes that target into a
similarity-weighted positive mean.
