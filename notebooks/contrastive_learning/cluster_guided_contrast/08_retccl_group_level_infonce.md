# RetCCL Group-Level InfoNCE

RetCCL adds an auxiliary projection space in which each instance predicts the
nearest cluster centroid computed from the opposite augmentation branch. This
is a centroid candidate-classification problem, not the weighted instance loss.

Primary anchor:

- RetCCL: https://pubmed.ncbi.nlm.nih.gov/36270093/

## Auxiliary Projection Space

The two query-side paths produce:

```math
g_{p1}
=
g_1
\left(
h(x_p)
\right),
\qquad
g_{q1}
=
g_1
\left(
h(x_q)
\right).
```

For a minibatch of `B` source patches, write:

```math
G_p
=
\left\{
g_{p1}^{(b)}
\right\}_{b=1}^{B},
\qquad
G_q
=
\left\{
g_{q1}^{(b)}
\right\}_{b=1}^{B}.
```

The paper clusters the two branch collections separately into `S`
groups.

## Branch-Specific Centroids

Let:

```math
\mathcal{S}^{p}
=
\left\{
S_1^{p},
\ldots,
S_S^{p}
\right\},
```

```math
\mathcal{S}^{q}
=
\left\{
S_1^{q},
\ldots,
S_S^{q}
\right\}.
```

These centroids are generated from current minibatch embeddings in the
auxiliary head space. They are not the persistent memory subqueue centroids
used by the weighted instance branch.

## Cross-Branch Positive Centroid

For anchor `g_{p1}`, select the most similar `q`-branch centroid:

```math
r_p^{\star}
=
\arg\max_{
r\in\left\{
1,\ldots,S
\right\}
}
g_{p1}^{\top}S_r^{q}.
```

Define:

```math
S^{q+}
=
S_{r_p^{\star}}^{q},
```

and the remaining centroids:

```math
\left\{
S_1^{q-},
\ldots,
S_{S-1}^{q-}
\right\}
=
\mathcal{S}^{q}
\setminus
\left\{
S^{q+}
\right\}.
```

Symmetrically:

```math
r_q^{\star}
=
\arg\max_r
g_{q1}^{\top}S_r^{p},
```

```math
S^{p+}
=
S_{r_q^{\star}}^{p}.
```

The group positive is selected by current cross-branch similarity. It is not
given by a shared cluster index established in advance.

## Exact Group-Level Loss

RetCCL writes:

```math
\mathcal{L}_{\mathrm{G\mbox{-}InfoNCE}}
=
-
\frac{1}{2}
\log
\frac{
\exp
\left(
g_{p1}^{\top}S^{q+}/\tau
\right)
}{
\exp
\left(
g_{p1}^{\top}S^{q+}/\tau
\right)
+
\sum_{i=1}^{S-1}
\exp
\left(
g_{p1}^{\top}S_i^{q-}/\tau
\right)
}
```

```math
\phantom{
\mathcal{L}_{\mathrm{G\mbox{-}InfoNCE}}
}
-
\frac{1}{2}
\log
\frac{
\exp
\left(
g_{q1}^{\top}S^{p+}/\tau
\right)
}{
\exp
\left(
g_{q1}^{\top}S^{p+}/\tau
\right)
+
\sum_{i=1}^{S-1}
\exp
\left(
g_{q1}^{\top}S_i^{p-}/\tau
\right)
}.
```

Each direction is an `S`-way centroid classification problem.

## Centroid Softmax Form

For the `p\rightarrow q` direction, define:

```math
\pi_r^{p\rightarrow q}
=
\frac{
\exp
\left(
g_{p1}^{\top}S_r^q/\tau
\right)
}{
\sum_{j=1}^{S}
\exp
\left(
g_{p1}^{\top}S_j^q/\tau
\right)
}.
```

Then:

```math
\ell_{p\rightarrow q}
=
-
\log
\pi_{r_p^{\star}}^{p\rightarrow q}.
```

The target is:

```math
e_{r_p^{\star}}.
```

## Exact Anchor Gradient

Treating centroids and the selected index as fixed:

```math
\nabla_{g_{p1}}
\ell_{p\rightarrow q}
=
\frac{1}{\tau}
\left[
\sum_{r=1}^{S}
\pi_r^{p\rightarrow q}
S_r^q
-
S_{r_p^{\star}}^q
\right].
```

The anchor moves toward its selected opposite-branch centroid and away from
the softmax barycenter of all opposite-branch centroids.

For centroid `S_r^q`:

```math
\nabla_{S_r^q}
\ell_{p\rightarrow q}
=
\frac{
\pi_r^{p\rightarrow q}
-
\mathbf{1}
\left[
r=r_p^{\star}
\right]
}{
\tau
}
g_{p1}.
```

Whether gradients are propagated through the clustering procedure is a
separate implementation question. The assignment and centroid construction
are naturally treated as generated targets in the paper-level algorithm.

## Nearest-Centroid Self-Labeling

The positive index is:

```math
r_p^{\star}
=
\arg\max_r
g_{p1}^{\top}S_r^q.
```

The loss then increases the probability of that same index. Thus the branch
contains a self-labeling loop:

```math
\text{current nearest centroid}
\longrightarrow
\text{categorical target}
\longrightarrow
\text{stronger compatibility}.
```

An incorrect nearest centroid can reinforce its own selection.

## Support Margin

Let the largest and second-largest opposite-branch centroid scores be:

```math
s_{(1)}
\ge
s_{(2)}.
```

Define:

```math
\gamma
=
s_{(1)}
-
s_{(2)}.
```

Small `\gamma` means the categorical group target can switch under a
small embedding or centroid perturbation. The loss is smooth conditional on
the selected target and nonsmooth across the argmax boundary.

## Two Projection Heads, Two Geometries

The weighted instance loss acts on:

```math
g_2(h),
```

while group discrimination acts on:

```math
g_1(h).
```

The shared encoder feature `h` receives both gradient paths:

```math
\nabla_h
\mathcal{L}
=
J_{g_2}(h)^{\top}
\nabla_{g_2}
\mathcal{L}_{\mathrm{W\mbox{-}InfoNCE}}
+
\lambda
J_{g_1}(h)^{\top}
\nabla_{g_1}
\mathcal{L}_{\mathrm{G\mbox{-}InfoNCE}}.
```

The two heads let the instance and group losses organize different projected
spaces while coupling them through the encoder.

## Exact Combined Objective

RetCCL combines the branches as:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{W\mbox{-}InfoNCE}}
+
\lambda
\mathcal{L}_{\mathrm{G\mbox{-}InfoNCE}}.
```

The coefficient `\lambda` controls group-level gradient strength relative
to the weighted instance objective.

## Why Classical InfoNCE Theory Does Not Transfer Directly

In candidate-identification InfoNCE, the positive index is generated by a
known positive conditional and negatives by a proposal.

Here:

```math
r_p^{\star}
=
\arg\max_r
g_{p1}^{\top}S_r^q
```

is selected from the same learned scores later optimized. The candidates are
data-dependent centroids and the positive is a nearest-centroid pseudo-label.

Therefore:

```math
\log S
-
\mathcal{L}_{\mathrm{G\mbox{-}InfoNCE}}
```

should not automatically be reported as a mutual-information lower bound under
the classical independent-negative experiment.

## C/R/G/S Placement

```text
\mathcal{G}:
    two minibatch-specific centroid sets and cross-branch nearest-centroid
    edges

\mathcal{C}:
    auxiliary g_1 head, separate branch clustering, and centroid comparison

\mathcal{R}:
    one selected cross-branch centroid among S candidates

\mathcal{S}:
    symmetric S-way centroid InfoNCE, combined with weighted instance InfoNCE
```

## Surviving Statistic

The group branch preserves cross-augmentation predictability of a
minibatch-derived centroid partition. It encourages instances to gather around
group centers without requiring every same-group pair to be compared directly.
