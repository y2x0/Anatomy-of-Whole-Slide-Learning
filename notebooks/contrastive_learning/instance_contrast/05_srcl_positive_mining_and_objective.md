# SRCL Positive Mining And Objective

Semantically-relevant contrastive learning, or SRCL, extends a same-instance
positive pair with top-S memory-bank neighbors. Its defining operation is
positive-set construction in a learned and time-dependent representation
space.

Primary anchor:

- Wang et al. "Transformer-based Unsupervised Contrastive Learning for
  Histopathological Image Classification." Medical Image Analysis 2022.
  https://pubmed.ncbi.nlm.nih.gov/35952419/

## Three Paths

For a source patch `x`, generate two augmented views:

```math
x_1
\sim
\mathcal{A}_1
\left(
\cdot
\mid
x
\right),
\qquad
x_2
\sim
\mathcal{A}_2
\left(
\cdot
\mid
x
\right).
```

SRCL uses:

```text
online path:
    trainable parameters \theta

target path:
    exponential-moving-average parameters \xi

shared-target path:
    the same target parameters \xi, applied to the original view for mining
```

The target update is:

```math
\xi
\leftarrow
m\xi
+
\left(
1-m
\right)
\theta.
```

The target and shared-target paths share parameters; they have different roles
because they receive different views.

## Online And Target Representations

For one direction:

```math
y_1
=
f_{\theta}(x_1),
\qquad
\widehat y_2
=
f_{\xi}(x_2),
```

```math
z_1
=
g_{\theta}(y_1),
\qquad
\widehat z_2
=
g_{\xi}
\left(
\widehat y_2
\right).
```

The paper uses a linear projection head `g`. The swapped direction
reverses which augmented view enters the online and target paths:

```math
z_2
=
g_{\theta}
\left(
f_{\theta}(x_2)
\right),
\qquad
\widehat z_1
=
g_{\xi}
\left(
f_{\xi}(x_1)
\right).
```

## Memory Bank

Let the target memory bank at training step `t` be:

```math
\mathcal{M}_t
=
\left\{
c_{t,1},
\ldots,
c_{t,Q}
\right\}.
```

The bank contains target features from previous minibatches and is updated
after each iteration. For cosine mining:

```math
D
\left(
z,c_{t,r}
\right)
=
\frac{
z^{\top}c_{t,r}
}{
\lVert z\rVert_2
\lVert c_{t,r}\rVert_2
}.
```

Let:

```math
\sigma_t
:
\left\{
1,\ldots,Q
\right\}
\longrightarrow
\left\{
1,\ldots,Q
\right\}
```

sort the memory indices so that:

```math
D
\left(
z,c_{t,\sigma_t(1)}
\right)
\ge
\cdots
\ge
D
\left(
z,c_{t,\sigma_t(Q)}
\right).
```

The mined support is:

```math
\mathcal{N}_{S,t}(z)
=
\left\{
c_{t,\sigma_t(1)},
\ldots,
c_{t,\sigma_t(S)}
\right\}.
```

## Original-View Mining Detail

The method description states that the original, non-altered input is passed
through the shared-target branch and used to retrieve semantically similar
memory samples. The printed cosine equation uses a generic query symbol
`z`. A faithful reconstruction is:

```math
r_x
=
g_{\xi}
\left(
f_{\xi}(x)
\right),
```

```math
\mathcal{N}_{S,t}(x)
=
\mathrm{TopS}
\left\{
D
\left(
r_x,c_{t,r}
\right)
\right\}_{r=1}^{Q}.
```

The original view stabilizes the mining query against two independently
augmented views. The selected memory features then enter the positive set of
the online contrastive direction.

## Positive And Negative Sets

For online anchor `z`, define:

```math
\mathcal{P}_t(z)
=
\left\{
\widehat z_{\mathrm{paired}}
\right\}
\cup
\mathcal{N}_{S,t}(x).
```

Thus:

```math
\left|
\mathcal{P}_t(z)
\right|
=
S+1.
```

Let the negative set be:

```math
\mathcal{K}^{-}_t(z)
=
\left\{
z_1^{-},
\ldots,
z_N^{-}
\right\}.
```

The paper's figure and method describe the memory bank as the source of mined
pseudo-positives and the current minibatch as the source of negatives.

Because the bank is populated from previous minibatches, the current anchor's
own target feature is not inserted into the bank before its top-`S` search.
This prevents the ordinary paired target view from being selected merely by
identity. It does not prevent duplicate images or near-duplicate patches from
appearing in the bank, so the mining rule is not a proof that every selected
pseudo-positive has a different source instance.

## Exact Log-Sum Positive Loss

For one direction, SRCL uses:

```math
\mathcal{L}_2
\left(
z,
\mathcal{P}_t,
\mathcal{K}^{-}_t
\right)
=
-
\log
\frac{
\sum\limits_{p\in\mathcal{P}_t(z)}
\exp
\left(
p^{\top}z/\tau
\right)
}{
\sum\limits_{p\in\mathcal{P}_t(z)}
\exp
\left(
p^{\top}z/\tau
\right)
+
\sum\limits_{n\in\mathcal{K}^{-}_t(z)}
\exp
\left(
n^{\top}z/\tau
\right)
}.
```

This is a log of total positive probability. It is not the outside-log
equal-weight average used by supervised contrastive learning.

Define:

```math
A_{+}(z)
=
\sum_{p\in\mathcal{P}_t(z)}
\exp
\left(
p^{\top}z/\tau
\right),
```

```math
A_{-}(z)
=
\sum_{n\in\mathcal{K}^{-}_t(z)}
\exp
\left(
n^{\top}z/\tau
\right).
```

Then:

```math
\mathcal{L}_2
=
\log
\left(
1
+
\frac{
A_{-}
}{
A_{+}
}
\right).
```

The loss only asks the positive set to receive high total softmax mass.

## Symmetric SRCL Objective

The paper writes:

```math
\mathcal{L}_{\mathrm{SRCL}}
=
\frac{1}{2}
\mathcal{L}_2
\left(
z_1,
\widehat z_2,
z^{-}
\right)
+
\frac{1}{2}
\mathcal{L}_2
\left(
\widehat z_2,
z_1,
z^{-}
\right).
```

The notation compresses the expanded mined positive set into the positive
argument. Operationally, each direction contains the paired view plus the
selected memory positives.

The displayed symmetry is a loss-level symmetry. During optimization, the
online branch supplies the differentiated anchor and the target/shared-target
branch supplies EMA features and mining queries; the target features are
treated as stop-gradient values for that update. Thus the two terms share the
same algebraic form without implying that the EMA parameters receive a
backpropagated gradient.

## Positive-Set Classifier Interpretation

Let the candidate universe be:

```math
\mathcal{C}
=
\mathcal{P}_t(z)
\cup
\mathcal{K}^{-}_t(z).
```

Define the candidate softmax:

```math
\pi(c\mid z)
=
\frac{
\exp
\left(
c^{\top}z/\tau
\right)
}{
\sum\limits_{r\in\mathcal{C}}
\exp
\left(
r^{\top}z/\tau
\right)
}.
```

Then:

```math
\mathcal{L}_2
=
-
\log
\Pr_{\pi}
\left(
C\in\mathcal{P}_t(z)
\mid
z
\right).
```

The training target is set membership:

```math
C
\in
\mathcal{P}_t(z),
```

not identification of one prescribed positive index.

## What SRCL Adds To Instance Contrast

Conventional instance contrast declares:

```math
\mathcal{P}_{\mathrm{instance}}(z)
=
\left\{
\widehat z_{\mathrm{paired}}
\right\}.
```

SRCL declares:

```math
\mathcal{P}_{\mathrm{SRCL}}(z)
=
\mathcal{P}_{\mathrm{instance}}(z)
\cup
\mathcal{N}_{S,t}(x).
```

Therefore it replaces exact identity-only supervision with a learned
neighborhood relation. That relation is:

```math
\text{data dependent},
\qquad
\text{parameter dependent},
\qquad
\text{time dependent}.
```

## C/R/G/S Placement

```text
\mathcal{G}:
    paired views, an original-view mining query, a target memory bank,
    top-S pseudo-positive edges, and minibatch negative edges

\mathcal{C}:
    online CTransPath and EMA target/shared-target CTransPath

\mathcal{R}:
    patch feature followed by a linear contrastive projection

\mathcal{S}:
    symmetric negative log total probability assigned to the positive set
```

## Surviving Relation

SRCL does not learn a fixed tissue ontology. It learns a representation whose
current nearest-neighbor graph is repeatedly used as supervision for its next
update.
