# SCL-WC Memory Banks And Selected Subbags

Primary anchor:

- Wang et al. "SCL-WC: Cross-Slide Contrastive Learning for Weakly-Supervised
  Whole-Slide Image Classification." NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Paper-Conference.pdf

## Three Bank Types

SCL-WC stores selected patch-feature bags in positive, negative, and hard
negative memories. For a positive slide:

```math
\mathcal{B}_i^{+}
=
\mathrm{TopK}
\left(
\left\{
f_{ij}
\right\},
A_i^{(Y_i)},
k
\right).
```

For a negative slide:

```math
\mathcal{B}_i^{-}
=
\mathrm{BottomK}
\left(
\left\{
f_{ij}
\right\},
A_i,
k
\right),
```

```math
\mathcal{B}_i^{\mathrm{hard}}
=
\mathrm{TopK}
\left(
\left\{
f_{ij}
\right\},
A_i,
k
\right).
```

Different positive disease categories use separate positive memories.

## Anchor-Conditional Sets

For an anchor bag, define:

```math
\mathcal{P}(\mathcal{B})
=
\text{same-class positive bags},
```

```math
\mathcal{N}(\mathcal{B})
=
\text{negative bags},
```

```math
\mathcal{H}(\mathcal{B})
=
\text{hard-negative bags}.
```

The total comparison collection is:

```math
\mathcal{Q}
=
\mathcal{P}
\cup
\mathcal{N}
\cup
\mathcal{H}.
```

## Selection Changes The Empirical Distribution

If slide patches follow empirical measure:

```math
\widehat\mu_i
=
\frac{1}{L_i}
\sum_{j=1}^{L_i}
\delta_{f_{ij}},
```

the top-k bank follows:

```math
\widehat\mu_i^{+,k}
=
\frac{1}{k}
\sum_{j\in I_i^{+,k}}
\delta_{f_{ij}}.
```

Cross-slide contrast acts on attention-truncated tails, not full slide
distributions.

## Hard Negatives Are Model-Defined

A patch enters the hard-negative bank because the current model gives it high
attention on a negative slide. This can mean:

```math
\text{tumor-like benign morphology},
```

or:

```math
\text{current attention error or shortcut}.
```

The loss does not observe which explanation is correct.

## Memory Staleness

Let feature stored at iteration `r-a` be:

```math
z^{\mathrm{mem}}
=
z_{\theta_{r-a}}(x).
```

The current anchor uses the present encoder state. Staleness error is:

```math
\Delta_a(x)
=
z_{\theta_r}(x)
-
z_{\theta_{r-a}}(x).
```

Memory enlarges cross-slide support but compares representations produced by
different model states.
