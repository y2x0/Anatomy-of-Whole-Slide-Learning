# SCL-WC Memory Banks And Selected Subbags

Primary anchor:

- Wang et al. "SCL-WC: Cross-Slide Contrastive Learning for Weakly-Supervised
  Whole-Slide Image Classification." NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/file/726204cea3ec27790a644e5b379175e3-Paper-Conference.pdf

## Three Bank Types

For a positive slide:

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

## Anchor-Conditional Collection

```math
\mathcal{Q}
=
\mathcal{P}
\cup
\mathcal{N}
\cup
\mathcal{H}.
```

The full empirical slide measure is:

```math
\widehat\mu_i
=
\frac{1}{L_i}
\sum_{j=1}^{L_i}
\delta_{f_{ij}}.
```

The positive bank instead follows:

```math
\widehat\mu_i^{+,k}
=
\frac{1}{k}
\sum_{j\in I_i^{+,k}}
\delta_{f_{ij}}.
```

Cross-slide contrast acts on attention-truncated tails, not full slide
distributions.

## Hard-Negative Ambiguity

High attention on a negative slide can indicate tumor-like benign morphology,
an attention error, or a shortcut. The bank construction does not observe
which interpretation is correct.

## Staleness

```math
z^{\mathrm{mem}}
=
z_{\theta_{r-a}}(x),
```

```math
\Delta_a(x)
=
z_{\theta_r}(x)
-
z_{\theta_{r-a}}(x).
```

Memory enlarges cross-slide support while comparing representations produced
by different model states.
