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

Here ``memory bank'' names a collection of selected subbags for the WSCL
comparison. The paper does not define these collections as a MoCo-style
momentum queue or as a separate encoder state. For an anchor bag
`\mathcal{B}`, write the collections as:

```math
\mathcal{P}_{\mathcal{B}},
\qquad
\mathcal{N}_{\mathcal{B}},
\qquad
\mathcal{H}_{\mathcal{B}},
```

The positive collection is class-conditional, while the negative and hard
negative collections provide the contrasting bag types.

## Anchor-Conditional Collection

```math
\mathcal{Q}_{\mathcal{B}}
=
\mathcal{P}_{\mathcal{B}}
\cup
\mathcal{N}_{\mathcal{B}}
\cup
\mathcal{H}_{\mathcal{B}}.
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

## Conditional Staleness

The paper does not require these selected subbags to persist across optimizer
steps. If an implementation caches a subbag for `a` updates, then its feature
can be written as:

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

This is therefore an implementation-conditional failure mode. If selected
subbags are recomputed in the same forward pass, `a=0` and there is no
cross-step staleness. If they are cached, the bank enlarges support while
comparing representations produced by different model states.
