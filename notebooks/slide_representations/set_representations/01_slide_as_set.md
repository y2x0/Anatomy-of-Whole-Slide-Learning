# Slide As Set

The simplest WSI representation is:

```math
\mathcal{X}_i
=
\{h_{ij}\}_{j=1}^{n_i}.
```

No order is assumed. The index $j$ is only a label used to enumerate patches.

The model should satisfy:

```math
f(h_{i1},\ldots,h_{in_i})
=
f(h_{i\pi(1)},\ldots,h_{i\pi(n_i)})
```

for every permutation $\pi$.

This is the core assumption behind classic MIL.

## What Is Kept

The set representation keeps:

```text
patch identities through embeddings
variable bag size
empirical distribution of morphology
```

It discards or does not explicitly represent:

```text
spatial adjacency
absolute coordinates
scan order
region hierarchy
long-range tissue layout
```

Any spatial signal must already be encoded inside $h_{ij}$ or reintroduced
later through positional features.

## Empirical Measure View

A finite set can be written as an empirical measure:

```math
\mu_i
=
\frac{1}{n_i}\sum_{j=1}^{n_i}\delta_{h_{ij}}.
```

A permutation-invariant slide function is a functional of $\mu_i$:

```math
f(H_i)=F(\mu_i).
```

Mean pooling computes the first moment:

```math
m_i
=
\int h\,d\mu_i(h)
=
\frac{1}{n_i}\sum_{j=1}^{n_i}h_{ij}.
```

Prototype methods approximate more of the distribution:

```math
p_{im}
=
\int q_m(h)\,d\mu_i(h).
```

Attention pooling defines a learned reweighted measure:

```math
d\nu_i(h)
\propto
a_\theta(h)\,d\mu_i(h).
```

Then:

```math
z_i=\int h\,d\nu_i(h).
```

## MIL Assumption

In weakly supervised WSI learning, the slide label $y_i$ is observed but patch
labels are not.

Set MIL assumes:

```math
y_i
\sim
p(y\mid \{h_{ij}\}_{j=1}^{n_i}).
```

The supervision is bag-level:

```math
\ell(f(\{h_{ij}\}),y_i).
```

The loss cannot directly identify which patch caused the label without
additional assumptions.

## Dense Summary

```math
\begin{aligned}
\mathcal{X}_i&=\{h_{ij}\}_{j=1}^{n_i},\\
f(H_i)&=f(PH_i)\quad\text{for every permutation }P,\\
\mu_i&=\frac{1}{n_i}\sum_j\delta_{h_{ij}},\\
z_i&=\mathcal{R}(\mu_i).
\end{aligned}
```

A slide-as-set model is a model of the empirical distribution of patch
morphology with no explicit geometry.
