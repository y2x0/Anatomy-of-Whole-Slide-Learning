# Prototypes As Quantized Measure

Start with the patch empirical measure:

```math
\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\delta_{h_{ij}}.
```

Choose prototypes:

```math
c_1,\ldots,c_M
\in
\mathbb{R}^{d}.
```

Prototype pooling maps each patch to assignments over prototypes:

```math
q_m(h)
\ge
0,
\qquad
\sum_{m=1}^{M}q_m(h)=1.
```

The slide-level prototype prevalence is:

```math
p_{im}
=
\int q_m(h)\,d\mu_i(h)
=
\frac{1}{n_i}\sum_{j=1}^{n_i}q_m(h_{ij}).
```

Thus:

```math
p_i
\in
\Delta^{M-1}.
```

## Hard Assignment

Hard prototype assignment is:

```math
r(h)
=
\arg\min_m \|h-c_m\|^2.
```

Then:

```math
q_m(h)
=
\mathbf{1}\{r(h)=m\}.
```

The prevalence statistic becomes:

```math
p_{im}
=
\frac{1}{n_i}
\sum_j
\mathbf{1}\{r(h_{ij})=m\}.
```

This is a histogram over morphology types.

## Soft Assignment

Soft assignment uses a temperature:

```math
q_m(h)
=
\frac{\exp(-\|h-c_m\|^2/\tau)}
{\sum_{r=1}^{M}\exp(-\|h-c_r\|^2/\tau)}.
```

As $\tau\to 0$, this approaches hard nearest-prototype assignment. As $\tau$
increases, assignments become diffuse.

## Pushforward View

Let $Q$ map embeddings to prototype assignment vectors:

```math
Q(h)
=
(q_1(h),\ldots,q_M(h)).
```

Prototype pooling computes:

```math
p_i
=
\mathbb{E}_{h\sim\mu_i}[Q(h)].
```

This is a quantized distribution summary. It preserves how much patch mass falls
near each prototype, not the exact patch set.

## What Survives?

Mean pooling preserves:

```math
\int h\,d\mu_i(h).
```

Prototype pooling preserves:

```math
\left(
\int q_1(h)\,d\mu_i(h),
\ldots,
\int q_M(h)\,d\mu_i(h)
\right).
```

Two slides collide under prototype pooling when:

```math
p_i=p_k
```

even if their patches are arranged differently or have different within-prototype
variation.

## Dense Summary

Prototype pooling is:

```math
\boxed{
\mu_i
\xrightarrow{Q}
p_i
=
\mathbb{E}_{\mu_i}[Q(h)]
}
```

It is useful when the task depends on morphology composition. It is weak when
the task depends on rare patterns, spatial arrangement, or within-prototype
variation.

