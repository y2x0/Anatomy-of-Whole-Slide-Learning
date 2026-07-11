# DINO Centering, Sharpening, And Collapse

Primary anchor:

- Caron et al. "Emerging Properties in Self-Supervised Vision Transformers."
  ICCV 2021. https://arxiv.org/abs/2104.14294

## Center Update

Let teacher logits in batch `r` have mean:

```math
\bar g_t^{(r)}
=
\frac{1}{B|\mathcal{V}_t|}
\sum_{i=1}^{B}
\sum_{v\in\mathcal{V}_t(x_i)}
g_{\theta_t}(v).
```

DINO updates:

```math
c^{(r+1)}
=
m_c c^{(r)}
+
\left(
1-m_c
\right)
\bar g_t^{(r)}.
```

Subtracting `c` discourages one output coordinate from dominating every
sample.

## Sharpening

For centered teacher logits `a`, teacher entropy is:

```math
H_{\tau_t}(a)
=
-\sum_k
p_k(\tau_t)
\log p_k(\tau_t).
```

Its derivative with respect to temperature is:

```math
\frac{\mathrm{d}H_{\tau}(a)}
{\mathrm{d}\tau}
=
\frac{
\mathrm{Var}_{p_{\tau}}(a)
}{
\tau^3
}
\ge
0.
```

Lower temperature therefore lowers entropy unless all logits are equal.

## Two Collapse Directions

Uniform collapse:

```math
P_t(x)
=
\frac{1}{K}\mathbf{1}
\qquad
\forall x.
```

Delta collapse:

```math
P_t(x)
=
e_{k^{\star}}
\qquad
\forall x.
```

Centering acts against persistent coordinate dominance, while sharpening acts
against uniformly high-entropy targets. Their opposition creates a usable
nonconstant regime; neither alone proves global non-collapse.

## Batch Dependence

The center estimates a population logit mean using the current sampling law:

```math
c
\approx
\mathbb{E}_{X\sim p_{\mathrm{train}}}
\left[
g_{\theta_t}(X)
\right].
```

If pathology minibatches are organ-imbalanced or one-WSI dominated, the center
tracks that mixture. The anti-collapse statistic is therefore data-distribution
dependent.

## Effective Teacher Memory

For constant EMA momentum `\lambda`, the weight on a student state `k` steps
old is:

```math
w_k
=
\left(
1-\lambda
\right)
\lambda^k.
```

Its expected age is:

```math
\mathbb{E}[K]
=
\frac{\lambda}{1-\lambda}.
```

Large momentum stabilizes targets but increases lag after rapid representation
changes.

## Assignment-Space Nonidentifiability

For any permutation matrix `P` acting on output coordinates:

```math
P_t'(x)
=
P P_t(x),
\qquad
P_s'(x)
=
P P_s(x),
```

cross-entropy is unchanged. DINO learns a useful partition geometry without
assigning fixed clinical meaning to output indices.

## Failure Principle

Avoiding constant outputs does not ensure retention of rare morphology. A
noncollapsed representation can still organize samples by stain, scanner,
organ, magnification, or tissue abundance.
