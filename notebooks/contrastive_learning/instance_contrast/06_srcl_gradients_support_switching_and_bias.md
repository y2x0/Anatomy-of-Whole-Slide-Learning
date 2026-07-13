# SRCL Gradients, Support Switching, And Bias

The SRCL loss appears to attract every mined positive. Its exact gradient shows
that attraction is allocated according to the current positive softmax, so the
easiest positive can dominate the entire set.

Primary anchor:

- CTransPath and SRCL: https://pubmed.ncbi.nlm.nih.gov/35952419/

## Fixed-Support Loss

Condition on a selected positive support `\mathcal{P}` and negative
support `\mathcal{N}`. Define logits:

```math
a_p
=
\frac{
z^{\top}p
}{
\tau
},
\qquad
p\in\mathcal{P},
```

```math
b_n
=
\frac{
z^{\top}n
}{
\tau
},
\qquad
n\in\mathcal{N}.
```

Let:

```math
A
=
\sum_{p\in\mathcal{P}}
e^{a_p},
\qquad
B
=
\sum_{n\in\mathcal{N}}
e^{b_n}.
```

The loss is:

```math
\mathcal{L}
=
-
\log
\frac{A}{A+B}.
```

These derivatives condition on the selected support and treat every candidate
feature as fixed. That is the optimization convention for the EMA target and
memory-bank features: gradients update the online branch, while the target
branch changes through its moving-average rule. The top-`S` search itself is
also stop-gradient and contributes no derivative inside a fixed ranking cell.

Define total positive probability:

```math
P_{+}
=
\frac{A}{A+B},
```

within-positive allocation:

```math
\alpha_p
=
\frac{
e^{a_p}
}{
A
},
```

and full-softmax negative probability:

```math
\pi_n
=
\frac{
e^{b_n}
}{
A+B
}.
```

## Exact Logit Gradients

For a positive logit:

```math
\frac{
\partial\mathcal{L}
}{
\partial a_p
}
=
-
\left(
1-P_{+}
\right)
\alpha_p.
```

For a negative logit:

```math
\frac{
\partial\mathcal{L}
}{
\partial b_n
}
=
\pi_n.
```

The total positive derivative mass is:

```math
\sum_{p\in\mathcal{P}}
-
\frac{
\partial\mathcal{L}
}{
\partial a_p
}
=
1-P_{+}.
```

The total negative derivative mass is:

```math
\sum_{n\in\mathcal{N}}
\frac{
\partial\mathcal{L}
}{
\partial b_n
}
=
1-P_{+}.
```

Attraction and repulsion have equal total logit mass, distributed differently
inside their supports.

## Anchor Gradient

Treating candidate vectors as fixed:

```math
\nabla_z
\mathcal{L}
=
\frac{1}{\tau}
\left[
\sum_{n\in\mathcal{N}}
\pi_n n
-
\left(
1-P_{+}
\right)
\sum_{p\in\mathcal{P}}
\alpha_p p
\right].
```

Since:

```math
\sum_{n\in\mathcal{N}}
\pi_n
=
1-P_{+},
```

write the normalized negative barycenter:

```math
\overline n
=
\frac{
\sum_{n\in\mathcal{N}}
\pi_n n
}{
1-P_{+}
}.
```

Then:

```math
\nabla_z
\mathcal{L}
=
\frac{
1-P_{+}
}{
\tau
}
\left(
\overline n
-
\overline p_{\alpha}
\right),
```

where:

```math
\overline p_{\alpha}
=
\sum_{p\in\mathcal{P}}
\alpha_p p.
```

SRCL moves the anchor relative to a softmax-weighted positive barycenter, not
the arithmetic mean of all selected positives.

## Easiest-Positive Dominance

If one positive `p_{\star}` has a larger logit than every other positive:

```math
a_{\star}
-
\max_{p\ne p_{\star}}
a_p
\gg
1,
```

then:

```math
\alpha_{\star}
\rightarrow
1,
```

and:

```math
\overline p_{\alpha}
\rightarrow
p_{\star}.
```

The paired augmentation or one already-close memory neighbor can satisfy most
of the positive attraction. Low-similarity mined positives receive
exponentially small gradient.

## Temperature Limit

For fixed similarities:

```math
\alpha_p
=
\frac{
\exp
\left(
z^{\top}p/\tau
\right)
}{
\sum\limits_{r\in\mathcal{P}}
\exp
\left(
z^{\top}r/\tau
\right)
}.
```

As:

```math
\tau
\rightarrow
0^{+},
```

the positive barycenter approaches the maximally similar positive. As:

```math
\tau
\rightarrow
\infty,
```

the allocation approaches:

```math
\alpha_p
\rightarrow
\frac{1}{|\mathcal{P}|}.
```

Temperature controls both hard-negative concentration and within-positive
competition.

## Top-S Support Is Piecewise Constant

Let memory similarities be:

```math
d_r(z)
=
\frac{
z^{\top}c_r
}{
\lVert z\rVert_2
\lVert c_r\rVert_2
}.
```

The selected support is:

```math
\mathcal{I}_S(z)
=
\mathrm{TopS}
\left\{
d_r(z)
\right\}_{r=1}^{Q}.
```

Within a region where the ranking is unchanged:

```math
\frac{
\partial
\mathcal{I}_S(z)
}{
\partial z
}
=
0
```

in the ordinary algorithmic sense: indices are discrete and selection is not
differentiated.

At a ranking boundary:

```math
d_{(S)}(z)
=
d_{(S+1)}(z),
```

the support can switch discontinuously.

## Top-S Margin

Define the selection margin:

```math
\gamma_S(z)
=
d_{(S)}(z)
-
d_{(S+1)}(z).
```

If normalized similarities change by at most `\varepsilon` per candidate:

```math
\left|
\widetilde d_r
-
d_r
\right|
\le
\varepsilon,
```

then the top-S support is guaranteed unchanged when:

```math
\gamma_S(z)
>
2\varepsilon.
```

Small margin means pseudo-positive identity is unstable even when the embedding
perturbation is small.

## Memory Staleness

Let memory item `c_r` have age `a_r`:

```math
c_r
=
g_{\xi_{t-a_r}}
\left(
f_{\xi_{t-a_r}}
\left(
x_r
\right)
\right).
```

The current query is compared against representations produced by different
historical encoders:

```math
d_r
=
d
\left(
z_t,c_{r,t-a_r}
\right).
```

Thus top-S mining is performed in a time-mixed geometry. The EMA target reduces
drift but does not make:

```math
\xi_{t-a_r}
=
\xi_t.
```

Neighbor rank can change because the sample relation changed or because the
encoder changed.

## Self-Reinforcing Neighborhoods

Let:

```math
G_t
=
\mathrm{TopSGraph}
\left(
Z_t
\right)
```

be the directed mined-positive graph. Training then updates:

```math
Z_{t+1}
=
\mathcal{U}
\left(
Z_t,
G_t
\right).
```

The next graph is:

```math
G_{t+1}
=
\mathrm{TopSGraph}
\left(
Z_{t+1}
\right).
```

This is a discrete outer-loop update, not a differentiable graph layer. The
gradient calculation above describes the inner loss after `G_t` has been
selected; the change from `G_t` to `G_{t+1}` is a support refresh between
optimization steps.

An early neighborhood based on stain, scanner, compression, or organ identity
can be strengthened because it becomes a positive relation in the next
optimization step.

## False Pseudo-Positive

Suppose `p_f` is biologically unrelated but has high current similarity.
Its attractive derivative magnitude is:

```math
\left|
\frac{
\partial\mathcal{L}
}{
\partial a_f
}
\right|
=
\left(
1-P_{+}
\right)
\alpha_f.
```

A high-similarity mistake receives more attraction than a low-similarity,
biologically correct positive. Mining errors are therefore confidence
weighted by the geometry that produced them.

## False Negative Remains

SRCL adds positives but retains a negative denominator. A semantically related
patch not selected into top-S can still be repelled:

```math
U(n)
\approx
U(z),
\qquad
n
\notin
\mathcal{P}_t(z).
```

Its repulsive gradient is:

```math
\frac{
\partial\mathcal{L}
}{
\partial b_n
}
=
\pi_n,
```

which is largest when the false negative is already close.

## Failure Matrix

| Mechanism | Mathematical source | Consequence |
|---|---|---|
| easiest-positive dominance | log-sum numerator | selected positives need not be equally aligned |
| support switching | discrete top-S ranking | nonsmooth target changes |
| stale geometry | historical target features | neighbor relation mixes encoder times |
| confirmation bias | graph rebuilt from learned features | early shortcuts can become supervision |
| residual false negatives | finite top-S support | related unselected patches remain repelled |
| low-temperature concentration | exponential logits | both positive and negative errors become sharper |

## Surviving Statistic

Conditional on support, SRCL preserves a softmax-weighted positive barycenter
against a softmax-weighted negative barycenter. Across optimization steps, even
the support of those barycenters is learned.
