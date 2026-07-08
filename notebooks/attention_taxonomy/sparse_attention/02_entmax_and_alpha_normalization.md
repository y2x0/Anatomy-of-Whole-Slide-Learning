# Entmax And Alpha Normalization

Entmax is a family of normalizations between softmax and sparsemax.

For scores:

```math
s
\in
\mathbb{R}^{n},
```

the entmax map with parameter `alpha` solves a regularized prediction problem:

```math
\mathrm{entmax}_{\alpha}(s)
=
\arg\max_{p\in\Delta^{n-1}}
\left[
p^\top s
+
H_{\alpha}(p)
\right].
```

Here `H_alpha` is Tsallis entropy.

In the usual entmax convention:

```math
H_{\alpha}(p)
=
\frac{1}{\alpha(\alpha-1)}
\sum_j
\left(
p_j-p_j^\alpha
\right).
```

## Special Cases

Softmax appears at:

```math
\alpha=1.
```

Sparsemax appears at:

```math
\alpha=2.
```

For:

```math
1<\alpha\le 2,
```

the solution can be sparse while remaining smoother than sparsemax.

## Shape Of The Solution

For entmax:

```math
p_j
=
\left[
(\alpha-1)s_j-\tau
\right]_+^{1/(\alpha-1)}.
```

The threshold `tau` is chosen so that:

```math
\sum_jp_j=1.
```

With this scaling, the limit `alpha -> 1` recovers softmax and `alpha=2`
recovers sparsemax:

```math
p_j
=
[s_j-\tau]_+.
```

As `alpha` increases, support tends to shrink.

## Why This Matters For WSI

Softmax assumes every patch contributes some positive amount:

```math
a_j>0.
```

Sparsemax assumes excluded patches contribute exactly zero:

```math
a_j=0
\quad
j\notin S.
```

Entmax gives a tunable middle ground:

```text
some exact zeros
less winner-take-all than hard top-k
sparser visual support than softmax, but not explanation by itself
```

## Support As An Inductive Bias

Let the true evidence set be:

```math
E
\subset
\{1,\ldots,n\}.
```

A sparse attention method implicitly estimates:

```math
\widehat E_\theta
=
\{j:a_j>0\}.
```

The model is only safe when:

```math
E
\subseteq
\widehat E_\theta
```

for task-relevant evidence. If not, evidence is removed before readout.

## Dense Summary

Entmax changes attention from:

```text
all patches always contribute
```

to:

```text
learn a sparse support and average over it
```

The mathematical question becomes not just "how much weight?" but:

```text
which patches are allowed to have nonzero influence?
```
