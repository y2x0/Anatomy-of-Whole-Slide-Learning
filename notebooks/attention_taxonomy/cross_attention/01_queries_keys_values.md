# Queries, Keys, And Values

Cross-attention separates three roles:

```text
query:
    what is asking?

key:
    what is being matched?

value:
    what is retrieved after matching?
```

Let query states be:

```math
X
=
\{x_a\}_{a=1}^{m},
\qquad
x_a\in\mathbb{R}^{d_x}.
```

Let source states be:

```math
Y
=
\{y_b\}_{b=1}^{n},
\qquad
y_b\in\mathbb{R}^{d_y}.
```

Project:

```math
q_a
=
W_Qx_a,
\qquad
k_b
=
W_Ky_b,
\qquad
v_b
=
W_Vy_b.
```

Scores are:

```math
s_{ab}
=
\frac{q_a^\top k_b}{\sqrt{d_k}}.
```

Weights are row-normalized over source tokens:

```math
a_{ab}
=
\frac{\exp(s_{ab})}
{\sum_{r=1}^{n}\exp(s_{ar})}.
```

The output for query `a` is:

```math
m_a
=
\sum_{b=1}^{n}a_{ab}v_b.
```

## Shape

The attention matrix has shape:

```math
A
\in
\mathbb{R}^{m\times n}.
```

Rows sum to one:

```math
A\mathbf{1}_n
=
\mathbf{1}_m.
```

Columns do not necessarily sum to one. A source token may be used by many
queries or by none.

## Cross-Attention Is Not Symmetric

Attention from `X` to `Y` is:

```math
A_{X\to Y}
=
\mathrm{softmax}
\left(
Q_XK_Y^\top
\right).
```

Attention from `Y` to `X` is:

```math
A_{Y\to X}
=
\mathrm{softmax}
\left(
Q_YK_X^\top
\right).
```

These are different matrices with different normalizations.

## WSI Examples

Cross-attention appears when:

```text
text prompt queries image patches
pathway token queries histology tokens
class token queries patch tokens
clinical token queries WSI representation
memory query retrieves similar slides
```

The important question is:

```text
which object defines the query geometry?
```

If the query comes from labels or text, the task definition enters the attention
operator before readout.

## Dense Summary

Cross-attention computes:

```math
\boxed{
m_a
=
\mathbb{E}_{b\sim\nu_{a,\theta}}[v_b],
\qquad
\nu_{a,\theta}
=
\sum_ba_{ab}\delta_b
}
```

Each query induces its own measure over the source object.

