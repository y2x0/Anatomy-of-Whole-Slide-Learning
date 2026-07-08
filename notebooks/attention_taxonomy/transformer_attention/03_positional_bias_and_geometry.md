# Positional Bias And Geometry

Self-attention is permutation equivariant unless geometry is injected.

For patch coordinates:

```math
c_j
\in
\mathbb{R}^{2},
```

geometry can enter attention through token features, score bias, or masks.

## Absolute Positional Encoding

Add an embedding:

```math
\tilde h_j
=
h_j+p(c_j).
```

Then:

```math
q_j
=
W_Q\tilde h_j,
\qquad
k_j
=
W_K\tilde h_j.
```

Geometry changes both query and key states.

## Relative Bias

Add a score bias:

```math
s_{uv}
=
\frac{q_u^\top k_v}{\sqrt{d_k}}
+
b(c_u-c_v).
```

The bias directly changes attention weights without altering values.

## Distance Mask

Use a mask:

```math
M_{uv}
=
\mathbf{1}
\{\|c_u-c_v\|\le r\}.
```

Then:

```math
a_{uv}
=
\frac{M_{uv}\exp(s_{uv})}
{\sum_r M_{ur}\exp(s_{ur})}.
```

This changes the support, not just the score.

This expression assumes every query has nonempty support:

```math
\sum_r M_{ur}
>
0.
```

In practice this is usually enforced by self-loops, a global token, or a
fallback rule for empty rows.

## Geometry Channels

The three mechanisms differ:

```text
absolute encoding:
    geometry changes token identity

relative bias:
    geometry changes compatibility

mask:
    geometry changes which messages are allowed
```

These are not interchangeable.

## WSI Caution

Coordinates may be:

```text
pixel coordinates
micron coordinates
normalized slide coordinates
tissue-mask grid coordinates
region-level coordinates
```

Changing coordinate scale changes the geometry the attention operator sees.

## Dense Summary

Geometry enters attention as:

```math
s_{uv}
=
\psi_\theta(h_u,h_v)
+
b_\theta(c_u,c_v),
\qquad
M_{uv}
=
g(c_u,c_v).
```

The key question is whether geometry changes identity, compatibility, support,
or all three.
