# Attention Operator Matrix

The attention part of a method can be described by five choices.

```math
m_u
=
\sum_{v\in\mathcal{A}(u)}
a_{uv}r_v.
```

## Matrix

```text
score:
    how compatibility is computed

normalization:
    how scores become weights

support:
    which sources are allowed

value:
    what is averaged

readout:
    what statistic survives
```

## Score

Examples:

```math
s_{uv}
=
w^\top\tanh(W_qh_u+W_kh_v)
```

```math
s_{uv}
=
\frac{(W_qh_u)^\top(W_kh_v)}{\sqrt{d}}
```

```math
s_{uv}
=
\psi(h_u,h_v,e_{uv})
```

## Normalization

Examples:

```math
a_{uv}
=
\mathrm{softmax}_v(s_{uv})
```

```math
a_{uv}
=
\mathrm{sparsemax}_v(s_{uv})
```

```math
a_{uv}
=
\mathrm{entmax}_{\alpha,v}(s_{uv})
```

## Support

Examples:

```math
\mathcal{A}(u)
=
\{1,\ldots,n\}
```

```math
\mathcal{A}(u)
=
\{v:(u,v)\in E\}
```

```math
\mathcal{A}(u)
=
\mathrm{TopK}_v(s_{uv})
```

## Value

Examples:

```math
r_v
=
h_v,
\qquad
r_v
=
W_V h_v,
\qquad
r_v
=
\phi(h_v,G_v).
```

## Surviving Statistic

Readout attention preserves:

```math
z
=
\mathbb{E}_{v\sim\nu}[r_v].
```

Transformer context preserves:

```math
\{h_u'\}_{u=1}^{n}.
```

PMA preserves:

```math
\{z_m\}_{m=1}^{M}.
```

## Dense Summary

The attention operator is specified by:

```math
\boxed{
(\psi,\sigma,\mathcal{A},r,\mathrm{readout})
}
```

This does not fully specify a pathology system. Patch encoder, tiling,
sampling, hierarchy, residual blocks, MLPs, pretraining, and supervision can be
as important as the attention operator. The matrix isolates only the weighting
mechanism.

## Operator Invariants

For a nonnegative row-normalized attention map:

```math
a_{uv}
\ge
0,
\qquad
\sum_{v\in\mathcal{A}(u)}a_{uv}
=
1,
\qquad
a_{uv}=0
\text{ when }v\notin\mathcal{A}(u).
```

The message therefore satisfies:

```math
m_u
\in
\mathrm{conv}
\{r_v:v\in\mathcal{A}(u)\}.
```

If a method claims to preserve a statistic outside this convex hull, that
statistic must enter through the value map, a residual path, multiple queries,
or a later nonlinear transformation. Attention weights alone cannot create it.
