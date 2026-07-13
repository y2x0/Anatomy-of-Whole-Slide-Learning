# CLS And PMA Readouts

Transformer context produces token states:

```math
\widetilde H
=
\{\tilde h_j\}_{j=1}^{n}.
```

A slide still needs a readout:

```math
\widetilde H
\to
z.
```

Two common readouts are CLS tokens and pooling by multihead attention.

## CLS Token

Add a learned token:

```math
h_{\mathrm{cls}}^{(0)}
\in
\mathbb{R}^{d}.
```

After transformer layers:

```math
h_{\mathrm{cls}}^{(L)}
=
\mathcal{C}_{\theta}
\left(
h_{\mathrm{cls}}^{(0)},H
\right)_{\mathrm{cls}}.
```

The slide vector is:

```math
z
=
h_{\mathrm{cls}}^{(L)}.
```

The CLS token is a learned state that participates in repeated self-attention.
It is a query for the patch-to-CLS message, but it can also be a source for
patch updates unless the attention mask prevents that direction. It is therefore
not automatically a post-hoc pooling operator.

## PMA Seed Queries

Pooling by multihead attention uses seed queries:

```math
S
=
\{s_m\}_{m=1}^{M}.
```

Each seed attends to token states:

```math
z_m
=
\sum_{j=1}^{n}
a_{mj}V\tilde h_j.
```

The slide representation is:

```math
Z
=
\{z_m\}_{m=1}^{M}.
```

If PMA is applied after the contextual encoder, its source states are fixed for
that readout step and the seeds do not alter the patch states in the same
forward pass. Its factorization is consequently closer to:

```math
H
\xrightarrow{\mathcal{C}}
\widetilde H
\xrightarrow{\mathcal{R}_{\mathrm{PMA}}}
Z,
```

whereas a CLS design commonly puts the learned token inside the context
operator. The two designs can have the same final shape while using different
context operators and different information paths.

With one seed:

```math
M=1,
```

PMA is a learned-query attention readout.

## Readout Bottleneck

If the final head uses one vector:

```math
\widehat y
=
\mathcal{H}(z),
```

then all context has to pass through one statistic.

With multiple seeds:

```math
\widehat y
=
\mathcal{H}(z_1,\ldots,z_M),
```

the model can preserve multiple query-conditioned moments.

## Dense Summary

CLS and PMA expose different readout mechanisms:

```text
CLS:
    one learned token accumulates context inside the encoder and may also
    broadcast context back to patch tokens

PMA:
    one or more learned seed queries read out final token states after the
    contextual encoder
```

The readout determines which statistics survive after transformer context.
