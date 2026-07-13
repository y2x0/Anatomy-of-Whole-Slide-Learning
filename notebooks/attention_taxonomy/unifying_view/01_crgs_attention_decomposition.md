# C/R/G/S Attention Decomposition

Attention can enter any part of:

```math
\widetilde H
=
\mathcal{C}(H;G,S),
\qquad
z
=
\mathcal{R}(\widetilde H;G,S),
\qquad
\widehat y
=
\mathcal{H}(z).
```

## Attention In Context

Context attention updates token states:

```math
\widetilde h_u
=
\sum_v a_{uv}W_V h_v.
```

This belongs to:

```math
\mathcal{C}.
```

Examples:

```text
transformer self-attention
graph attention
cross-modal token updates
```

## Attention In Readout

Readout attention summarizes tokens:

```math
z
=
\sum_j a_jW_V h_j.
```

This belongs to:

```math
\mathcal{R}.
```

Examples:

```text
ABMIL
gated attention MIL
CLAM class-specific attention
PMA seed pooling
attention survival readout
```

The same local Q/K/V formula does not determine whether the operator belongs to
`C` or `R`. The output shape and composition do:

```math
\mathcal{C}_{\theta}:
\mathbb{R}^{n\times d}
\to
\mathbb{R}^{n\times d'},
\qquad
\mathcal{R}_{\theta}:
\mathbb{R}^{n\times d'}
\to
\mathbb{R}^{m\times d''}.
```

An attention map with one target query and no token updates is naturally a
readout. An `n`-query map that returns one state per token is naturally a
context operator. A CLS token can occupy both roles across layers, so the
architecture name alone does not fix the C/R placement.

## Geometry In Attention

Geometry enters through:

```math
s_{uv}
=
\psi(h_u,h_v)
+
b(G_{uv}),
```

or through support:

```math
M_{uv}
=
g(G_{uv}).
```

Thus geometry can affect:

```text
compatibility
support
token identity
```

## Supervision In Attention

Supervision changes attention through the loss:

```math
\nabla_\theta
\ell
\left(
\mathcal{H}
\left(
\mathcal{R}_\theta
\left(
\mathcal{C}_\theta(H)
\right)
\right),
y
\right).
```

It can also enter directly through class queries, prompts, pseudo-labels, or
contrastive pairs:

```math
q_c
=
q(S,c).
```

## Dense Summary

Attention is not one coordinate in C/R/G/S. It is an operator pattern that can
appear in:

```text
C:
    token-to-token context

R:
    token-to-slide readout

G:
    positional bias, graph mask, spatial neighborhood

S:
    class query, prompt, pseudo-label, contrastive objective
```
