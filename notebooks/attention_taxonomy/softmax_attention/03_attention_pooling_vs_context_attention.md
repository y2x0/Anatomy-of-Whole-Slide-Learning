# Attention Pooling Versus Context Attention

Attention can be a readout or a context operator. Confusing these two creates a
lot of false equivalences.

## Readout Attention

Readout attention maps a set of patch states to one slide statistic:

```math
H
=
\{h_j\}_{j=1}^{n}
\quad\longmapsto\quad
z
=
\sum_{j=1}^{n}a_jv_j.
```

The weight vector has shape:

```math
a
\in
\Delta^{n-1}.
```

The output has shape:

```math
z
\in
\mathbb{R}^{d_v}.
```

No patch state is updated. The bag is summarized.

## Context Attention

Context attention maps patch states to updated patch states:

```math
h_u'
=
\sum_{v=1}^{n}a_{uv}W_V h_v.
```

The attention matrix has shape:

```math
A
\in
\mathbb{R}^{n\times n},
\qquad
A\mathbf{1}
=
\mathbf{1}.
```

Each row is a probability distribution:

```math
a_{u:}
\in
\Delta^{n-1}.
```

The output remains a collection:

```math
H'
=
\{h_u'\}_{u=1}^{n}.
```

## Complete-Graph Message Passing

Self-attention can be written as message passing on a complete directed graph:

```math
h_u'
=
\sum_{v\in V}
a_{uv}r_v.
```

The edge weight is:

```math
a_{uv}
=
\mathrm{softmax}_{v}
\left(
\frac{(Qh_u)^\top(Kh_v)}{\sqrt{d_k}}
\right).
```

The graph is dense unless masked.

## Why This Matters In WSI

ABMIL-style attention:

```text
patches do not communicate before pooling
```

Transformer-style attention:

```text
patches communicate before readout
```

Thus they have different C/R placement:

```math
\text{ABMIL:}
\qquad
\mathcal{C}=\mathrm{id},
\quad
\mathcal{R}=\mathrm{attention}.
```

```math
\text{Transformer MIL:}
\qquad
\mathcal{C}=\mathrm{self\ attention},
\quad
\mathcal{R}=\mathrm{CLS/PMA/mean/attention}.
```

Two methods can both say "attention" while preserving different statistics.

## Dense Summary

Readout attention chooses:

```text
what weighted first moment represents the slide?
```

Context attention chooses:

```text
which patch messages update each patch state?
```

The first is an aggregation operator. The second is a data-dependent graph
operator.
