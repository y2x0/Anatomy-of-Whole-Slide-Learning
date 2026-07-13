# Soft Prompts And Context Vectors

Soft prompt tuning replaces hand-written prompt tokens with learned vectors.

For a class name token embedding
```math
e_c
```
, define context vectors:

```math
P_\eta
=
(p_1,\ldots,p_m),
\qquad
p_k\in\mathbb{R}^{d}.
```

The text encoder receives:

```math
v_c(\eta)
=
f_T(p_1,\ldots,p_m,e_c).
```

Only
```math
\eta=\{p_k\}
```
is trained:

```math
\nabla_{\eta}\mathcal{L}\ne 0,
\qquad
\nabla_{\phi_0}\mathcal{L}=0.
```

Reference: [Prompt tuning](https://arxiv.org/abs/2104.08691),
[CoOp](https://arxiv.org/abs/2109.01134).

## Prompt As Classifier Adaptation

The image score becomes:

```math
s_c(x;\eta)
=
\frac{
f_I(x)^\top v_c(\eta)
}{\tau}.
```

Training changes the class prototypes:

```math
v_c(0)
\to
v_c(\eta).
```

The visual encoder remains fixed.

## Capacity

Soft prompt capacity is bounded by:

```math
md
```

trainable prompt parameters, and by the text encoder's response to those
vectors. It can rotate class text embeddings but cannot arbitrarily reshape the
image embedding space.

## Dense Summary

Soft prompts adapt the question, not the visual representation:

```math
x
\xrightarrow{f_I}
u,
\qquad
p_c(\eta)
\xrightarrow{f_T}
v_c(\eta).
```

They work when the right text-side query exposes an already aligned visual
direction.
