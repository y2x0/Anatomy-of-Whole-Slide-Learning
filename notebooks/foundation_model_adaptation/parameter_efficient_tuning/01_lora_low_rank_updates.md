# LoRA Low-Rank Updates

LoRA adapts a weight matrix by adding a low-rank update:

```math
W
=
W_0+\Delta W,
\qquad
\Delta W
=
\frac{\alpha}{r}BA.
```

where:

```math
B\in\mathbb{R}^{d_{\mathrm{out}}\times r},
\qquad
A\in\mathbb{R}^{r\times d_{\mathrm{in}}}.
```

The update rank is bounded:

```math
\mathrm{rank}(\Delta W)
\le
r.
```

Reference: [LoRA](https://arxiv.org/abs/2106.09685).

## Forward Map

For an input activation $u$:

```math
Wu
=
W_0u
+
\frac{\alpha}{r}BAu.
```

The pretrained path remains:

```math
W_0u.
```

The task-specific path is:

```math
\frac{\alpha}{r}B(Au).
```

Thus task information enters through a rank-$r$ bottleneck.

## Parameter Count

Full fine-tuning a matrix has:

```math
d_{\mathrm{out}}d_{\mathrm{in}}
```

trainable parameters. LoRA has:

```math
r(d_{\mathrm{out}}+d_{\mathrm{in}}).
```

When:

```math
r
\ll
\min(d_{\mathrm{out}},d_{\mathrm{in}}),
```

the trainable subspace is much smaller.

## Pathology Interpretation

For pathology FMs, LoRA asks:

```text
Can the downstream morphology distinction be expressed as a low-rank change to
attention or MLP layers?
```

If the task requires many independent new directions, rank $r$ may be too
small.

## Dense Summary

LoRA is not "almost full fine-tuning." It is constrained fine-tuning:

```math
\Delta W
\in
\{BA:\mathrm{rank}(BA)\le r\}.
```

The inductive bias is that task adaptation lies in a low-rank parameter
subspace.
