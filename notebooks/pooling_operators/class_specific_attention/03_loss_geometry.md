# Loss Geometry

Class-specific attention has two coupled geometries:

```text
slide-level geometry:
    which class logits separate bags

instance-level geometry:
    which high-attention and low-attention patches become separable
```

CLAM-style training combines them.

## Slide Loss Gradient

Let:

```math
z_i^{(c)}
=
\sum_j a_{ij}^{(c)}h_{ij},
\qquad
o_i^{(c)}
=
q_c^\top z_i^{(c)}.
```

The derivative with respect to the attention score is:

```math
\frac{\partial o_i^{(c)}}{\partial s_{i\ell}^{(c)}}
=
a_{i\ell}^{(c)}
q_c^\top
(h_{i\ell}-z_i^{(c)}).
```

The slide loss therefore changes attention by comparing each patch to the
current class-specific pooled embedding.

## Instance Loss Gradient

For selected pseudo instances:

```math
j\in T_i^{+}(c)\cup T_i^{-}(c),
```

the instance loss adds direct gradients to $h_{ij}$:

```math
\frac{\partial \mathcal{L}_{\text{inst}}}{\partial h_{ij}}
\ne
0.
```

Unselected patches receive no direct instance-clustering gradient:

```math
j\notin T_i^{+}(c)\cup T_i^{-}(c)
\quad\Rightarrow\quad
\frac{\partial \mathcal{L}_{\text{inst}}}{\partial h_{ij}}
=
0.
```

They still receive slide-level gradients through the weighted sum.

## Top-K Selection Is A Hard Bottleneck

The pseudo-instance set is:

```math
T_i^{+}(c)
=
\mathrm{TopK}_j a_{ij}^{(c)}.
```

This selection is not a smooth average. It creates a hard training subset.

Small changes in attention can change the selected set:

```math
T_i^{+}(c)
\to
\widetilde T_i^{+}(c).
```

Thus the instance loss is stable only if attention rankings are stable.

## Class Competition

For multi-class prediction:

```math
p_i^{(c)}
=
\frac{\exp(o_i^{(c)})}
{\sum_{r=1}^{C}\exp(o_i^{(r)})}.
```

The cross-entropy gradient couples classes:

```math
\frac{\partial \mathcal{L}_{\text{slide}}}{\partial o_i^{(c)}}
=
p_i^{(c)}-\mathbf{1}\{y_i=c\}.
```

So class-specific attention heads are not independent. A head can improve by
increasing evidence for its class or by separating from evidence used by other
heads.

## Dense Summary

CLAM-style training changes the geometry by adding:

```math
\lambda\mathcal{L}_{\text{inst}}
```

to a slide-level MIL objective. This does not create true patch labels. It
creates an inductive bias:

```text
attention extremes should be clusterable as positive and negative evidence
```

The method works when attention extremes are reliable. It fails when they are
early-training artifacts or dataset shortcuts.

