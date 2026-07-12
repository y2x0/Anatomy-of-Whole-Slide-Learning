# WSI Chain Rule And Context-Mediated Saliency

## Composed WSI Map

```math
F
=
\mathcal{H}
\circ
\mathcal{R}
\circ
\mathcal{C}
\circ
\mathcal{P},
```

where `P` is the patch encoder, `C` is slide context, `R` is readout, and `H`
is the prediction head.

## Pixel-To-Slide Gradient

For pixels of patch `i`:

```math
\frac{\partial F}
{\partial x_i}
=
\sum_{j=1}^{n}
\frac{\partial F}
{\partial h_j'}
\frac{\partial h_j'}
{\partial h_i}
\frac{\partial h_i}
{\partial x_i},
```

where:

```math
h_j'
=
\mathcal{C}_j(H).
```

The term `j=i` is direct. Terms `j` different from `i` are context-mediated.

## Frozen Encoder Boundary

If patch embeddings are precomputed and detached:

```math
h_i
=
\mathrm{stopgrad}
\left(
\mathcal{P}(x_i)
\right),
```

then:

```math
\frac{\partial F}
{\partial x_i}
=
0.
```

One can explain the slide score with respect to embedding coordinates, but not
obtain a true end-to-end pixel gradient from the detached model.

## Patch-Level Saliency

Embedding gradient magnitude:

```math
s_i^{\mathrm{grad}}
=
\left\|
\nabla_{h_i}F
\right\|_2.
```

Signed gradient-times-input:

```math
s_i^{\mathrm{GxI}}
=
\sum_d
h_{id}
\frac{\partial F}
{\partial h_{id}}.
```

These answer sensitivity and local credit questions respectively.

## Coordinate Dependence

For invertible embedding change:

```math
\widetilde h_i
=
Ah_i,
```

the raw gradient norm changes unless `A` is orthogonal. Patch rankings can
depend on representation basis even when the predictor function is preserved
through inverse downstream transformation.

## Self-Attention Paths

For attention layer:

```math
H'
=
A(H)V(H),
```

the Jacobian contains value and routing terms:

```math
\mathrm{d}H'
=
A\,\mathrm{d}V
+
\mathrm{d}A\,V.
```

Gradient saliency propagates both; raw attention rollout captures only selected
routing products.

## Spatial Rendering

Upsampling one embedding score over an entire patch creates patch-level
attribution rendered at pixel resolution. Fine internal morphology requires an
end-to-end differentiable patch encoder or a separate within-patch explanation.
