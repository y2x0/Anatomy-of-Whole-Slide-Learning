# Relative Displacements And Positional Encodings

Absolute coordinates answer:

```text
Where is this patch?
```

Relative displacements answer:

```text
Where is this patch relative to another patch?
```

The distinction matters because many WSI tasks depend on local arrangement, not
absolute slide position.

## Relative Displacement

For two patches:

```math
d_{ijk}
=
c_{ij}-c_{ik}
\in
\mathbb{R}^{2}.
```

A pairwise context operator can use:

```math
m_{ij}
=
\sum_{k=1}^{n_i}
\alpha_{ijk}
\psi_\theta(h_{ij},h_{ik},d_{ijk}).
```

The attention score may be:

```math
s_{ijk}
=
q(h_{ij})^\top k(h_{ik})
+
b_\theta(d_{ijk}).
```

The bias $b_\theta(d)$ favors or suppresses interactions by displacement.

## Distance-Only Geometry

A distance-only model uses:

```math
r_{ijk}
=
\|c_{ij}-c_{ik}\|_2.
```

Then it is invariant to rigid rotations and translations of the coordinate
frame:

```math
\|Rc_j+a-(Rc_k+a)\|_2
=
\|c_j-c_k\|_2.
```

This is useful when orientation should not matter. It is limiting when direction
or anatomy-aligned axes matter.

## Positional Encoding

A coordinate encoding:

```math
\eta:\mathbb{R}^{2}\to\mathbb{R}^{q}
```

maps coordinates into model features. A sinusoidal form can be written:

```math
\eta(c)
=
\left[
\sin(\omega_1^\top c),
\cos(\omega_1^\top c),
\ldots,
\sin(\omega_q^\top c),
\cos(\omega_q^\top c)
\right].
```

Low frequencies encode coarse location. High frequencies encode fine spatial
variation but can be sensitive to registration and tiling artifacts.

## Coordinate Bias In Attention

A coordinate-aware attention layer can be written:

```math
\alpha_{jk}
=
\mathrm{softmax}_{k}
\left(
\frac{q_j^\top k_k}{\sqrt{d}}
+
b_\theta(c_j-c_k)
\right).
```

If $b_\theta$ is strongly negative for large distances, the layer becomes
local. If $b_\theta$ is nearly constant, the layer behaves like global
attention.

## C/R/G/S Placement

```text
G:
    relative coordinate field c_j - c_k

C:
    attention or message passing depends on displacement

R:
    may remain ordinary pooling after coordinate-aware context

S:
    task loss determines which displacements become predictive
```

## Dense Summary

Absolute geometry breaks translation symmetry. Relative geometry preserves it.

```math
c_j
\quad\text{encodes location},
\qquad
c_j-c_k
\quad\text{encodes relation}.
```

Choosing between them is choosing which spatial transformations should be
invisible to the model.
