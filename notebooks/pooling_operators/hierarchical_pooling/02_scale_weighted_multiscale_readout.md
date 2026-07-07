# Scale-Weighted Multiscale Readout

Some signals are local. Others are regional or global. Multiscale pooling keeps
statistics from several levels.

Let:

```math
H_i^{(\ell)}
=
\{h_{iv}^{(\ell)}:v\in V_i^{(\ell)}\},
\qquad
\ell=0,\ldots,L.
```

At each scale:

```math
z_i^{(\ell)}
=
\mathcal{R}^{(\ell)}(H_i^{(\ell)}).
```

The multiscale slide representation is:

```math
z_i
=
\psi_\theta
\left(
z_i^{(0)},\ldots,z_i^{(L)}
\right).
```

## Scale-Weighted Sum

A simple version is:

```math
z_i
=
\sum_{\ell=0}^{L}\gamma_\ell z_i^{(\ell)},
\qquad
\gamma_\ell\ge 0,
\qquad
\sum_\ell\gamma_\ell=1.
```

The weights can be fixed or learned.

## Slide-Conditioned Scale Weights

Scale weights can depend on the slide:

```math
\gamma_{i\ell}
=
\frac{\exp(r_\theta(z_i^{(\ell)}))}
{\sum_{m=0}^{L}\exp(r_\theta(z_i^{(m)}))}.
```

Then:

```math
z_i
=
\sum_{\ell=0}^{L}\gamma_{i\ell}z_i^{(\ell)}.
```

This is attention over scales.

## What Survives?

If only the coarsest level is used:

```math
z_i=z_i^{(L)}.
```

All fine-scale evidence must survive every compression step.

If all levels are used:

```math
z_i
=
\psi(z_i^{(0)},\ldots,z_i^{(L)}),
```

fine and coarse statistics can both reach the head.

## C/R/G/S Placement

```text
G:
    hierarchy levels and parent maps

C:
    scale-specific encoders and context operators

R:
    per-scale pooling plus scale fusion

S:
    task loss shaping which scales matter
```

## Dense Summary

Multiscale pooling answers:

```text
which physical scales should survive into z?
```

It is the readout version of the slide-as-hierarchy representation.

