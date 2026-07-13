# End-To-End Gradient Paths

Full fine-tuning makes patch features task-dependent:

```math
h_{ij}
=
f_\phi(x_{ij}).
```

The slide prediction is:

```math
\widehat y_i
=
\mathcal{H}_\omega
\left(
\mathcal{R}_\psi(\{f_\phi(x_{ij})\}_{j=1}^{n_i})
\right).
```

The encoder gradient is:

```math
\nabla_\phi\mathcal{L}_i
=
\sum_{j=1}^{n_i}
\frac{\partial\mathcal{L}_i}{\partial h_{ij}}
\frac{\partial f_\phi(x_{ij})}{\partial\phi}.
```

## Attention Gradient Coupling

If:

```math
z_i
=
\sum_j a_{ij}v(h_{ij}),
```

then a patch affects the loss through its value and through attention
normalization:

```math
\frac{\partial\mathcal{L}}{\partial h_{ij}}
=
\frac{\partial\mathcal{L}}{\partial z_i}
\frac{\partial z_i}{\partial h_{ij}}.
```

The derivative
```math
\partial z_i/\partial h_{ij}
```
includes:

```text
value path:
    changing v(h_ij)

score path:
    changing a_ij and all normalized weights
```

## Computational Constraint

For WSIs,
```math
n_i
```
can be large. Full fine-tuning cost scales with:

```math
\sum_i n_i
```

patch forward/backward passes, plus context/readout memory. This is why many WSI
pipelines freeze patch features.

## Dense Summary

End-to-end fine-tuning is the only adaptation regime where downstream labels can
change the visual encoder itself. It is also the regime where weak slide labels
can most directly rewrite patch features toward shortcuts.
