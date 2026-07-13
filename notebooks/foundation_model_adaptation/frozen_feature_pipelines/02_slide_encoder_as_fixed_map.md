# Slide Encoder As Fixed Map

A slide-level foundation model directly maps the whole slide token set to a
slide embedding:

```math
z_i^{(0)}
=
F_{\phi_0}(\{x_{ij},c_{ij}\}_{j=1}^{n_i}).
```

Downstream adaptation may train only a head:

```math
\widehat y_i
=
\mathcal{H}_\omega(z_i^{(0)}).
```

This is stricter than frozen patch-feature MIL because both context and readout
have already been fixed by pretraining:

```math
\mathcal{C},\mathcal{R}
\subset
F_{\phi_0}.
```

## Task Information Bottleneck

The downstream task sees only:

```math
z_i^{(0)}
\in
\mathbb{R}^{D}.
```

If the target
```math
Y
```
 depends on slide information discarded by
```math
F_{\phi_0}
```
:

```math
I(Y;X\mid z^{(0)})
>
0,
```

then a head on
```math
z^{(0)}
```
cannot be Bayes optimal.

If instead:

```math
P(Y\mid X)
=
P(Y\mid z^{(0)}),
```

then the frozen slide embedding is sufficient for the task.

## Probe Families

A linear probe uses:

```math
\widehat y_i
=
\mathrm{softmax}(Wz_i^{(0)}+b).
```

A small MLP probe uses:

```math
\widehat y_i
=
g_\omega(z_i^{(0)}).
```

The MLP can carve nonlinear boundaries in the frozen slide space, but it still
cannot recover information absent from
```math
z_i^{(0)}
```
.

## C/R/G/S Placement

```text
G:
    fixed inside the pretrained slide encoder

C:
    fixed inside F_phi0

R:
    fixed inside F_phi0

S:
    downstream labels shape only the task head
```

## Dense Summary

Frozen slide encoders are the strongest possible bottleneck:

```math
X_i
\to
z_i^{(0)}
\to
\widehat y_i.
```

They are powerful when pretraining produced a sufficient slide representation
and brittle when task evidence was compressed away.
