# Patch Encoder As Fixed Map

Patch-level foundation encoders turn image patches into embeddings:

```math
h_{ij}
=
f_{\phi_0}(x_{ij}),
\qquad
h_{ij}\in\mathbb{R}^{d}.
```

The slide model then operates on:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i}.
```

The downstream predictor is:

```math
\widehat y_i
=
\mathcal{H}_\omega
\left(
\mathcal{R}_\psi
(\mathcal{C}_\psi(H_i;G_i))
\right).
```

Only
```math
\psi
```
 and
```math
\omega
```
are trained:

```math
\nabla_{\phi_0}\mathcal{L}
=
0,
\qquad
\nabla_{\psi,\omega}\mathcal{L}
\ne
0.
```

## What The Task Can Change

The task can change:

```text
context:
    patch-to-patch mixing after the encoder

readout:
    which patch statistics survive

head:
    decision boundary on the slide statistic
```

The task cannot change:

```math
f_{\phi_0}(x).
```

If two patches collapse under the frozen encoder:

```math
f_{\phi_0}(x)
=
f_{\phi_0}(x'),
```

then no downstream deterministic head can separate them using only
```math
h
```
:

```math
T(h)
=
T(h').
```

## Patch Geometry Inheritance

The pretrained metric is:

```math
d_0(x,x')
=
\|f_{\phi_0}(x)-f_{\phi_0}(x')\|_2.
```

Downstream MIL inherits this geometry before labels act. A kNN graph, prototype
assignment, or attention score built from frozen features is therefore based on
```math
d_0
```
, not on a task-learned metric unless an additional map is trained.

## C/R/G/S Placement

```text
G:
    optional coordinates or graphs built from frozen features

C:
    trainable post-encoder context, if used

R:
    trainable slide aggregation over fixed patch embeddings

S:
    downstream labels never update the patch encoder
```

## Dense Summary

Frozen patch features make WSI learning:

```math
X_i
\xrightarrow{f_{\phi_0}}
H_i
\xrightarrow{\eta}
\widehat y_i.
```

This is computationally efficient and stable, but every downstream method is
bounded by the information preserved in
```math
H_i
```
.
