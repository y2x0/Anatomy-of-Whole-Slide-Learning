# Bias, Norm, And Head Tuning

The smallest adaptation methods train only selected scalar or affine parameters.

## Bias Tuning

For a layer:

```math
u_{\ell+1}
=
W_\ell u_\ell+b_\ell,
```

bias tuning freezes
```math
W_\ell
```
and trains:

```math
b_\ell.
```

The update is an activation translation:

```math
u_{\ell+1}
\to
u_{\ell+1}+\Delta b_\ell.
```

## Normalization Tuning

Layer normalization has:

```math
\mathrm{LN}_{\gamma,\beta}(u)
=
\gamma\odot
\frac{u-\mu(u)}
{\sigma(u)}
+
\beta.
```

Norm tuning trains:

```math
\gamma,\beta
```

while most weights remain frozen.

## Head Tuning

Head tuning trains:

```math
\widehat y
=
\mathcal{H}_\omega(z),
```

and nothing else. This is a nonlinear probe if
```math
\mathcal{H}_\omega
```
is an MLP.

## Expressivity

These methods can rescale or shift existing features:

```math
h
\to
\gamma\odot h+\beta.
```

They cannot create a new feature direction unless that direction is already
available through existing activations.

## Dense Summary

Bias/norm/head tuning is the limit of PEFT where the model is almost frozen.
It is stable and cheap, but it assumes task information is already encoded and
mostly needs recalibration.
