# Task Information Injection

Task information can enter a pretrained model at different mathematical
locations.

## Head Injection

Only the final head moves:

```math
\widehat y
=
\mathcal{H}_\omega(z_0),
\qquad
z_0=F_{\phi_0}(X).
```

Task information:

```math
S
\to
\omega.
```

## Readout Injection

Patch features are frozen, but aggregation moves:

```math
h_j
=
f_{\phi_0}(x_j),
\qquad
z
=
\mathcal{R}_\psi(\{h_j\}).
```

Task information:

```math
S
\to
\psi,\omega.
```

## Prompt Injection

Weights are frozen, but task tokens move:

```math
z
=
F_{\phi_0}(X;P_\eta).
```

Task information:

```math
S
\to
P_\eta.
```

## Parameter Injection

A constrained update moves:

```math
\phi
=
\phi_0+\Delta_\eta.
```

Task information:

```math
S
\to
\Delta_\eta.
```

## Memory Injection

The model retrieves from task memory:

```math
\widehat y
=
\mathcal{H}(z,\mathrm{Retrieve}(z,\mathcal{M})).
```

Task information:

```math
S
\to
\mathcal{M}.
```

## Dense Summary

Adaptation should be described by the arrow:

```math
S
\to
\{\omega,\psi,P_\eta,\Delta_\eta,\mathcal{M}\}.
```

That arrow is the method.
