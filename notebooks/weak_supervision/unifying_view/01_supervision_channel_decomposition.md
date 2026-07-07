# Supervision Channel Decomposition

Every weak-supervision method can be decomposed into four objects:

```text
latent truth U
observed supervision S
model prediction F_theta
training loss L
```

The statistical channel is:

```math
U
\xrightarrow{Q}
S.
```

The learning channel is:

```math
(H,G,S)
\xrightarrow{\mathcal{L}}
\theta.
```

## Latent Truth

For WSI learning:

```math
U_i
=
\left(
Y_i,
Z_i,
B_i,
A_i,
R_i
\right).
```

where:

```text
Y_i:
    true slide target

Z_i:
    latent patch or region labels

B_i:
    burden or prevalence object

A_i:
    latent assignment, attention, or evidence variable

R_i:
    report, region, or annotation object
```

## Observed Signal

The observed signal can be:

```math
S_i
\in
\{
Y_i,
\widetilde Y_i,
M_i\odot Z_i,
\widehat Z_i,
\mathcal{P}_i,
T_i
\}.
```

These correspond to:

```text
bag label
noisy label
partial label
pseudo-label
contrastive relation
text or report pair
```

## Surrogate Loss

The loss is:

```math
\mathcal{L}_{\mathrm{weak}}(\theta)
=
\sum_i
\ell
\left(
F_\theta(H_i,G_i),
S_i
\right).
```

The desired target loss is:

```math
\mathcal{L}_{\mathrm{target}}(\theta)
=
\sum_i
L
\left(
F_\theta(H_i,G_i),
U_i
\right).
```

The gap is:

```math
\Delta(\theta)
=
\mathcal{L}_{\mathrm{weak}}(\theta)
-
\mathcal{L}_{\mathrm{target}}(\theta).
```

Weak-supervision theory is about controlling or understanding this gap.

## Channel Types

Bag label:

```math
S
=
\Gamma(Z).
```

Noisy label:

```math
S
\sim
T(\cdot\mid Y).
```

Partial label:

```math
S
=
(M,M\odot U).
```

Pseudo-label:

```math
S
=
(Y,\Psi_\theta(H,G,Y)).
```

Contrastive label:

```math
S
=
\{(a,b):U_a\sim U_b\}.
```

## Dense Summary

The core decomposition is:

```math
U
\to
S
\to
\mathcal{L}_{\mathrm{weak}}.
```

A method is mathematically clear when it states all three pieces.
