# Problem Setup

Let a WSI contain image patches and coordinates:

```math
X_i
=
\{(x_{ij},c_{ij})\}_{j=1}^{n_i}.
```

A pretrained foundation model supplies parameters $\phi_0$. It may be a patch
encoder:

```math
h_{ij}
=
f_{\phi_0}(x_{ij}),
```

or a slide encoder:

```math
z_i^{(0)}
=
F_{\phi_0}(\{x_{ij},c_{ij}\}_{j=1}^{n_i}).
```

Downstream adaptation observes task supervision:

```math
S_i
=
(Y_i,\text{optional metadata, text, survival target, or retrieval relation}).
```

The adapted predictor is:

```math
\widehat y_i
=
T_{\phi_0,\eta}(X_i,S_i).
```

The adaptation parameters $\eta$ define what can change.

## Parameter Partition

Write all trainable quantities as:

```math
\Theta
=
(\phi,\psi,\omega,\pi,M),
```

where:

```text
phi:
    foundation encoder parameters

psi:
    context/readout parameters

omega:
    task head parameters

pi:
    prompt, prefix, or adapter parameters

M:
    retrieval memory, prototypes, or exemplar bank
```

An adaptation method chooses a trainable subset:

```math
\Theta_{\mathrm{train}}
\subseteq
\Theta.
```

Frozen-feature learning uses:

```math
\Theta_{\mathrm{train}}
=
(\psi,\omega).
```

Linear probing uses:

```math
\Theta_{\mathrm{train}}
=
\omega.
```

Prompt tuning uses:

```math
\Theta_{\mathrm{train}}
=
\pi.
```

LoRA/adapters use:

```math
\Theta_{\mathrm{train}}
=
(\pi,\omega)
\quad
\text{with }
\Delta_\pi
\text{ low-dimensional}.
```

Full fine-tuning uses:

```math
\Theta_{\mathrm{train}}
=
(\phi,\psi,\omega).
```

## Adaptation Constraint Set

A useful abstraction is:

```math
\phi
=
\phi_0+\Delta,
\qquad
\Delta\in\mathcal{A}.
```

The set $\mathcal{A}$ is method-specific:

```text
linear probe:
    Delta = 0

LoRA:
    Delta W = BA with rank at most r

adapter:
    Delta is a residual bottleneck module

prompt tuning:
    Delta changes input tokens, not most weights

full fine-tuning:
    Delta is unconstrained except by optimization and regularization
```

## Objective

For classification:

```math
\mathcal{L}_{\mathrm{task}}
=
\sum_i
\mathrm{CE}(T_{\phi_0,\eta}(X_i),Y_i).
```

For survival:

```math
\mathcal{L}_{\mathrm{task}}
=
\mathcal{L}_{\mathrm{Cox}}
\quad
\text{or}
\quad
\mathcal{L}_{\mathrm{hazard}}.
```

For retrieval:

```math
\mathcal{L}_{\mathrm{task}}
=
-
\log
\frac{
\sum_{p\in\mathcal{P}(i)}
\exp(z_i^\top z_p/\tau)
}{
\sum_{m\in\mathcal{M}}
\exp(z_i^\top z_m/\tau)
}.
```

Regularized adaptation adds a drift penalty:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda
D(\phi_0+\Delta,\phi_0).
```

## Core Question

The foundation model already contains a geometry:

```math
d_0(x,x')
=
\|f_{\phi_0}(x)-f_{\phi_0}(x')\|.
```

Adaptation asks whether the downstream task can be solved by:

```text
1. reading out the existing geometry,
2. reweighting patches within the geometry,
3. prompting a text-aligned geometry,
4. changing a low-rank parameter subspace,
5. changing the full representation.
```

## Dense Summary

Every adaptation method can be written as:

```math
X
\xrightarrow{\phi_0+\Delta_\eta}
H
\xrightarrow{\psi_\eta}
z
\xrightarrow{\omega_\eta}
\widehat y.
```

The method's mathematical meaning is the constraint on $\Delta_\eta$ and the
place where task gradients are allowed to act.
