# Parameter Partition And Gradient Flow

Let:

```math
\Theta
=
(\phi,\psi,\omega,\pi,M).
```

The trainable subset determines the adaptation class:

```math
\Theta_{\mathrm{train}}
\subseteq
\Theta.
```

## Gradient Mask

Define a binary mask:

```math
m_k
\in
\{0,1\}
```

for parameter $\Theta_k$. The update is:

```math
\Theta_k
\leftarrow
\Theta_k
-
\alpha
m_k
\nabla_{\Theta_k}\mathcal{L}.
```

Frozen parameters have:

```math
m_k=0.
```

Trainable parameters have:

```math
m_k=1.
```

## Adaptation Families

```text
linear probe:
    m_omega = 1, all other masks 0

frozen feature MIL:
    m_psi = 1, m_omega = 1

prompt tuning:
    m_pi = 1

LoRA/adapters:
    m_pi = 1 with structured Delta

full fine-tuning:
    m_phi = m_psi = m_omega = 1

retrieval:
    memory M is curated, updated, or searched
```

## Gradient Reachability

A task can only change a quantity $q$ if:

```math
\frac{\partial q}{\partial \Theta_k}
\ne
0
```

for some trainable $\Theta_k$. If the encoder is frozen:

```math
\frac{\partial h}{\partial \phi}
\ne
0
\quad
\text{but}
\quad
m_\phi=0,
```

then $h$ is not task-adapted.

## Dense Summary

Gradient flow is the cleanest way to state adaptation:

```math
\nabla_{\Theta_{\mathrm{frozen}}}\mathcal{L}
=
0,
\qquad
\nabla_{\Theta_{\mathrm{train}}}\mathcal{L}
\ne
0.
```

Everything else is commentary unless this partition is explicit.
