# Adapters As Residual Bottlenecks

Adapters insert a small trainable residual module into a frozen network.

For hidden state
```math
u_\ell
```
:

```math
u_{\ell+1}
=
F_{\ell,\phi_0}(u_\ell)
+
A_\eta(u_\ell).
```

A bottleneck adapter has:

```math
A_\eta(u)
=
W_{\mathrm{up}}
\sigma
\left(
W_{\mathrm{down}}u
\right),
```

where:

```math
W_{\mathrm{down}}\in\mathbb{R}^{r\times d},
\qquad
W_{\mathrm{up}}\in\mathbb{R}^{d\times r}.
```

Reference: [Adapter modules](https://arxiv.org/abs/1902.00751).

## Low-Dimensional Residual

The adapter perturbation lies in the column space of
```math
W_{\mathrm{up}}
```
:

```math
A_\eta(u)
\in
\mathrm{span}(W_{\mathrm{up}}).
```

Thus every layer can add task-specific directions, but only through a
bottleneck.

## Relation To LoRA

If
```math
\sigma
```
is identity:

```math
A_\eta(u)
=
W_{\mathrm{up}}W_{\mathrm{down}}u,
```

which is a low-rank linear update. Nonlinearity makes adapters more expressive
than a single linear low-rank perturbation, but the bottleneck remains.

## C/R/G/S Placement

```text
G:
    unchanged unless adapter is placed in coordinate-aware layers

C:
    adapter changes contextual hidden states

R:
    adapter can change readout if inserted into slide encoder layers

S:
    labels train only residual bottlenecks and the task head
```

## Dense Summary

Adapters preserve the pretrained computation as a backbone and add a
task-specific residual field:

```math
F_{\phi_0}(u)
\to
F_{\phi_0}(u)+A_\eta(u).
```

They are useful when the downstream task needs local corrections rather than a
full rewrite of the representation.
