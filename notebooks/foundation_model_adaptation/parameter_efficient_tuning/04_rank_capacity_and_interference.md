# Rank, Capacity, And Interference

PEFT methods are defined by capacity constraints.

For LoRA:

```math
\mathrm{rank}(\Delta W)
\le
r.
```

For bottleneck adapters:

```math
A_\eta(u)
\in
\mathrm{span}(W_{\mathrm{up}}),
\qquad
\dim\mathrm{span}(W_{\mathrm{up}})
\le
r.
```

## Task Direction Condition

Suppose the ideal update is
```math
\Delta W^\star
```
. A rank-
```math
r
```
LoRA update can
represent it only if:

```math
\mathrm{rank}(\Delta W^\star)
\le
r.
```

More generally, the approximation error is:

```math
\inf_{\mathrm{rank}(\Delta W)\le r}
\|\Delta W^\star-\Delta W\|.
```

If this is large, the adaptation class is too small.

## Layer Placement

Let updates be allowed only in layers
```math
\mathcal{L}
```
:

```math
\Delta_\eta
=
\{\Delta W_\ell:\ell\in\mathcal{L}\}.
```

If task information requires changing early morphology features but PEFT is
inserted only in late layers, the model may adapt decision geometry without
fixing the underlying representation.

## Interference

A small update can still damage pretrained geometry. Let
```math
z_0(x)
```
be the frozen
embedding and
```math
z_\eta(x)
```
the adapted embedding. Drift is:

```math
D_{\mathrm{drift}}
=
\mathbb{E}_x
\|z_\eta(x)-z_0(x)\|_2^2.
```

Low parameter count does not guarantee low functional drift.

## Dense Summary

PEFT capacity is not just parameter count. It is:

```text
rank:
    how many update directions are available

placement:
    where those directions enter the network

functional drift:
    how much the representation changes
```

Those three quantities determine whether PEFT is underpowered, useful, or
silently destructive.
