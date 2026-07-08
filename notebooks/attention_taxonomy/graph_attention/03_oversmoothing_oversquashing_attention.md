# Oversmoothing And Oversquashing With Attention

Attention changes graph message weights, but it does not eliminate graph depth
failure modes.

## Repeated Message Passing

A simplified graph attention layer is:

```math
H^{(\ell+1)}
=
A^{(\ell)}H^{(\ell)}W^{(\ell)}.
```

After `L` layers:

```math
H^{(L)}
\approx
A^{(L-1)}\cdots A^{(0)}H^{(0)}
W^{(0:L)}.
```

Information travels through products of row-stochastic matrices.

## Oversmoothing

If repeated mixing makes node states similar:

```math
h_u^{(L)}
\approx
h_v^{(L)}
```

for many nodes, then discriminative local morphology is washed out.

Attention can slow smoothing by concentrating weights, but if every layer still
averages neighbors, the risk remains.

## Oversquashing

If many distant nodes influence one node through a small cut, information must
be compressed through limited-dimensional messages.

Let a target node depend on a distant set:

```math
B_R(u)
=
\{v:\mathrm{dist}(u,v)\le R\}.
```

If:

```math
|B_R(u)|
\gg
d,
```

then many signals must be encoded into `d` dimensions at intermediate nodes.
Attention can select paths, but it cannot increase channel capacity by itself.

## Edge Attention As Bottleneck Selection

Attention may learn to route mass through a few edges:

```math
a_{uv}
\approx
1
\quad
\text{for selected }v.
```

This can reduce noise, but it also discards alternate paths. If long-range WSI
context requires multiple tissue compartments, hard routing can oversquash the
wrong evidence.

## Dense Summary

Graph attention changes:

```text
which neighbor messages dominate
```

but depth still creates:

```text
repeated averaging
finite-dimensional bottlenecks
topology-dependent communication paths
```

Attention is not a cure for wrong graph geometry.

