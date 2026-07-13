# Coarsening And Multiscale Context

A hierarchy representation needs operators that move information across scale.

There are three basic directions:

```text
bottom-up:
    children to parent

top-down:
    parent to children

lateral:
    neighbors at the same scale
```

## Bottom-Up Coarsening

Given children
```math
\mathrm{Ch}(u)
```
 of a parent node
```math
u
```
, bottom-up coarsening
computes:

```math
h_u^{(\ell+1)}
=
\mathcal{R}_{\ell}
\left(
\{h_v^{(\ell)}:v\in\mathrm{Ch}(u)\}
\right).
```

For mean coarsening:

```math
h_u^{(\ell+1)}
=
\frac{1}{|\mathrm{Ch}(u)|}
\sum_{v\in\mathrm{Ch}(u)}h_v^{(\ell)}.
```

For attention coarsening:

```math
a_{vu}^{(\ell)}
=
\mathrm{softmax}_{v\in\mathrm{Ch}(u)}
s_\theta(h_v^{(\ell)}),
```

```math
h_u^{(\ell+1)}
=
\sum_{v\in\mathrm{Ch}(u)}
a_{vu}^{(\ell)}\phi_\theta(h_v^{(\ell)}).
```

This is MIL repeated locally at every parent node.

## Lateral Context

At each scale, units can exchange information with same-scale neighbors:

```math
\bar h_u^{(\ell)}
=
\mathcal{C}_{\ell}
(h_u^{(\ell)},\{h_v^{(\ell)}:v\in\mathcal{N}_\ell(u)\}).
```

For region-level graphs:

```math
\mathcal{N}_\ell(u)
=
\mathrm{kNN}(c_u;\{c_v:v\in V_i^{(\ell)}\}).
```

For image-pyramid transformers, lateral context can be full attention among
tokens at the same scale:

```math
\bar H^{(\ell)}
=
\mathrm{Transformer}_\ell(H^{(\ell)}+P^{(\ell)}).
```

## Top-Down Conditioning

Parent context can condition child states:

```math
\widetilde h_v^{(\ell)}
=
\phi_\theta
\left(
h_v^{(\ell)},
h_{\pi(v)}^{(\ell+1)}
\right).
```

This lets a patch representation depend on the larger region it belongs to.

Example:

```text
the same nuclear pattern may mean different things in tumor, stroma, or necrosis
```

Mathematically, top-down conditioning makes the child state nonlocal:

```math
\widetilde h_v^{(\ell)}
=
F_\theta
\left(
h_v^{(\ell)},
\mathcal{R}_{\ell}
\{h_w^{(\ell)}:w\in\mathrm{Ch}(\pi(v))\}
\right).
```

## Alternating Operators

A hierarchical model can alternate:

```math
H^{(0)}
\xrightarrow{\text{coarsen}}
H^{(1)}
\xrightarrow{\text{lateral}}
\bar H^{(1)}
\xrightarrow{\text{coarsen}}
H^{(2)}.
```

or:

```math
H^{(0)}
\xrightarrow{\text{local attention}}
\bar H^{(0)}
\xrightarrow{\text{pool}}
H^{(1)}
\xrightarrow{\text{global attention}}
\bar H^{(1)}.
```

The order matters. Coarsening before context discards fine variation before
regions interact. Context before coarsening lets each child state carry local
neighborhood information into the parent.

## Computational Benefit

If there are
```math
n
```
patches and full attention costs:

```math
O(n^2),
```

a hierarchy can split the slide into
```math
R
```
 regions, each with
```math
m
```
patches, where
```math
n=Rm
```
.

Local attention within regions costs:

```math
O(Rm^2).
```

Global attention among region tokens costs:

```math
O(R^2).
```

Total:

```math
O(Rm^2+R^2).
```

This is smaller than flat
```math
O(n^2)=O(R^2m^2)
```
only in the regime where the
number of regions is much smaller than the number of patches and each region is
moderate in size. The hierarchy is a computational win when:

```math
Rm^2+R^2
\ll
R^2m^2.
```

If
```math
R\approx n
```
 because the regions are tiny, the global
```math
O(R^2)
```
term recovers
flat attention cost. If regions overlap heavily or multiple scales duplicate
tokens, the actual cost can exceed the simple
```math
O(Rm^2+R^2)
```
accounting.

The cost reduction is not free. It assumes that fine-scale interactions mostly
matter within local regions, and that cross-region effects can be mediated by
coarse tokens.

## Dense Summary

```math
\boxed{
h_u^{(\ell+1)}
=
\mathcal{R}_{\ell}
\left(
\mathcal{C}_{\ell}
\{h_v^{(\ell)}:v\in\mathrm{Ch}(u)\}
\right)
}
```

Hierarchical context is scale-aware compression. The design question is:

```text
At what scale should evidence interact before it is compressed?
```
