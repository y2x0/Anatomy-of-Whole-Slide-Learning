# Hierarchical Context And Readout

Hierarchy changes both context and readout.

## Bottom-Up Context

Fine nodes form local region states:

```math
h_u^{(\ell+1)}
=
\mathcal{C}_{\ell\to\ell+1}
\left(
\{h_v^{(\ell)}:v\in\mathrm{Ch}(u)\}
\right).
```

This is bottom-up composition.

## Top-Down Context

Coarse states can return context to fine states:

```math
\widetilde h_v^{(\ell)}
=
\phi_\theta
\left(
h_v^{(\ell)},
h_{\pi(v)}^{(\ell+1)}
\right).
```

This lets a patch representation depend on its region context.

## Within-Level Context

At each level, nodes may also form a graph:

```math
E_i^{(\ell)}
\subset
V_i^{(\ell)}\times V_i^{(\ell)}.
```

Then:

```math
h_v^{(\ell,t+1)}
=
\mathrm{GNN}_\theta
\left(
h_v^{(\ell,t)},
\{h_u^{(\ell,t)}:u\in\mathcal{N}^{(\ell)}(v)\}
\right).
```

HACT-style geometry combines within-level graph context with cross-level
containment.

## Hierarchical Readout

A slide-level readout may use only the top node:

```math
z_i
=
h_{\mathrm{slide}}^{(L)}.
```

or fuse levels:

```math
z_i
=
\sum_{\ell=0}^{L}
\beta_\ell
\mathcal{R}_\ell(\{h_v^{(\ell)}:v\in V_i^{(\ell)}\}).
```

The weights $\beta_\ell$ choose scale emphasis.

## Effective Patch Weight

If each level uses attention, a patch's effective weight is the product along
its path:

```math
w_v^{\mathrm{eff}}
=
\prod_{\ell=0}^{L-1}
a_{\pi^{(\ell)}(v_\ell)\mid v_\ell}^{(\ell)},
```

where $v_{\ell+1}=\pi^{(\ell)}(v_\ell)$.

This product can make some fine regions nearly invisible.

## C/R/G/S Placement

```text
G:
    parent maps, scale levels, and within-level edges

C:
    bottom-up, top-down, and within-level context

R:
    top-level or multiscale readout

S:
    slide labels and pretraining objectives decide which scales are useful
```

## Dense Summary

Hierarchy turns WSI learning into:

```math
\text{local evidence}
\to
\text{regional state}
\to
\text{slide state}.
```

The strength is compositionality. The risk is that early compression decides
what later levels can never recover.
