# Full HEAT Forward Map

This note puts the Chan et al. pipeline into one mathematical chain.

## Construction

Start with a WSI:

```math
X_i.
```

The heterogeneous graph construction map is:

```math
\Gamma_{\mathrm{hetero}}:
X_i
\longmapsto
G_i.
```

The constructed graph is:

```math
G_i
=
\left(
H_i^{(0)},
B_i^{(0)},
E_i,
\tau_i,
\phi_i
\right).
```

The pieces are:

```text
H_i^(0):
    KimiaNet patch embeddings

B_i^(0):
    initial edge-attribute embeddings derived from Pearson correlation

E_i:
    feature-kNN support, oriented source-neighbor to target for these formulas

tau_i:
    patch node pseudo-type from HoVer-Net majority vote

phi_i:
    Pearson-correlation edge attribute
```

Expanded:

```math
H_i^{(0)}
=
\{h_{iv}^{(0)}\}_{v\in V_i},
```

```math
E_i
=
\{(s,t):s\in\mathcal{K}_k(t)\},
```

```math
\tau_i:
V_i
\to
\mathcal{T},
```

```math
\phi_i:
E_i
\to
\mathcal{B},
```

```math
B_i^{(0)}
=
\left\{
b_{ie}^{(0)}
=
\mathrm{Embed}_{\mathrm{edge}}
\left(
\phi_i(e)
\right)
:
e\in E_i
\right\}.
```

If an implementation stores query-to-neighbor edges, the message-passing
orientation above is the transposed adjacency convention.

## Context

HEAT applies `L` heterogeneous edge-attribute transformer layers:

```math
\left(
H_i^{(\ell)},
B_i^{(\ell)}
\right)
=
\mathcal{C}_{\theta}^{\mathrm{HEAT},\ell}
\left(
H_i^{(\ell-1)},
B_i^{(\ell-1)};
E_i,
\tau_i
\right).
```

After `L` layers:

```math
\widetilde H_i
=
H_i^{(L)}.
```

At each layer, the context operator uses:

```text
node-type-specific projections:
    tau controls the Q/K/V projection families

edge-attribute modulation:
    B_i^(ell-1) controls edge gates or bilinear compatibility

feature-kNN attention:
    sources are normalized per target node over K_k(target)
```

A static-edge simplification is possible by holding:

```math
B_i^{(\ell)}
=
B_i^{(0)}
```

for every layer. The essential paper-level object remains the Pearson-derived
edge attribute used to modulate attention.

## Pseudo-Label Pooling

For each type:

```math
a\in\mathcal{T},
```

define:

```math
V_{ia}
=
\{v\in V_i:\tau_i(v)=a\}.
```

Pool nodes of the same pseudo-label type:

```math
z_{ia}
=
\mathcal{R}_a
\left(
\{\widetilde h_{iv}:v\in V_{ia}\}
\right),
\qquad
V_{ia}\neq\varnothing.
```

With a missing-type convention for empty groups, stack type rows:

```math
Z_i^{\mathrm{PL}}
=
\left[
z_{ia}
\right]_{a\in\mathcal{T}}
\in
\mathbb{R}^{|\mathcal{T}|\times d_{\mathrm{PL}}}.
```

This is the PL-pool output. Its row index is fixed by the teacher
pseudo-label vocabulary:

```text
row a means teacher pseudo-label type a
```

The final graph vector is:

```math
z_i
=
\mathcal{R}_{\mathrm{graph}}
\left(
Z_i^{\mathrm{PL}},
m_i
\right).
```

The paper gives mean readout as an example, not as the only possible graph
readout.

## Head And Loss

For `K_cls` slide-level classes:

```math
\widehat p_i
=
\mathrm{softmax}
\left(
z_i W_{\mathrm{cls}}
+
b_{\mathrm{cls}}
\right).
```

The cross-entropy objective is:

```math
\mathcal{L}_i
=
-
\sum_{c=1}^{K_{\mathrm{cls}}}
y_{ic}
\log
\widehat p_{ic}.
```

This is used for WSI-level classification-type tasks:

```text
cancer staging
normal/tumor classification
cancer typing
```

## Localization

After training, define the fixed-graph node-deletion intervention:

```math
G_i^{-v}
=
G_i\setminus\{v\},
```

where node `v` and its incident edges are removed.

The paper's displayed perturbation contrast is:

```math
\Delta_{iv}
=
\mathcal{L}
\left(
y_i,
M_\theta(G_i)
\right)
-
\mathcal{L}
\left(
y_i,
M_\theta(G_i^{-v})
\right).
```

This raw sign means a helpful node has:

```math
\Delta_{iv}
<
0
```

when removing it worsens the loss.

If the desired heatmap convention is "larger means more helpful," use:

```math
I_{iv}^{\mathrm{help}}
=
-\Delta_{iv}.
```

So the positive-helpfulness map is:

```math
v
\mapsto
I_{iv}^{\mathrm{help}}.
```

The distinction matters: `Delta` is the paper-style loss contrast; `I_help` is
a plotting convention.

## One-Line Pipeline

```math
X_i
\xrightarrow{\Gamma_{\mathrm{hetero}}}
\left(
H_i^{(0)},
B_i^{(0)},
E_i,
\tau_i,
\phi_i
\right)
\xrightarrow{\mathrm{HEAT}}
\widetilde H_i
\xrightarrow{\mathrm{PLPool}}
Z_i^{\mathrm{PL}}
\xrightarrow{\mathcal{R}_{\mathrm{graph}}}
z_i
\xrightarrow{\mathcal{H}_{\theta}}
\widehat p_i.
```

## C/R/G/S Placement

```text
\mathcal{G}:
    E_i, tau_i, and phi_i define graph support and heterogeneity

\mathcal{C}:
    HEAT message passing over node and edge states

\mathcal{R}:
    PL pooling followed by graph-level readout

\mathcal{S}:
    slide-level task label and node-deletion loss contrast for localization
```
