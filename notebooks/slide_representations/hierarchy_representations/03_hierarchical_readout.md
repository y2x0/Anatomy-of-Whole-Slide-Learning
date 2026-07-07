# Hierarchical Readout

Hierarchical models can read out at one scale or several scales.

Let $H_i^{(\ell)}$ be the states at level $\ell$.

## Top-Level Readout

The simplest readout uses only the coarsest level:

```math
z_i
=
\mathcal{R}^{(L)}(H_i^{(L)}).
```

If $V_i^{(L)}$ has a single slide node $r_i$, then:

```math
z_i=h_{r_i}^{(L)}.
```

This is efficient but risky. All fine evidence must survive each coarsening
step.

## Multiscale Readout

A richer readout combines levels:

```math
z_i
=
\psi_\theta
\left(
z_i^{(0)},z_i^{(1)},\ldots,z_i^{(L)}
\right),
```

where:

```math
z_i^{(\ell)}
=
\mathcal{R}^{(\ell)}(H_i^{(\ell)}).
```

For example:

```math
z_i
=
\sum_{\ell=0}^{L}\gamma_\ell z_i^{(\ell)},
\qquad
\sum_{\ell=0}^{L}\gamma_\ell=1.
```

or:

```math
z_i
=
\mathrm{MLP}
\left(
[z_i^{(0)};\cdots;z_i^{(L)}]
\right).
```

Multiscale readout preserves signals that appear at different physical scales.

## Worked Shape Sketches

An HIPT-style hierarchy can be written as a two-stage token map. Suppose a slide
is divided into $R_i$ regions, each containing up to $m$ local tiles. A local tile
encoder gives:

```math
x_{irj}^{(0)}
\in
\mathbb{R}^{P\times P\times 3},
\qquad
h_{irj}^{(0)}
=
E_0(x_{irj}^{(0)})
\in
\mathbb{R}^{d_0}.
```

Region $r$ is therefore:

```math
H_{ir}^{(0)}
=
[h_{ir1}^{(0)},\ldots,h_{irm}^{(0)}]
\in
\mathbb{R}^{m\times d_0}.
```

A region encoder compresses local tokens:

```math
u_{ir}
=
E_1(H_{ir}^{(0)})
\in
\mathbb{R}^{d_1}.
```

The slide-level sequence is:

```math
U_i
=
[u_{i1},\ldots,u_{iR_i}]
\in
\mathbb{R}^{R_i\times d_1},
```

and the slide representation is:

```math
z_i
=
E_2(U_i)
\in
\mathbb{R}^{D}.
```

The HACT-style graph hierarchy has a different object. Let:

```math
G_i^{\text{cell}}
=
(V_i^{\text{cell}},E_i^{\text{cell}}),
\qquad
H_i^{\text{cell}}
\in
\mathbb{R}^{N_c\times d_c}.
```

Let tissue units be:

```math
G_i^{\text{tissue}}
=
(V_i^{\text{tissue}},E_i^{\text{tissue}}),
\qquad
|V_i^{\text{tissue}}|=N_t.
```

A hard cell-to-tissue assignment is:

```math
A_i
\in
\{0,1\}^{N_c\times N_t},
\qquad
\sum_{t=1}^{N_t}A_{ct}=1.
```

The initial tissue state can be:

```math
H_i^{\text{tissue},0}
=
D_i^{-1}A_i^\top H_i^{\text{cell}}W,
\qquad
D_{tt}
=
\sum_c A_{ct}.
```

Then tissue-level message passing gives:

```math
H_i^{\text{tissue},L}
=
\mathcal{C}_{\text{GNN}}
\left(
H_i^{\text{tissue},0},
E_i^{\text{tissue}}
\right),
```

followed by graph readout:

```math
z_i
=
\mathcal{R}_{\text{graph}}(H_i^{\text{tissue},L}).
```

HIPT-like hierarchies emphasize nested visual tokens. HACT-like hierarchies
emphasize typed cross-scale assignment plus graph context.

## Regional Evidence Map

A hierarchy can expose a regional score before global prediction:

```math
r_u
=
g_\theta(h_u^{(\ell)}),
\qquad
u\in V_i^{(\ell)}.
```

Then:

```math
\widehat y_i
=
\mathcal{H}
\left(
\{r_u:u\in V_i^{(\ell)}\}
\right).
```

This supports region-level inspection, but the scores are not automatically
causal patch labels. They are statistics shaped by a slide-level loss.

## Surviving Statistics

For bottom-up mean coarsening:

```math
h_u^{(\ell+1)}
=
\frac{1}{|\mathrm{Ch}(u)|}
\sum_{v\in\mathrm{Ch}(u)}h_v^{(\ell)}.
```

After two levels:

```math
h_r^{(\ell+2)}
=
\frac{1}{|\mathrm{Ch}(r)|}
\sum_{u\in\mathrm{Ch}(r)}
\frac{1}{|\mathrm{Ch}(u)|}
\sum_{v\in\mathrm{Ch}(u)}h_v^{(\ell)}.
```

This is not necessarily the same as a flat mean over all fine nodes. Children of
small regions may receive more weight than children of large regions.

The effective fine-node weight is:

```math
w_v
=
\prod_{\ell=0}^{L-1}
\frac{1}{|\mathrm{Ch}(\pi^{(\ell)}(v))|}.
```

Thus hierarchy changes the measure being averaged:

```math
\mu_i
=
\frac{1}{n_i}\sum_j\delta_{h_{ij}}
```

becomes:

```math
\mu_i^{\text{hier}}
=
\sum_j w_j\delta_{h_{ij}}.
```

## Additive Evidence Across Regions

Some tasks are naturally additive across regions:

```math
r_i
=
\sum_{u\in V_i^{(\ell)}}\alpha_u e_u.
```

Here $e_u$ is regional evidence and $\alpha_u$ is a region weight. This works
when multiple independent regions contribute to risk or class probability.

Sparse lesion tasks may instead need:

```math
r_i
=
\max_{u\in V_i^{(\ell)}}e_u.
```

Diffuse phenotypes may need:

```math
r_i
=
\frac{1}{|V_i^{(\ell)}|}
\sum_{u\in V_i^{(\ell)}}e_u.
```

The hierarchy does not decide this automatically. It only decides which regional
units exist before readout.

## Dense Summary

```math
\boxed{
z_i
=
\psi_\theta
\left(
\mathcal{R}^{(0)}(H_i^{(0)}),
\ldots,
\mathcal{R}^{(L)}(H_i^{(L)})
\right)
}
```

Hierarchical readout asks which scales survive. A top-level token is compact,
but multiscale readout keeps more of the slide's mathematical object visible.
