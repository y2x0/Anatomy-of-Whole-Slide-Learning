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
