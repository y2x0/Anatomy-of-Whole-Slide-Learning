# Region-To-Slide Pooling

Let the slide have regions:

```math
R_i
=
\{1,\ldots,M_i\}.
```

Each region
```math
r
```
has child patches:

```math
\mathrm{Ch}(r)
\subseteq
\{1,\ldots,n_i\}.
```

## Patch-To-Region

Patch states are pooled into region states:

```math
g_{ir}
=
\mathcal{R}_{\text{local}}
\left(
\{u_{ij}:j\in\mathrm{Ch}(r)\}
\right).
```

For local mean:

```math
g_{ir}
=
\frac{1}{|\mathrm{Ch}(r)|}
\sum_{j\in\mathrm{Ch}(r)}u_{ij}.
```

For local attention:

```math
g_{ir}
=
\sum_{j\in\mathrm{Ch}(r)}
a_{ij\mid r}v(u_{ij}).
```

## Region-To-Slide

Region states are then pooled:

```math
z_i
=
\mathcal{R}_{\text{global}}
\left(
\{g_{ir}\}_{r=1}^{M_i}
\right).
```

For region attention:

```math
z_i
=
\sum_{r=1}^{M_i}b_{ir}g_{ir}.
```

## Two-Stage Compression

The full readout is:

```math
H_i
\xrightarrow{\mathcal{R}_{\text{local}}}
G_i
\xrightarrow{\mathcal{R}_{\text{global}}}
z_i.
```

Information lost at the local stage cannot be recovered globally.

## C/R/G/S Placement

```text
G:
    region membership Ch(r)

C:
    optional local context inside each region

R:
    local pooling followed by global pooling

S:
    slide labels shaping both pooling stages
```

## Dense Summary

Hierarchical pooling changes the order of compression:

```math
\text{patches}
\to
\text{regions}
\to
\text{slide}.
```

The benefit is scale structure. The risk is premature loss of patch-level
evidence.
