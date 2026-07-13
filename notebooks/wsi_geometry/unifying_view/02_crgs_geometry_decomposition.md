# C/R/G/S Geometry Decomposition

WSI geometry is the
```math
G
```
in:

```math
\widetilde H_i
=
\mathcal{C}(H_i;G_i,S_i),
\qquad
z_i
=
\mathcal{R}(\widetilde H_i;G_i,S_i),
\qquad
\widehat y_i
=
\mathcal{H}(z_i).
```

The central question is:

```text
Where does G enter the computation?
```

## Geometry In Context

Geometry enters context when patch states are updated using spatial relations:

```math
\widetilde h_{ij}
=
\phi_\theta
\left(
h_{ij},
\mathrm{AGG}_{k\in\mathcal{N}_{G_i}(j)}
\psi_\theta(h_{ij},h_{ik},e_{ijk})
\right).
```

Examples:

```text
coordinate attention:
    e_jk = c_j - c_k

graph message passing:
    N_G(j) = graph neighbors

grid window attention:
    N_G(j) = patches in same window

hierarchical context:
    N_G(j) = siblings, parent, children

learned topology:
    N_G(j) = topK learned relation scores
```

## Geometry In Readout

Geometry enters readout when aggregation depends on structure:

```math
z_i
=
\mathcal{R}_\theta
\left(
\{\widetilde h_{ij}\}_{j=1}^{n_i},
G_i
\right).
```

Examples:

```text
region pooling:
    aggregate patches within spatial regions first

graph pooling:
    pool over connected components or graph clusters

spatial moments:
    aggregate h_j c_j^top or location-conditioned statistics

multiscale readout:
    fuse scale-specific summaries
```

## Geometry In Supervision

Geometry enters supervision when training includes spatial assumptions:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda\mathcal{L}_{\mathrm{geom}}.
```

Examples:

```math
\mathcal{L}_{\mathrm{smooth}}
=
\sum_{(j,k)\in E}
\|r_j-r_k\|_2^2,
```

```math
\mathcal{L}_{\mathrm{dist}}
=
\sum_{j,k}A_{jk}\|c_j-c_k\|_2^2,
```

```math
\mathcal{L}_{\mathrm{scale}}
=
\sum_{\ell}
\|z_i^{(\ell)}-z_i^{(\ell+1)}\|_2^2.
```

These losses are geometry priors, not neutral regularizers.

## Geometry Strength

Geometry can be weak or strong.

```text
weak:
    coordinates concatenated to patch features

moderate:
    relative positional bias in attention

strong:
    graph restricts message passing neighborhoods

very strong:
    hierarchy forces early local compression

adaptive:
    learned topology changes geometry per slide
```

## C/R/G/S Table

| Geometry | C | R | G | S |
|---|---|---|---|---|
| none | patchwise or set context | invariant pooling | empty | slide label |
| coordinate | coordinate-aware features or attention | spatial statistics | C_i | task loss may reward location |
| grid | convolution, windows, scan | grid/window pooling | lattice Lambda_i | augmentation and task loss |
| graph | message passing | graph pooling | adjacency A_i | task loss, graph regularizers |
| hierarchy | cross-scale context | region-to-slide pooling | parent maps pi | scale objectives, pseudo labels |
| learned topology | dynamic message passing | dynamic graph pooling | A_theta(H,C) | task loss and topology priors |

## Dense Summary

Geometry is not useful because it exists. It is useful only if it enters the
computation at the right place.

```text
G in C:
    changes patch states

G in R:
    changes the slide statistic

G in S:
    changes what geometry the model is rewarded for learning
```

The failure mode is a mismatch among biological geometry, model geometry, and
training geometry.
