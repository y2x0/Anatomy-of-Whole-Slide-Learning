# Problem Setup

The mathematical problem is not:

```text
Should we use coordinates?
```

The sharper problem is:

```text
Which transformations of the slide should leave the prediction unchanged?
```

Geometry determines the symmetry class of a whole-slide model.

## Slide With Marks

Let a WSI be tiled into patches:

```math
x_{ij}
\in
\mathbb{R}^{p\times p\times 3},
\qquad
j=1,\ldots,n_i.
```

A patch encoder gives:

```math
h_{ij}
=
E_\phi(x_{ij})
\in
\mathbb{R}^{d}.
```

Each patch also has a coordinate:

```math
c_{ij}
=
(x_{ij}^{\mathrm{coord}},y_{ij}^{\mathrm{coord}})
\in
\Omega_i\subset\mathbb{R}^{2}.
```

The most explicit slide object is the marked point cloud:

```math
X_i
=
\{(h_{ij},c_{ij})\}_{j=1}^{n_i}.
```

The geometry $G_i$ is any structure on the coordinate set:

```math
G_i
=
g(C_i,H_i).
```

It may be empty, fixed, handcrafted, or learned.

## Geometry As Symmetry

If a model ignores coordinates, then any permutation of patches gives the same
prediction:

```math
f(\{h_{ij}\}_{j=1}^{n_i})
=
f(\{h_{i\sigma(j)}\}_{j=1}^{n_i}).
```

This is permutation invariance.

If a model uses coordinates only through relative distances, then translations
may be symmetries:

```math
f(\{(h_j,c_j)\}_j)
=
f(\{(h_j,c_j+a)\}_j).
```

If it uses absolute coordinates, translations are no longer necessarily
symmetries:

```math
f(\{(h_j,c_j)\}_j)
\ne
f(\{(h_j,c_j+a)\}_j).
```

That may be correct when tissue orientation, scanning convention, or anatomical
location matters. It may be a shortcut when absolute location encodes artifacts
or lab-specific processing.

## C/R/G/S Placement

Geometry can enter the model in three places.

First, geometry can enter the context operator:

```math
\widetilde h_{ij}
=
\mathcal{C}_\theta
\left(
h_{ij},
\{h_{ik}:k\in\mathcal{N}_{G_i}(j)\},
G_i
\right).
```

Second, geometry can enter the readout:

```math
z_i
=
\mathcal{R}_\theta
\left(
\{\widetilde h_{ij}\}_{j=1}^{n_i},
G_i
\right).
```

Third, geometry can enter supervision:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda\mathcal{L}_{\mathrm{geom}}(H_i,G_i,S_i).
```

Examples include smoothness penalties, region pseudo-labels, graph
regularizers, contrastive neighborhood losses, or scale-consistency losses.

## Geometry-Induced Equivalence

A geometry-aware model defines an equivalence relation:

```math
X_i
\sim_f
X_{i'}
\quad\Longleftrightarrow\quad
f(X_i;G_i)
=
f(X_{i'};G_{i'}).
```

No-geometry MIL makes many slides equivalent:

```math
\{(h_j,c_j)\}_j
\sim
\{(h_j,c'_j)\}_j
```

whenever the feature multiset is the same.

Graph geometry makes two slides equivalent only if the graph-structured
computation cannot separate them:

```math
f(H_i,A_i)
=
f(H_{i'},A_{i'}).
```

Thus geometry is a statement about which spatial differences can affect the
prediction.

## Spatial Statistics

A geometry-aware method may preserve:

```text
absolute location:
    where morphology appears

relative displacement:
    which morphologies are near each other

neighborhood composition:
    local tissue context around a patch

multi-hop interaction:
    context beyond immediate neighbors

scale structure:
    how fine units compose coarse regions

learned relation:
    task-dependent topology inferred from data
```

Each statistic has a failure mode.

## Minimal Counterexample

Construct two slides with the same patch embeddings but different layout:

```math
X_A
=
\{(h_j,c_j)\}_{j=1}^{n},
\qquad
X_B
=
\{(h_j,c'_j)\}_{j=1}^{n}.
```

If the label depends only on morphology prevalence, then:

```math
y_A=y_B.
```

If the label depends on arrangement, such as tumor-stroma interface, then:

```math
y_A\ne y_B.
```

Any no-geometry model must predict the same value for both:

```math
f_{\varnothing}(X_A)
=
f_{\varnothing}(X_B).
```

This is not a bug in the implementation. It is the theorem implied by the
model's invariance.

## Dense Summary

Geometry asks:

```text
Which spatial differences are visible to the model?
```

The answer depends on where $G_i$ enters:

```text
G in C:
    spatial context changes patch states

G in R:
    spatial structure changes aggregation

G in S:
    spatial assumptions change training pressure
```

The failure mode is always a mismatch between the true task geometry and the
geometry assumed by the model.
