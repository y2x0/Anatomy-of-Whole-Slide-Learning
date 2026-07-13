# Problem Setup

A whole-slide image
```math
S_i
```
is too large to model directly. The usual first step
is to tile the slide:

```math
S_i
\mapsto
\{x_{ij}\}_{j=1}^{n_i},
```

where each
```math
x_{ij}
```
 is a patch. A patch encoder
```math
E
```
gives:

```math
h_{ij}=E(x_{ij})\in\mathbb{R}^{d}.
```

The slide must then be represented by some mathematical object:

```math
\mathcal{X}_i
=
\mathrm{Rep}(H_i,\Gamma_i),
\qquad
H_i=\{h_{ij}\}_{j=1}^{n_i}.
```

Here
```math
\Gamma_i
```
contains optional structure:

```text
coordinates
adjacency
region hierarchy
scan order
prototype dictionary
retrieval memory
foundation encoder
```

The representation is not just implementation detail. It defines the symmetry
and the inductive bias of the whole model.

## Representation Versus Aggregation

Representation:

```math
S_i\mapsto\mathcal{X}_i.
```

Aggregation:

```math
\mathcal{X}_i\mapsto z_i.
```

Prediction:

```math
z_i\mapsto\widehat{y}_i.
```

The same aggregator can operate on different representation objects, but it
will mean different things.

Example:

```math
z_i=\frac{1}{n_i}\sum_{j=1}^{n_i}h_{ij}
```

is a set readout if
```math
H_i
```
 is an unordered set. If
```math
h_{ij}
```
are contextualized by
a graph first, the same formula becomes the mean of graph-contextualized node
states.

## Symmetry

A representation imposes invariances or equivariances.

Set:

```math
\mathcal{F}(\{h_1,\ldots,h_n\})
=
\mathcal{F}(\{h_{\pi(1)},\ldots,h_{\pi(n)}\})
```

for every permutation
```math
\pi
```
.

Sequence:

```math
\mathcal{F}(h_1,\ldots,h_n)
\ne
\mathcal{F}(h_{\pi(1)},\ldots,h_{\pi(n)})
```

unless the permutation preserves the chosen order.

Graph:

```math
\mathcal{F}(PH,PAP^\top)
=
P\mathcal{F}(H,A)
```

for node permutation matrix
```math
P
```
.

## Starting Objects

Set representation:

```math
\mathcal{X}_i=(\{h_{ij}\}_{j=1}^{n_i},\ \text{no order}).
```

Sequence representation:

```math
\mathcal{X}_i=(h_{i1},h_{i2},\ldots,h_{in_i}).
```

Graph representation:

```math
\mathcal{X}_i=(V_i,E_i,H_i).
```

Hierarchy representation:

```math
\mathcal{X}_i
=
\left(
\{V_i^{(\ell)}\}_{\ell=0}^{L},
\{\pi_i^{(\ell)}\}_{\ell=0}^{L-1}
\right).
```

Distribution representation:

```math
\mathcal{X}_i
=
\mu_i
=
\frac{1}{n_i}\sum_j\delta_{h_{ij}}.
```

Retrieval-memory representation:

```math
\mathcal{X}_i
=
(z_i,\mathcal{M}(z_i)).
```

Foundation-latent representation:

```math
\mathcal{X}_i
=
F_{\text{FM}}(S_i).
```

The same slide can be converted into any of these. The choice changes which
functions are easy to learn and which failures are likely.

## Dense Summary

```math
\boxed{
S_i
\to
H_i
\to
\mathcal{X}_i
\to
z_i
\to
\widehat{y}_i
}
```

The central question is not only how to pool patches. It is what object the
model believes a slide is.
