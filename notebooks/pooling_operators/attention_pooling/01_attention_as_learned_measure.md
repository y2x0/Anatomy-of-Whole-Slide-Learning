# Attention As Learned Measure

ABMIL starts with a bag of instance embeddings:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}\in\mathbb{R}^{d}.
```

The attention score is an instance-wise map:

```math
s_{ij}
=
w^\top\tanh(Vh_{ij}).
```

The normalized attention weight is:

```math
a_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{\ell=1}^{n_i}\exp(s_{i\ell})}.
```

The slide embedding is:

```math
z_i
=
\sum_{j=1}^{n_i}a_{ij}h_{ij}.
```

More generally, if values are transformed:

```math
z_i
=
\sum_{j=1}^{n_i}a_{ij}v_\theta(h_{ij}).
```

## Reweighted Empirical Measure

Mean pooling uses:

```math
\mu_i
=
\frac{1}{n_i}\sum_j\delta_{h_{ij}}.
```

Attention pooling defines a learned measure:

```math
\nu_{i,\theta}
=
\sum_{j=1}^{n_i}a_{ij}\delta_{h_{ij}}.
```

Then:

```math
z_i
=
\int v_\theta(h)\,d\nu_{i,\theta}(h).
```

Thus attention does not preserve the unweighted first moment. It preserves the
first moment under a learned, label-trained reweighting.

## What Is Context-Free Attention?

In ABMIL, the score
```math
s_{ij}
```
 is computed from
```math
h_{ij}
```
before pairwise mixing.
The only cross-instance operation is normalization:

```math
a_{ij}
=
\frac{\exp(s_\theta(h_{ij}))}
{\sum_\ell \exp(s_\theta(h_{i\ell}))}.
```

So the attention score is local, but the weight is globally competitive. Adding
a new instance changes every weight through the denominator.

## Convex Hull Constraint

If
```math
v_\theta(h)=h
```
, then:

```math
z_i
\in
\mathrm{conv}\{h_{i1},\ldots,h_{in_i}\}.
```

Attention cannot create a slide embedding outside the convex hull of instance
embeddings. It can select, interpolate, or average; it cannot represent
interactions unless those interactions are already encoded in
```math
h_{ij}
```
or
```math
v_\theta(h_{ij})
```
.

With a nonlinear value map:

```math
z_i
\in
\mathrm{conv}\{v_\theta(h_{i1}),\ldots,v_\theta(h_{in_i})\}.
```

The convex hull moves to value space, but the readout remains a weighted first
moment.

## Mean And Max As Limits

Uniform scores give mean-like pooling:

```math
s_{ij}=c
\quad\Longrightarrow\quad
a_{ij}=\frac{1}{n_i}.
```

Temperature-scaled attention is:

```math
a_{ij}^{(\beta)}
=
\frac{\exp(\beta s_{ij})}
{\sum_\ell \exp(\beta s_{i\ell})}.
```

As
```math
\beta\to 0
```
:

```math
a_{ij}^{(\beta)}
\to
\frac{1}{n_i}.
```

As
```math
\beta\to\infty
```
, if the maximizer is unique:

```math
a_{ij}^{(\beta)}
\to
\mathbf{1}\{j=j^\star\},
\qquad
j^\star=\arg\max_j s_{ij}.
```

Attention therefore spans a continuum from diffuse mean-like pooling to
winner-take-most pooling. The important caveat is that the maximized object is
the attention score, not necessarily the class evidence score.

## Effective Number Of Instances

The concentration of attention can be measured by:

```math
N_{\mathrm{eff}}(a_i)
=
\frac{1}{\sum_j a_{ij}^2}.
```

Uniform attention gives:

```math
N_{\mathrm{eff}}=n_i.
```

Hard attention gives:

```math
N_{\mathrm{eff}}=1.
```

This is a useful diagnostic because two models can have similar accuracy but
very different surviving statistics: one may average broad morphology, while the
other may depend on a few patches.

## Dense Summary

Attention pooling answers:

```text
which patch distribution should the slide label train us to see?
```

Mathematically:

```math
\boxed{
z_i
=
\mathbb{E}_{h\sim\nu_{i,\theta}}[v_\theta(h)]
}
```

The cost is that every slide is compressed to one weighted first moment. All
unweighted prevalence, multimodality, and spatial layout vanish unless they have
already been encoded into the values being averaged.
