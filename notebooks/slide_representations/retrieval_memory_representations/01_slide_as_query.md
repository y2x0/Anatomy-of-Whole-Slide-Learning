# Slide As Query

A standard slide model maps:

```math
S_i\to z_i\to \widehat y_i.
```

A retrieval representation maps:

```math
S_i\to z_i\to \mathcal{N}_K(i)\to \widehat y_i.
```

The slide object is:

```math
\mathcal{X}_i
=
\left(
z_i,
\{(z_k,y_k,m_k):k\in\mathcal{N}_K(i)\}
\right),
```

where $m_k$ may include reports, metadata, diagnoses, survival outcomes, or
patch-level evidence.

## Query Encoder

The query embedding can come from any slide representation:

```math
z_i
=
T(\{h_{ij}\}_{j=1}^{n_i}).
```

For mean pooling:

```math
z_i
=
\frac{1}{n_i}\sum_jh_{ij}.
```

For prototype prevalence:

```math
z_i
=
(p_{i1},\ldots,p_{iM}).
```

For a slide foundation model:

```math
z_i
=
F_{\text{slide}}(H_i,C_i).
```

Retrieval is downstream of representation, but it changes the prediction object.

## Neighbor Set

Given a database:

```math
\mathcal{D}
=
\{(z_k,v_k)\}_{k=1}^{N},
```

with keys $z_k$ and values $v_k$, retrieve:

```math
\mathcal{N}_K(i)
=
\mathrm{TopK}_{k}
\mathrm{sim}(z_i,z_k).
```

The values can be:

```text
diagnosis
organ
molecular status
survival outcome
caption
full report
representative patches
```

## Retrieval As Nonparametric Context

The retrieved neighborhood induces a local empirical distribution:

```math
\widehat p_i(y)
=
\sum_{k\in\mathcal{N}_K(i)}
w_{ik}\delta_{y_k}(y).
```

Weights can be:

```math
w_{ik}
=
\frac{\exp(\mathrm{sim}(z_i,z_k)/\tau)}
{\sum_{\ell\in\mathcal{N}_K(i)}
\exp(\mathrm{sim}(z_i,z_\ell)/\tau)}.
```

The prediction may be:

```math
\widehat y_i
=
\mathrm{argmax}_y
\widehat p_i(y).
```

This is $K$-nearest-neighbor prediction in slide representation space.

## Retrieval As Explanation

A retrieval model can expose:

```text
this slide is similar to these prior cases
```

But similarity is only meaningful relative to the embedding and metric:

```math
\mathrm{sim}(z_i,z_k)
\ne
\text{clinical equivalence}.
```

Retrieval explains the model's memory neighborhood, not necessarily the disease
mechanism.

## Dense Summary

```math
\boxed{
\mathcal{X}_i
=
\left(
z_i,
\mathcal{M}(z_i)
\right)
}
```

A slide-as-query representation makes external cases part of the mathematical
object used for prediction.
