# Memory Bank As Adaptation

A memory bank stores labeled or unlabeled exemplars:

```math
\mathcal{M}
=
\{(m_k,s_k)\}_{k=1}^{K},
```

where
```math
m_k
```
 is an embedding and
```math
s_k
```
is metadata, label, text, report, or
prototype identity.

For a query:

```math
z_i
=
F_{\phi_0}(X_i),
```

retrieval computes:

```math
\alpha_{ik}
=
\frac{\exp(\mathrm{sim}(z_i,m_k)/\tau)}
{\sum_{\ell\in\mathcal{M}}\exp(\mathrm{sim}(z_i,m_\ell)/\tau)}.
```

## Memory-Conditioned Prediction

A weighted memory prediction is:

```math
\widehat p_i
=
\sum_{k=1}^{K}
\alpha_{ik}y_k.
```

A learned head can use:

```math
r_i
=
\sum_{k=1}^{K}
\alpha_{ik}g(m_k,s_k),
```

then:

```math
\widehat y_i
=
\mathcal{H}(z_i,r_i).
```

## What Is Adapted?

The trainable or curated object may be:

```text
memory contents:
    which cases are stored

memory labels:
    which metadata or outcomes are attached

metric:
    how query and memory are compared

retrieval head:
    how retrieved information affects prediction
```

The foundation encoder can remain frozen.

## Dense Summary

Memory adaptation moves task information into:

```math
\mathcal{M}
```

and the metric over
```math
\mathcal{M}
```
. The model's prediction is only as good as
the memory coverage and similarity notion.
