# KNN And Prototype Decision Rules

The simplest retrieval head is k-nearest-neighbor prediction.

For query $z_i$, define:

```math
\mathcal{N}_K(i)
=
\mathrm{TopK}_{k}
\mathrm{sim}(z_i,m_k).
```

A weighted kNN classifier is:

```math
\widehat p_i(c)
=
\frac{
\sum_{k\in\mathcal{N}_K(i)}
w_{ik}\mathbf{1}\{y_k=c\}
}{
\sum_{k\in\mathcal{N}_K(i)}w_{ik}
},
```

with:

```math
w_{ik}
=
\exp(\mathrm{sim}(z_i,m_k)/\tau).
```

## Prototype Rule

Class prototypes are:

```math
\mu_c
=
\frac{1}{|\mathcal{M}_c|}
\sum_{k:y_k=c}
m_k.
```

The prototype score is:

```math
s_c(z_i)
=
\mathrm{sim}(z_i,\mu_c).
```

## Relation To Linear Probes

If similarity is dot product:

```math
s_c(z)
=
z^\top \mu_c.
```

This is a linear classifier with weights equal to class prototypes. The
difference is how weights are obtained:

```text
linear probe:
    optimized by gradient descent

prototype:
    estimated from memory statistics
```

## Dense Summary

kNN and prototype adaptation are nonparametric or statistic-based task heads.
They adapt through the memory distribution, not necessarily through model
weights.
