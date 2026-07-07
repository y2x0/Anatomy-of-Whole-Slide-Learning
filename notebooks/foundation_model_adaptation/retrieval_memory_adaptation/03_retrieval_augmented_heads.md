# Retrieval-Augmented Heads

Retrieval can condition a trainable prediction head.

Let retrieved neighbors be:

```math
\mathcal{N}_K(i)
=
\{k_1,\ldots,k_K\}.
```

Build a retrieval context:

```math
r_i
=
\rho
\left(
\{(m_k,s_k):k\in\mathcal{N}_K(i)\}
\right).
```

The prediction is:

```math
\widehat y_i
=
\mathcal{H}_\omega(z_i,r_i).
```

## Cross-Attention To Memory

A query can attend to memory values:

```math
r_i
=
\sum_{k\in\mathcal{M}}
\alpha_{ik}V m_k,
```

where:

```math
\alpha_{ik}
=
\frac{
\exp((Qz_i)^\top(Km_k)/\tau)
}{
\sum_{\ell}
\exp((Qz_i)^\top(Km_\ell)/\tau)
}.
```

If $Q,K,V$ are trained, the retrieval metric is adapted. If they are frozen,
only the memory supplies task information.

## Dense Summary

Retrieval-augmented prediction inserts memory into the context operator:

```math
z_i
\xrightarrow{\mathrm{retrieve}}
r_i
\xrightarrow{\mathcal{H}}
\widehat y_i.
```

The model can use similar cases, reports, or prototypes without fully
fine-tuning the foundation encoder.
