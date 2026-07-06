# Memory Banks And Similarity

Retrieval requires a memory:

```math
\mathcal{M}
=
\{(k_r,v_r)\}_{r=1}^{N}.
```

Each memory item has:

```text
key:
    vector or structured representation used for search

value:
    retrieved information used for prediction or explanation
```

For slide retrieval:

```math
k_r=z_r.
```

For patch retrieval:

```math
k_r=h_{rj}.
```

For multimodal retrieval:

```math
k_r\in\{z_r,t_r\},
```

where $t_r$ is a text or report embedding.

## Similarity Functions

Cosine similarity:

```math
\operatorname{sim}(z_i,z_k)
=
\frac{z_i^\top z_k}{\|z_i\|\|z_k\|}.
```

Negative Euclidean distance:

```math
\operatorname{sim}(z_i,z_k)
=
-\|z_i-z_k\|_2^2.
```

Mahalanobis similarity:

```math
\operatorname{sim}(z_i,z_k)
=
-(z_i-z_k)^\top\Sigma^{-1}(z_i-z_k).
```

Learned similarity:

```math
\operatorname{sim}_\theta(z_i,z_k)
=
g_\theta(z_i,z_k).
```

The metric is an inductive bias. It defines which variation counts as close.

## Patch-Level Memory

Some systems represent each slide by selected patches:

```math
B_i
=
\{b_{i1},\ldots,b_{iM}\}.
```

The slide-to-slide similarity can be:

```math
\operatorname{sim}(S_i,S_k)
=
\frac{1}{M}
\sum_{m=1}^{M}
\max_{r}
\operatorname{sim}(b_{im},b_{kr}).
```

This is a set-to-set matching score, not a single-vector comparison.

Yottixel-style systems can be interpreted as selecting a compact mosaic or
barcode-like summary that makes WSI retrieval tractable.

## Discrete Latent Memory

A slide may be encoded by discrete codes:

```math
c_{ij}
=
Q(h_{ij})
\in
\{1,\ldots,M\}.
```

Then the slide can be represented by code counts:

```math
p_{im}
=
\frac{1}{n_i}
\sum_j\mathbf{1}\{c_{ij}=m\}.
```

or by a searchable discrete structure:

```math
b_i
\in
\{0,1\}^{M}.
```

Discrete memory can make retrieval fast, but it increases quantization error.

SISH-style systems use learned representations and indexed search structures so
that retrieval can scale to large archives.

## Key-Value Attention View

Retrieval can be written as attention over memory:

```math
\alpha_{ir}
=
\operatorname*{softmax}_{r}
\operatorname{sim}(q_i,k_r),
```

```math
m_i
=
\sum_{r=1}^{N}\alpha_{ir}v_r.
```

Hard top-$K$ retrieval is a sparse approximation:

```math
\alpha_{ir}=0
\qquad
\text{for}
\qquad
r\notin\mathcal{N}_K(i).
```

This makes retrieval a memory-augmented context operator:

```math
\widetilde z_i
=
\mathcal{C}(z_i;\mathcal{M}).
```

## Dense Summary

```math
\boxed{
\mathcal{M}
=
\{(k_r,v_r)\}_{r=1}^{N},
\qquad
\mathcal{M}(z_i)
=
\operatorname{Retrieve}_K(z_i;\mathcal{M})
}
```

The memory bank determines what external evidence is available. The similarity
metric determines what evidence is considered relevant.
