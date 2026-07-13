# Prov-GigaPath: Tile-to-Slide Representation

Reference: [Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w).

## 1. Factorized WSI Encoder

Prov-GigaPath separates tile encoding from slide encoding:

```math
h_j=f_{\phi}(x_j),
\qquad
z=F_{\psi}(\{h_j,c_j\}_{j=1}^{n}).
```

This permits large-scale tile pretraining and a dedicated whole-slide context
operator.

## 2. Long-Context Slide Map

The slide encoder can be written abstractly as

```math
z=\mathcal R_{\psi}
\left(
\mathcal C_{\psi}
(\{h_j,c_j\}_{j=1}^{n})
\right).
```

The ordering, coordinate encoding, and long-context approximation define which
patch relations survive at slide scale.

## 3. Why This Is Not Mean MIL

Mean pooling has

```math
z_{\mathrm{mean}}=\frac1n\sum_jh_j.
```

A slide encoder can preserve higher-order arrangement or long-range context:

```math
F_{\psi}(\{h_j,c_j\})
\ne
F_{\psi}(\{h_{\pi(j)},c_j\})
```

for a general permutation `pi`.

## 4. Failure Mode

Long context does not imply perfect context. Missing tiles, coordinate errors,
regional batching, and finite receptive support can alter the slide statistic.
