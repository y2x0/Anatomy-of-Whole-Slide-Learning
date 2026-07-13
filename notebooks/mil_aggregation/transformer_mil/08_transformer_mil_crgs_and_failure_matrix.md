# Transformer MIL C/R/G/S and Failure Matrix

| Family | Context `C` | Readout `R` | Geometry `G` | Surviving statistic | Failure |
|---|---|---|---|---|---|
| TransMIL | self-attention | CLS/global token | PPEG grid injection | relational global token | wrong layout or support approximation |
| exact transformer MIL | dense self-attention | pooling or CLS | optional order/coordinates | pairwise contextual statistic | quadratic cost and shortcut context |
| sampled transformer | sparse/landmark attention | sampled readout | sampler-defined support | approximate relational statistic | rare patch loss |

## Explanation Rule

```text
final attention row -> routing into readout;
full gradient -> local computational sensitivity;
token deletion -> recomputed model effect;
topology/layout swap -> geometry dependence.
```

The transformer family is not “attention pooling with more layers.” Its context
operator changes the object that the readout sees.

## Forward Maps

For contextual transformer MIL, let positional or geometric features be E_i. A
single self-attention block has the schematic map

```math
\widetilde h_{ij}
=
\sum_{\ell\in\mathcal A_i(j)}
\alpha_{ij\ell}W_Vh_{i\ell},
\qquad
\alpha_{ij\ell}
=
\frac{\exp(s_{ij\ell})}
{\sum_{r\in\mathcal A_i(j)}\exp(s_{ijr})}.
```

The support set A_i(j) is all tokens for exact attention, a spatial/window
neighborhood for local attention, or a landmark-induced approximation for
sampled attention. The contextualized states then reach the slide head through

```math
z_i
=
\mathcal R\left(\{\widetilde h_{ij}\}_{j=1}^{n_i}\right),
\qquad
\widehat y_i=\mathcal H(z_i).
```

This separates context from readout: a CLS token or PMA query may be a readout,
whereas token-to-token attention before it is a context operator.

## Approximation And Failure

If exact attention uses an n by n score matrix and a landmark method uses m
landmarks, the interaction approximation has a low-rank form such as

```math
\widetilde K
=
K_{nm}K_{mm}^{\dagger}K_{mn},
\qquad
\mathrm{rank}(\widetilde K)\le m.
```

The approximation can be computationally useful while removing pairwise
directions needed by a rare patch. A support or layout intervention makes the
failure test explicit:

```math
\Delta_{\mathrm{layout}}
=
\left\|
\mathcal H\left(\mathcal R(\mathcal C(H;G))\right)
-
\mathcal H\left(\mathcal R(\mathcal C(H;G'))\right)
\right\|.
```

If G and G' differ only in the claimed spatial arrangement, a nonzero delta is
evidence that geometry entered the computation. It does not by itself establish
that the learned geometry is biologically correct.
