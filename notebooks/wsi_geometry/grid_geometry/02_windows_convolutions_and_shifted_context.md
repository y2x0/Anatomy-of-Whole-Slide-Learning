# Windows, Convolutions, And Shifted Context

Grid geometry makes local operators natural.

## Convolutional Context

For a dense grid:

```math
H_i[q]
\in
\mathbb{R}^{d},
\qquad
q\in\mathbb{Z}^{2},
```

a convolution layer computes:

```math
\widetilde H_i[q]
=
\sum_{\delta\in\mathcal{K}}
W_{\delta}H_i[q+\delta],
```

where
```math
\mathcal{K}
```
is a finite kernel support.

This assumes translation sharing:

```math
W_{\delta}
\text{ is reused for every }q.
```

## Window Attention

Partition grid sites into windows:

```math
\mathcal{W}_1,\ldots,\mathcal{W}_B.
```

Within each window:

```math
\widetilde h_q
=
\sum_{q'\in\mathcal{W}(q)}
\alpha_{qq'}v_{q'},
```

with:

```math
\alpha_{qq'}
=
\mathrm{softmax}_{q'\in\mathcal{W}(q)}
\left(
\frac{q_q^\top k_{q'}}{\sqrt{d}}
+
b(q-q')
\right).
```

Window attention reduces all-pairs cost:

```math
O(n_i^2)
\quad\to\quad
O(Bm^2),
```

where
```math
m
```
is the number of tokens per window.

## Shifted Windows

If windows are fixed, patches across a window boundary cannot interact in one
layer. A shifted-window layer changes the partition:

```math
\mathcal{W}^{(1)}
\to
\mathcal{W}^{(2)}.
```

After alternating windows, information can cross boundaries.

The effective context graph is the union:

```math
E_{\mathrm{eff}}
=
\bigcup_{\ell=1}^{L}
E(\mathcal{W}^{(\ell)}).
```

## Grid Scan

A sequence model can impose a scan order:

```math
\sigma:\{1,\ldots,n_i\}\to\Lambda_i.
```

Then the context path is:

```math
\sigma(1)\to\sigma(2)\to\cdots\to\sigma(n_i).
```

MambaMIL and state-space MIL methods fit the broader idea that a 2D slide is
linearized into a sequence before long-range context is applied. The geometry is
therefore not only the coordinates, but the chosen ordering.

## C/R/G/S Placement

```text
G:
    grid, windows, and scan order

C:
    convolution, window attention, shifted windows, or sequence recurrence

R:
    usually pooling after grid/sequence contextualization

S:
    downstream labels determine which local or long-range grid context is useful
```

## Dense Summary

Grid operators define context by structured neighborhoods:

```text
convolution:
    fixed local stencil

window attention:
    learned interaction inside fixed windows

shifted windows:
    boundary-crossing through multiple partitions

grid scan:
    2D geometry converted into a 1D path
```

The key question is whether the induced neighborhoods match tissue
neighborhoods.
