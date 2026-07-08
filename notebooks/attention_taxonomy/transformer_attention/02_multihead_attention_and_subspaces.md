# Multi-Head Attention And Subspaces

Multi-head attention runs several attention operators in parallel.

For head `m`:

```math
Q^{(m)}
=
HW_Q^{(m)},
\qquad
K^{(m)}
=
HW_K^{(m)},
\qquad
V^{(m)}
=
HW_V^{(m)}.
```

The head output is:

```math
O^{(m)}
=
A^{(m)}V^{(m)},
\qquad
A^{(m)}
=
\mathrm{softmax}
\left(
\frac{Q^{(m)}K^{(m)\top}}{\sqrt{d_m}}
\right).
```

The final output is:

```math
O
=
\mathrm{concat}
\left(
O^{(1)},\ldots,O^{(M)}
\right)
W_O.
```

## Heads As Parallel Measures

For token `u`, head `m` defines:

```math
\nu_{u}^{(m)}
=
\sum_v
a_{uv}^{(m)}\delta_v.
```

The token receives multiple weighted moments:

```math
o_u
=
\mathrm{concat}_{m=1}^{M}
\mathbb{E}_{v\sim\nu_u^{(m)}}[(HW_V^{(m)})_v].
```

Multi-head attention increases the number of conditional statistics preserved
per token.

## Head Redundancy

Heads are useful only if their measures or value subspaces differ:

```math
A^{(m)}
\not\approx
A^{(r)}
```

or:

```math
HW_V^{(m)}
\not\approx
HW_V^{(r)}.
```

If all heads attend similarly and project similarly, the effective number of
heads is smaller than the architectural number.

## WSI Interpretation

Different heads may learn:

```text
tumor-tumor similarity
immune-tumor interface
necrosis proximity
background/artifact suppression
global subtype evidence
```

But the architecture does not guarantee this. It only provides multiple
attention measures and value spaces.

## Dense Summary

Multi-head attention is not one attention map. It is:

```math
\{\nu_u^{(m)}\}_{m=1}^{M}
```

plus head-specific value projections. The surviving token statistic is a
concatenation of multiple weighted moments.
