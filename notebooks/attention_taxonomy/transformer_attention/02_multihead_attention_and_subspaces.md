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

The concatenation is not a guarantee that `M` independent statistics survive.
The output projection can mix or suppress head coordinates, and the value
projections can map different heads into overlapping subspaces.

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

One possible diagnostic for attention diversity is the normalized pairwise
distance:

```math
D_A
=
\frac{1}{M(M-1)}
\sum_{m\ne r}
\frac{\|A^{(m)}-A^{(r)}\|_F^2}
{\|A^{(m)}\|_F^2+\|A^{(r)}\|_F^2+\varepsilon}.
```

Value-space diversity can be checked separately, for example through the
principal angles or normalized overlap of the column spaces of
`W_V^(m)`. High `D_A` without value-space diversity, or high value-space
diversity without different attention measures, does not by itself imply that
the final representation contains distinct interpretable mechanisms.

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
