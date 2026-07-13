# Top-k And Block Sparse Attention

Sparse attention can be produced by normalization or by an explicit mask.

Let scores be:

```math
s_{uv}.
```

A mask is:

```math
M_{uv}
\in
\{0,1\}.
```

Masked softmax is:

```math
a_{uv}
=
\frac{M_{uv}\exp(s_{uv})}
{\sum_{r}M_{ur}\exp(s_{ur})}.
```

The mask defines the support before normalization.

The denominator is valid only when query `u` has nonempty support:

```math
\sum_r M_{ur}
>
0.
```

Sparse designs usually enforce this with self-attention, at least one selected
top-k source, a global token, or an explicit empty-row fallback.

## Top-k Attention

For each query `u`, define:

```math
T_k(u)
=
\mathrm{TopK}_{v}
\left(s_{uv}\right).
```

Then:

```math
M_{uv}
=
\mathbf{1}\{v\in T_k(u)\}.
```

The output is:

```math
m_u
=
\sum_{v\in T_k(u)}
a_{uv}r_v.
```

This is a learned sparse neighborhood.

## Block Sparse Attention

For tiled WSIs, patches may be assigned to windows:

```math
b(j)
\in
\{1,\ldots,B\}.
```

Block attention permits:

```math
M_{uv}
=
\mathbf{1}\{b(u)=b(v)\}.
```

or shifted/windowed variants:

```math
M_{uv}
=
\mathbf{1}\{|c_u-c_v|\le r\}.
```

Here coordinates `c_u` create a geometry-induced mask.

## Global Tokens

Many sparse transformers add global tokens:

```math
g
\in
\mathcal{G}.
```

The mask includes:

```math
M_{ug}=1,
\qquad
M_{gv}=1.
```

Global tokens create long-range paths even when patch-patch attention is local.

## Complexity

Full attention costs:

```math
O(n^2d).
```

Top-k attention costs approximately:

```math
O(nkd)
```

only after the candidate scores or neighbors are available. Dense scoring
followed by top-k selection still pays the dense score cost; the savings appear
when candidate generation is already sparse, local, approximate, or indexed.

Block attention with block size `b` costs:

```math
O(nbd).
```

For top-k attention, the displayed `O(nkd)` cost is valid only when each query
already has a candidate set of size `c` with `c` near `k`. If every query must
score all `n` sources before selecting its top `k`, the score computation still
costs:

```math
O(n^2d),
```

followed by selection. Approximate nearest-neighbor search, coordinate windows,
or an indexed graph can reduce the candidate-generation cost to approximately
`O(ncd)`, after which top-k normalization costs `O(nkd)`.

The computational gain is therefore inseparable from a support assumption:
interactions outside the candidate set or mask are unavailable in that layer.

## Hard Support As A Function

Unlike softmax, top-k support is a discontinuous function of the scores at a
tie. If `s_(k)` and `s_(k+1)` denote the k-th and `(k+1)`-th order statistics,
then the selected set is locally stable only while:

```math
s_{(k)}
>
s_{(k+1)}.
```

When the gap approaches zero, an arbitrarily small score perturbation can
replace one source with another. The resulting readout can change sharply when
the two values differ, even if the score perturbation is tiny.

## Dense Summary

Masked sparse attention is:

```math
A
=
\mathrm{softmax}
\left(
S+\log M
\right),
```

where zero mask entries are treated as negative infinity.

The support is no longer only learned by scores. It is imposed by a top-k rule,
window, graph, or global-token design.
