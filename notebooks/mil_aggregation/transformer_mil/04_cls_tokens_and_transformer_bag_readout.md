# CLS Tokens and Transformer Bag Readout

## 1. Token Augmentation

Add a learnable class token `h_cls`:

```math
H^0=[h_{\mathrm{cls}};h_1;\ldots;h_n].
```

After transformer context,

```math
z=\widetilde h_{\mathrm{cls}}.
```

The class token is a learned global readout query.

## 2. Attention Path

At one layer, the class-token update contains

```math
\widetilde h_{\mathrm{cls}}
=\sum_j a_{\mathrm{cls},j}V_j.
```

At multiple layers, each `V_j` is already contextualized, so the path from a
patch to the final token includes indirect patch-patch interactions.

## 3. Readout Collision

Different token configurations can yield the same class token:

```math
\widetilde h_{\mathrm{cls}}(H)
=\widetilde h_{\mathrm{cls}}(H').
```

The CLS vector is therefore a bottleneck even when self-attention is globally
expressive.

## 4. Explanation

Class-token attention is a context-routing statistic. Exact patch credit requires
the full computational path, not only the final layer's attention row.

