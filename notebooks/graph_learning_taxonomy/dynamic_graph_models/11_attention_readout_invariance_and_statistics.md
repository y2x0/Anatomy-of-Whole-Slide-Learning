# Attention Readout, Invariance, And Statistics

This note derives the optional global attention readout, proves permutation invariance, and compares the statistics preserved by mean, max, and attention pooling.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Attention Readout In The Model Class

The released model class also supports a global attention readout. Its gate
network is:

```math
c_v
=
\phi
\left(
\overline h_vW_A+b_A
\right)
w_2
+
b_2,
```

where:

```math
W_A
\in
\mathbb{R}^{d\times d/2},
\qquad
w_2
\in
\mathbb{R}^{(d/2)\times 1}.
```

The model uses LeakyReLU for `phi`. Global attention weights are:

```math
\alpha_v
=
\frac{\exp(c_v)}
{\sum_{u=1}^{n}\exp(c_u)}.
```

The slide vector is:

```math
z^{\mathrm{attn}}
=
\sum_{v=1}^{n}
\alpha_v\overline h_v.
```

This is a weighted first moment of contextualized nodes. It differs from the
local knowledge-aware weights:

```text
pi_vu:
    source-tail weighting inside target v's selected neighborhood

alpha_v:
    updated-node weighting across the whole slide
```

The model constructor defaults to attention pooling, but the released training
script overrides that default with mean pooling. Reproducing a result requires
stating which entry point and readout configuration were used.

## Permutation Invariance

Mean and max pooling are permutation invariant:

```math
\mathcal{R}(PH^{+})
=
\mathcal{R}(H^{+})
```

for every node permutation matrix `P`.

The attention readout is also invariant because its gate is applied rowwise and
its softmax denominator sums over all nodes:

```math
\mathcal{R}_{\mathrm{attn}}(PH^{+})
=
\mathcal{R}_{\mathrm{attn}}(H^{+}).
```

Combined with permutation-equivariant graph construction and context, the
slide prediction is invariant to patch enumeration, except at exact top-k ties
with index-dependent tie breaking.

## Surviving Statistic

Mean configuration used by the released training script:

```math
z
=
\frac{1}{n}
\sum_{v=1}^{n}h_v^{+}.
```

Max option:

```math
z_r
=
\max_v[h_v^{+}]_r.
```

Attention option:

```math
z
=
\sum_v\alpha_vh_v^{+}.
```

The graph can encode pairwise interactions into each node before readout, but
the final slide statistic is still one first moment, weighted first moment, or
coordinatewise extremum of those contextualized nodes.
