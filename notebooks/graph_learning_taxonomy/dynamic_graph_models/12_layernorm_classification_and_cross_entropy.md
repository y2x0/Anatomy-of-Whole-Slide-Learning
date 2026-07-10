# LayerNorm, Classification, And Cross-Entropy

This note derives the post-readout normalization, linear class head, softmax probabilities, slide-level objective, and the supervision path back into graph construction.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Layer Normalization

After graph readout, the released model applies LayerNorm. For slide vector
`z`, define:

```math
\mu_z
=
\frac{1}{d}
\sum_{r=1}^{d}z_r,
```

```math
\sigma_z^2
=
\frac{1}{d}
\sum_{r=1}^{d}
\left(
z_r-\mu_z
\right)^2.
```

Then:

```math
\widetilde z_r
=
\gamma_r
\frac{z_r-\mu_z}
{\sqrt{\sigma_z^2+\varepsilon}}
+
\beta_r.
```

Thus:

```math
\widetilde z
=
\mathrm{LN}_{\gamma,\beta}(z)
\in
\mathbb{R}^{d}.
```

LayerNorm removes the pre-affine featurewise location and scale of each slide
vector. The learned `gamma` and `beta` can restore coordinate-specific scale
and offset, but the classifier receives the normalized statistic rather than
the raw pooled vector.

## Classification Head

For `C` classes, the released model computes logits:

```math
\ell
=
\widetilde zW_C+b_C
\in
\mathbb{R}^{C},
```

with:

```math
W_C
\in
\mathbb{R}^{d\times C},
\qquad
b_C
\in
\mathbb{R}^{C}.
```

Class probabilities are:

```math
p_c
=
\frac{\exp(\ell_c)}
{\sum_{a=1}^{C}\exp(\ell_a)}.
```

The model forward pass returns `ell`, not `p`. The released metric code applies
softmax afterward when computing AUC probabilities.

## Cross-Entropy Supervision

For a one-hot label:

```math
y
\in
\{0,1\}^{C},
\qquad
\sum_{c=1}^{C}y_c
=
1,
```

the per-slide cross-entropy is:

```math
\mathcal{L}_{\mathrm{CE}}
=
-\sum_{c=1}^{C}
y_c\log p_c.
```

Equivalently, if the observed class is `c_star`:

```math
\mathcal{L}_{\mathrm{CE}}
=
-\ell_{c_{\star}}
+
\log
\sum_{a=1}^{C}
\exp(\ell_a).
```

The logit gradient is:

```math
\frac{\partial\mathcal{L}_{\mathrm{CE}}}
{\partial\ell_c}
=
p_c-y_c.
```

This gradient reaches graph construction only through the selected forward
path:

```math
\mathcal{L}_{\mathrm{CE}}
\to
\ell
\to
\widetilde z
\to
z
\to
H^{+}
\to
(\Pi,R,\Omega,A).
```

There is no edge-level or patch-level label in the reported objective. The
directed topology is weakly supervised by the slide class.

## C/R/G/S Placement

```text
\mathcal{G}:
    learned directed support and relation embeddings are already constructed

\mathcal{C}:
    produces contextualized nodes H+

\mathcal{R}:
    mean in the released training script; model class also offers max or
    global attention

\mathcal{S}:
    slide-level class label through logit cross-entropy
```

## Dense Summary

The implementation-faithful tail of the pipeline is:

```math
H^{+}
\xrightarrow{\mathrm{Dropout}(0.3)}
\overline H
\xrightarrow{\mathcal{R}}
z
\xrightarrow{\mathrm{LayerNorm}}
\widetilde z
\xrightarrow{W_C,b_C}
\ell
\xrightarrow{\mathrm{CrossEntropy}}
\mathcal{L}_{\mathrm{CE}}.
```

Softmax is implicit inside cross-entropy during training and explicit only when
probabilities are needed for reporting.
