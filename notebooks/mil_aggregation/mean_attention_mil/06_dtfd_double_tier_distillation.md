# DTFD-MIL Double-Tier Distillation

Source:

Zhang et al., DTFD-MIL: Double-Tier Feature Distillation Multiple Instance
Learning for Histopathology Whole Slide Image Classification, CVPR 2022.
https://arxiv.org/abs/2203.12081

The full derivation is in
notebooks/mil_aggregation/hierarchical_mil/03_dtfd_pseudo_bags_and_two_tier_distillation.md.

## 1. Pseudo-Bags

DTFD divides a large WSI bag into subsets:

```math
B_i=\biguplus_{r=1}^{R}B_{ir}.
```

Each pseudo-bag inherits the parent slide label. That is a training device, not
an assertion that every pseudo-bag contains a positive instance.

## 2. First Tier

For each pseudo-bag, Tier 1 applies an attention-based MIL model:

```math
z_{ir}=\mathcal R_1(B_{ir}),
\qquad
q_{ir}=\mathcal H_1(z_{ir}).
```

The inherited slide label supervises this auxiliary Tier-1 loss. A positive
parent slide can therefore create a noisy positive pseudo-bag when its subset
contains no positive patch.

## 3. Second Tier

DTFD does not simply pass every Tier-1 representation to Tier 2. It first
distills one feature d_ir from each pseudo-bag using a paper-defined rule
(MaxS, MaxMinS, MAS, or AFS), then applies a second attention-based MIL model:

```math
z_i=\mathcal R_2(\{d_{ir}\}_{r=1}^{R}),
\qquad
\widehat y_i=\mathcal H_2(z_i).
```

The second-tier output is the parent-slide prediction. A high Tier-1 score or
attention weight is therefore only a candidate-generation signal, not a final
patch attribution.

## 4. Explanation Boundary

A high-scoring pseudo-bag identifies a subset surviving the first tier. It does
not prove every patch in that subset contributes equally to the final score;
second-tier routing and the final head must be included.
