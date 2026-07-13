# DTFD-MIL Double-Tier Distillation

## 1. Pseudo-Bags

DTFD divides a large WSI bag into subsets:

```math
B_i=\biguplus_{r=1}^{R}B_{ir}.
```

Each pseudo-bag receives an instance-level representation and prediction.

## 2. First Tier

For pseudo-bag `r`, an attention or classifier readout gives

```math
z_{ir}=\mathcal R_1(B_{ir}),
\qquad
q_{ir}=\mathcal H_1(z_{ir}).
```

Pseudo-bag scores identify informative subsets but are not final slide scores.

## 3. Second Tier

The pseudo-bag representations are distilled into a slide representation:

```math
z_i=\mathcal R_2(\{z_{ir},q_{ir}\}_{r=1}^{R}),
\qquad
\widehat y_i=\mathcal H_2(z_i).
```

Distillation transfers local evidence into a second aggregation level.

## 4. Explanation Boundary

A high-scoring pseudo-bag identifies a subset surviving the first tier. It does
not prove every patch in that subset contributes equally to the final score;
second-tier routing and the final head must be included.

