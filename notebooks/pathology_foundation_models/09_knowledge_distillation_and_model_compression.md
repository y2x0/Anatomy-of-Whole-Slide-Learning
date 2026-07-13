# Knowledge Distillation and Model Compression

## 1. Teacher Targets

Let a large pathology foundation teacher output `p_t` and a student output
`p_s`. Distillation minimizes

```math
\mathcal L_{\mathrm{KD}}
=\tau^2D_{\mathrm{KL}}
\left(
p_t^{(\tau)}\,\|\,p_s^{(\tau)}
\right),
```

where temperature `tau` softens the target distribution.

## 2. Feature Distillation

If teacher and student features have compatible dimensions,

```math
\mathcal L_{\mathrm{feat}}
=\|g_s(h_s)-g_t(h_t)\|_2^2.
```

For mismatched dimensions, a projection head defines the comparison space. The
student preserves the teacher only under the chosen projection and data views.

## 3. Unified Knowledge Distillation

Pathology distillation can combine output, feature, relation, or multimodal
teacher signals:

```math
\mathcal L
=\lambda_y\mathcal L_y
+\lambda_h\mathcal L_{\mathrm{feat}}
+\lambda_r\mathcal L_{\mathrm{rel}}
+\lambda_m\mathcal L_{\mathrm{modal}}.
```

Each term creates a different invariance. Matching teacher logits does not
guarantee matching retrieval neighborhoods or spatial slide structure.

## 4. Compression Failure

Student compression can remove rare pathology, long-range geometry, or
uncertainty while preserving average teacher accuracy. Evaluate subgroup,
nearest-neighbor, spatial, and calibration behavior separately.

