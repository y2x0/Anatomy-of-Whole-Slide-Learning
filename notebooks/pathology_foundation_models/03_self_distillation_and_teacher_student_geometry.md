# Self-Distillation and Teacher-Student Geometry

## 1. Teacher and Student Views

Let teacher and student outputs be centered and temperature-scaled:

```math
p_t^{(v)}=\mathrm{softmax}\left(
\frac{g_t(f_{\phi_t}(v(x))-c)}{\tau_t}
\right),
```

```math
p_s^{(v')}=\mathrm{softmax}\left(
\frac{g_s(f_{\phi_s}(v'(x)))}{\tau_s}
\right).
```

The cross-view distillation loss is

```math
\mathcal L_{\mathrm{dist}}
=-\sum_k p_{t,k}^{(v)}\log p_{s,k}^{(v')}.
```

HIPT uses hierarchical self-supervised ViT training with DINO-style teacher
student structure across visual scales.

## 2. Teacher Update

An exponential moving average teacher follows the student:

```math
\phi_t\leftarrow m\phi_t+(1-m)\phi_s.
```

The teacher is a smoothed target, not an independent source of pathology truth.

## 3. Hierarchical Distillation

At patch-to-region and region-to-slide levels,

```math
h_{r}=F_{\mathrm{region}}(\{h_j:j\in r\}),
\qquad
z=F_{\mathrm{slide}}(\{h_r\}_r).
```

Distillation across levels pressures local and global representations to be
compatible. The hierarchy preserves only relations exposed to the teacher.

## 4. Collapse and Shortcut

If all outputs become identical, the cross-view loss can be small without
retaining useful variation. Centering, sharpening, architectural asymmetry, and
multi-crop design are mechanisms against collapse, not proofs of semantic
disentanglement.

