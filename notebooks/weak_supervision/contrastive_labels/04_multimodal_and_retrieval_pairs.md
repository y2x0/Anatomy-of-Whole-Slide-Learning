# Multimodal And Retrieval Pairs

Multimodal pathology models often use image-text or image-memory pairs.

Let:

```math
u_i
=
f_\theta(\text{image}_i),
\qquad
v_i
=
g_\phi(\text{text}_i).
```

The paired supervision is:

```math
(u_i,v_i)
\text{ is positive}.
```

All other items in the batch are usually treated as negatives.

## Symmetric Image-Text Contrastive Loss

Image-to-text:

```math
\mathcal{L}_{I\to T}
=
-
\frac{1}{N}
\sum_i
\log
\frac{
\exp(u_i^\top v_i/\tau)
}{
\sum_k
\exp(u_i^\top v_k/\tau)
}.
```

Text-to-image:

```math
\mathcal{L}_{T\to I}
=
-
\frac{1}{N}
\sum_i
\log
\frac{
\exp(v_i^\top u_i/\tau)
}{
\sum_k
\exp(v_i^\top u_k/\tau)
}.
```

Total:

```math
\mathcal{L}
=
\frac{1}{2}
\left(
\mathcal{L}_{I\to T}
+
\mathcal{L}_{T\to I}
\right).
```

CONCH, PLIP, and related pathology vision-language models fit this supervision
shape.

## Retrieval Memory Supervision

For retrieval, the supervision may be:

```math
k\in\mathcal{N}^{+}(i)
```

if slide
```math
k
```
 is considered relevant to query slide
```math
i
```
.

A retrieval contrastive loss is:

```math
\mathcal{L}_{\mathrm{ret}}
=
-
\log
\frac{
\sum_{k\in\mathcal{N}^{+}(i)}
\exp(u_i^\top u_k/\tau)
}{
\sum_{m\in\mathcal{M}}
\exp(u_i^\top u_m/\tau)
}.
```

RetCCL-style retrieval and clustering-guided methods can be read as defining
positive neighborhoods in representation space.

## Weakness Of Text Pairs

Text may not describe every visual feature:

```math
T_i
\not\supset
\text{all morphology in slide }i.
```

The contrastive loss aligns image features to report-described content and may
suppress unlabeled visual factors.

## C/R/G/S Placement

```text
G:
    memory or text creates external relation geometry

C:
    image encoder is shaped by paired modality or retrieval neighbors

R:
    slide embedding is contrasted with text or memory embedding

S:
    paired image-text or query-neighbor relation
```

## Dense Summary

Multimodal contrastive supervision says:

```math
\text{this image}
\sim
\text{this text}.
```

That is a weak label about alignment, not a full annotation of all pathology in
the image.
