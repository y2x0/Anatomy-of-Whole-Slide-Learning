# Pathology Vision-Language Alignment

## 1. Shared Embedding Space

For image patch `x` and text description `t`, encoders produce normalized
vectors

```math
u=\frac{f_{\mathrm{img}}(x)}{\|f_{\mathrm{img}}(x)\|_2},
\qquad
v=\frac{f_{\mathrm{text}}(t)}{\|f_{\mathrm{text}}(t)\|_2}.
```

Similarity is

```math
s(x,t)=u^{\top}v.
```

## 2. Symmetric Contrastive Alignment

For paired batch entries, an image-to-text term is

```math
\mathcal L_{I\to T}
=-\frac1B\sum_i
\log
\frac{\exp(s_{ii}/\tau)}
{\sum_j\exp(s_{ij}/\tau)}.
```

The text-to-image term reverses the denominator; their average is the symmetric
objective used by CLIP-style pathology models such as PLIP and CONCH.

## 3. Semantic Contract

The model learns visual directions aligned with caption distribution `P(T|X)`.
It does not guarantee alignment with expert morphology outside the captions:

```math
\text{VLM similarity}
\nequivalence
\text{pathology truth probability}.
```

Prompt paraphrases, negation, stain, and institution tests are necessary for
concept-grounded claims.

## 4. WSI Extension

For slide patches, a concept prior may be

```math
M_{jc}=\mathrm{sim}(f_{\mathrm{img}}(x_j),f_{\mathrm{text}}(t_c)).
```

An aggregator then defines the slide representation. The language index makes
coordinates readable, but aggregation can still mix prevalence, attention, and
semantic similarity.

