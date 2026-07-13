# Multimodal Alignment

Multimodal foundation models align image representations with text, reports, or
other clinical modalities.

The slide object can become:

```math
\mathcal{X}_i=(z_i,t_i)
```

or:

```math
\mathcal{X}_i
=
(z_i,\mathcal{T}),
```

where
```math
\mathcal{T}
```
is a shared text embedding space.

## Image-Text Contrastive Alignment

Let:

```math
z_i
=
F_{\text{image}}(S_i),
\qquad
t_i
=
F_{\text{text}}(R_i),
```

where
```math
R_i
```
is a report, caption, diagnosis string, or generated description.

Contrastive alignment uses:

```math
p(i\mid z_i)
=
\frac{\exp(z_i^\top t_i/\tau)}
{\sum_{k}\exp(z_i^\top t_k/\tau)}.
```

Loss:

```math
\ell_i
=
-\log p(i\mid z_i).
```

Often the symmetric text-to-image loss is also used:

```math
\ell_i^{\text{sym}}
=
\ell_{\mathrm{image}\to\mathrm{text}}
+
\ell_{\mathrm{text}\to\mathrm{image}}.
```

This makes visual similarity partly semantic:

```math
z_i^\top z_k
```

is shaped by report or caption language, not only morphology.

## Prompt-Based Prediction

For zero-shot classification, create class text embeddings:

```math
t_c
=
F_{\text{text}}(\mathrm{prompt}(c)).
```

Prediction:

```math
p(y=c\mid S_i)
=
\frac{\exp(z_i^\top t_c/\tau)}
{\sum_{r}\exp(z_i^\top t_r/\tau)}.
```

The classifier is a set of text vectors:

```math
\{t_c\}_{c=1}^{C}.
```

Thus task information is injected by language rather than by a learned linear
head.

## Report-Conditioned Slide Representation

A slide-level multimodal model can use reports during pretraining:

```math
(S_i,R_i)
\mapsto
(z_i,t_i).
```

At inference, the slide embedding can be used without the report:

```math
z_i=F_{\text{image}}(S_i).
```

But the geometry of
```math
z_i
```
still reflects report supervision.

PRISM- and TITAN-style models fit this pattern: slide-level visual embeddings
are trained to interact with report or caption representations, enabling
retrieval, zero-shot classification, report generation, or linear probing.

## Cross-Modal Retrieval

Image-to-text retrieval:

```math
\mathcal{N}_K^{\text{text}}(i)
=
\mathrm{TopK}_{r}
z_i^\top t_r.
```

Text-to-image retrieval:

```math
\mathcal{N}_K^{\text{image}}(q)
=
\mathrm{TopK}_{i}
z_i^\top t_q.
```

The slide representation is useful when semantically similar slides and texts
are close:

```math
z_i^\top t_q
\quad
\text{large}
```

for the correct query
```math
q
```
.

## Dense Summary

```math
\boxed{
z_i
\approx
t_i
\quad
\text{in a shared latent geometry}
}
```

Multimodal alignment makes the slide a visual object whose coordinates are
partly defined by language.
