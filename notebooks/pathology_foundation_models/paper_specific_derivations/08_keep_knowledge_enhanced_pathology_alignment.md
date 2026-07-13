# KEEP: Knowledge-Enhanced Pathology Alignment

Reference: [KEEP](https://arxiv.org/html/2412.13126v1).

## 1. Knowledge-Conditioned Text

KEEP augments visual-language alignment with pathology knowledge. Let a text
concept be transformed into a knowledge-enriched representation:

```math
t_c'=K_\omega(t_c,\mathcal K),
```

where `K` denotes the learned knowledge-enhancement map and `K` the knowledge
source or graph.

## 2. Alignment Objective

The image and enriched text embeddings are aligned through a contrastive loss:

```math
\mathcal L
=\frac12
(\mathcal L_{I\to T'}+\mathcal L_{T'\to I}).
```

The representation preserves visual directions that are both image-text
compatible and compatible with the injected knowledge structure.

## 3. New Assumption

Knowledge enhancement changes the semantic prior:

```math
P(H\mid T')\ne P(H\mid T).
```

This can improve rare concept coverage, but incorrect or incomplete knowledge
can impose a misleading alignment direction.

## 4. Evaluation

Knowledge-enhanced zero-shot performance should be tested against plain text
prompts, paraphrases, held-out diseases, and external institutions. Higher
accuracy alone does not show that the learned explanations are medically valid.
