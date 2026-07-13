# Multimodal Co-Attention

Multimodal WSI models often have pathology tokens and another modality:

```math
P
=
\{p_j\}_{j=1}^{n},
\qquad
M
=
\{m_r\}_{r=1}^{R}.
```

Examples include genomics, pathways, clinical variables, reports, or text
prompts.

## Pathology Queried By Modality

If modality tokens query pathology:

```math
a_{rj}
=
\mathrm{softmax}_{j}
\left(
\frac{(Q_Mm_r)^\top(K_Pp_j)}{\sqrt{d_k}}
\right),
```

then:

```math
\tilde m_r
=
\sum_{j=1}^{n}a_{rj}V_Pp_j.
```

Each modality token receives a pathology summary.

## Pathology Querying Modality

If pathology queries modality:

```math
b_{jr}
=
\mathrm{softmax}_{r}
\left(
\frac{(Q_Pp_j)^\top(K_Mm_r)}{\sqrt{d_k}}
\right),
```

then:

```math
\tilde p_j
=
\sum_{r=1}^{R}b_{jr}V_Mm_r.
```

Each patch receives modality context.

## Co-Attention Is Two Directed Maps

Co-attention is often:

```math
P\to M
\quad
\text{and}
\quad
M\to P.
```

Here the arrow denotes the query direction: `M -> P` means modality tokens
query pathology tokens, so the resulting pathology-weighted values are written
back into modality states. It does not mean that the two attention matrices
are transposes or that information is exchanged along one symmetric edge.

These are not inverses. They answer different questions:

```text
M -> P:
    which tissue supports this pathway or clinical token?

P -> M:
    which pathway or clinical token contextualizes this tissue?
```

The normalizations are over different axes: each `M -> P` row sums over the
pathology tokens, while each `P -> M` row sums over the modality tokens. A
pathology token can therefore receive high total mass from many modality
queries even when no individual modality row is concentrated.

## MCAT-Style Survival Interpretation

In multimodal survival, the final head may output a Cox risk score:

```math
\eta_i
=
w^\top z_i.
```

Even if co-attention is rich, the survival head may still collapse the fused
representation to one scalar ordering. Attention complexity does not remove the
risk-object bottleneck.

## Dense Summary

Multimodal co-attention learns conditional measures:

```math
\nu(P\mid m_r)
\qquad
\text{and}
\qquad
\nu(M\mid p_j).
```

The strength is conditional alignment. The failure mode is imported geometry:
the query modality may impose its own categories on tissue, even when those
categories are not the task-relevant pathology structure.
