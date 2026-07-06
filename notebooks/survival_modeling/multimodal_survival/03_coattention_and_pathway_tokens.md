# Co-Attention And Pathway Tokens

Co-attention models survival as interaction between pathology tokens and
biological or genomic tokens.

Let pathology tokens be:

```math
P_i\in\mathbb{R}^{n_i\times d}.
```

Let genomic/pathway tokens be:

```math
G_i\in\mathbb{R}^{R\times d}.
```

## Co-Attention

Queries from pathways:

```math
Q=G_iW_Q.
```

Keys and values from pathology:

```math
K=P_iW_K,
\qquad
V=P_iW_V.
```

Cross-attended pathology:

```math
A_{G\leftarrow P}
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V.
```

Each pathway token retrieves relevant morphology.

The reverse direction is:

```math
A_{P\leftarrow G}
=
\operatorname{softmax}
\left(
\frac{P_iW_Q(G_iW_K)^\top}{\sqrt{d}}
\right)G_iW_V.
```

## MCAT View

MCAT-style models use co-attention to learn dense interactions between WSI
patches and genomic features.

Mathematically:

```math
(P_i,G_i)
\xrightarrow{\text{coattention}}
Z_i
\xrightarrow{\text{aggregation}}
z_i
\xrightarrow{\text{survival}}
\eta_i.
```

The interaction matrix:

```math
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)
```

is a learned alignment between modality tokens.

## SurvPath View

SurvPath tokenizes transcriptomics into biological pathways.

Let:

```math
G_i=\{g_{ir}\}_{r=1}^{R}
```

where each token corresponds to a pathway or biological function.

Fusion:

```math
Z_i
=
\operatorname{Transformer}_{\text{multi}}
([P_i;G_i]).
```

Survival readout:

```math
z_i=\mathcal{R}(Z_i),
\qquad
\eta_i=f(z_i).
```

Pathway tokens make cross-modal attention more interpretable than raw gene
vectors.

## Dense Interaction Statistic

For attention matrix $A\in\mathbb{R}^{R\times n_i}$:

```math
A_{rj}
=
\frac{\exp(q_r^\top k_j)}
{\sum_{\ell}\exp(q_r^\top k_{\ell})}.
```

The morphology retrieved by pathway $r$ is:

```math
u_{ir}=\sum_jA_{rj}v_{ij}.
```

The survival score may be:

```math
\eta_i
=
\sum_r\beta_r^\top u_{ir}.
```

This exposes the statistic:

```text
sum over pathway-conditioned morphology summaries
```

## Dense Summary

```math
\begin{aligned}
A_{rj}
&=
\operatorname*{softmax}_{j}(q_r^\top k_j),\\
u_{ir}
&=
\sum_jA_{rj}v_{ij},\\
\eta_i
&=
f(\{u_{ir},g_{ir}\}_{r=1}^{R}).
\end{aligned}
```

Co-attention survival models ask which morphology is relevant conditional on
which biological token.
