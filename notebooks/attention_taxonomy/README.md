# Attention Taxonomy

This notebook family asks:

```text
What weighting mechanism is being learned?
```

Attention is often described as if it were one operation. In whole-slide
learning it is several different mathematical objects that share notation:

```text
attention as readout:
    patches are weighted before slide pooling

attention as context:
    patch states are updated by weighted messages

attention as cross-modal alignment:
    one modality queries another modality

attention as graph message passing:
    edges receive learned weights

attention as explanation:
    weights are interpreted as evidence, often too quickly
```

The common form is:

```math
a_{uv}
=
\sigma_{\Omega,u}
\left(
s_\theta(q_u,k_v)
\right),
\qquad
m_u
=
\sum_{v\in\mathcal{A}(u)}a_{uv}r_v.
```

Here:

```text
q_u      = query state
k_v      = key state
r_v      = value state
s_theta  = compatibility score
Omega    = optional geometry, mask, edge, or modality constraint
A(u)     = allowed source set for query u
sigma    = normalization map over A(u)
m_u      = weighted message or readout
```

The taxonomy is not based on architecture names. It is based on which part of
the attention operator changes:

```text
score:
    additive, dot product, gated, bilinear, graph-aware

normalization:
    softmax, sparsemax, entmax, top-k, masked/block sparse

neighborhood:
    all patches, local window, graph edges, cross-modal tokens, memory bank

value map:
    raw patch states, projected states, class-specific values, modality values

readout:
    pooled slide vector, CLS token, query token, graph node update, prototype
```

## Folder Map

```text
00_problem_setup.md
    attention as a learned weighting operator in C/R/G/S

softmax_attention/
    softmax geometry, temperature, entropy, normalization, and readout limits

sparse_attention/
    sparsemax, entmax, top-k, support selection, and sparse failure modes

cross_attention/
    query-key-value alignment across sets, modalities, labels, and memory

graph_attention/
    edge-restricted attention, learned adjacency, and message passing

transformer_attention/
    self-attention as a complete directed graph, multi-head subspaces,
    positional bias, CLS/PMA readouts, and transformer failure modes

interpretability_and_diagnostics/
    why attention weights are not automatically explanations; entropy,
    support, gradients, and evidence diagnostics

unifying_view/
    C/R/G/S placement, operator matrix, paper placement, checklist, and
    failure-mode matrix
```

## Core Question

For a slide with patch states:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}\in\mathbb{R}^{d},
```

an attention layer defines a data-dependent matrix:

```math
A_i(H_i,G_i,S_i)
\in
\mathbb{R}^{m_i\times n_i}.
```

Here `n_i` is the number of source tokens and `m_i` is the number of target
queries. Thus:

```text
self-attention:
    m_i = n_i

single readout attention:
    m_i = 1

class-specific, seed, or prompt attention:
    m_i = number of classes, seeds, or prompts

cross-attention:
    m_i = number of query tokens in the receiving modality
```

The central mathematical question is:

```text
which probability measure, sparse support, graph, or alignment does A_i induce,
and what statistic survives after values are summed?
```

## Anchor Papers

Only papers whose operators are explicitly used in this family are listed here.
Additional papers should be added section-by-section when their forward map is
actually derived.

- Bahdanau, Cho, and Bengio. "Neural Machine Translation by Jointly Learning to
  Align and Translate." 2014. Additive query-key attention.
- Vaswani et al. "Attention Is All You Need." 2017. Scaled dot-product
  self-attention and multi-head context attention.
  https://arxiv.org/abs/1706.03762
- Ilse, Tomczak, and Welling. "Attention-based Deep Multiple Instance
  Learning." ICML 2018. Single-query attention MIL readout.
  https://arxiv.org/abs/1802.04712
- Lu et al. "Data-efficient and weakly supervised computational pathology on
  whole-slide images." Nature Biomedical Engineering 2021. Class-specific
  CLAM attention and instance clustering supervision.
- Li et al. "Dual-stream Multiple Instance Learning Network for Whole Slide
  Image Classification with Self-supervised Contrastive Learning." CVPR 2021.
  Max-instance plus critical-instance attention MIL.
- Shao et al. "TransMIL: Transformer based Correlated Multiple Instance
  Learning for Whole Slide Image Classification." NeurIPS 2021. Nystrom
  self-attention and PPEG-style WSI position encoding.
- Chen et al. "Multimodal Co-Attention Transformer for Survival Prediction in
  Gigapixel Whole Slide Images." ICCV 2021. Genomic/pathway-query co-attention
  over pathology tokens.
- Velickovic et al. "Graph Attention Networks." ICLR 2018. Edge-masked
  neighborhood attention.
  https://arxiv.org/abs/1710.10903
- Lee et al. "Set Transformer." ICML 2019. PMA seed attention and inducing
  point set attention.
  https://arxiv.org/abs/1810.00825
- Martins and Astudillo. "From Softmax to Sparsemax." ICML 2016. Sparsemax
  projection attention.
- Peters, Niculae, and Martins. "Sparse Sequence-to-Sequence Models." ACL 2019.
  Entmax sparse attention.
- Jain and Wallace. "Attention is not Explanation." NAACL 2019. Attention-map
  interpretability caution.
- Wiegreffe and Pinter. "Attention is not not Explanation." EMNLP-IJCNLP 2019.
  Counterpoint on when attention can support explanation claims.
