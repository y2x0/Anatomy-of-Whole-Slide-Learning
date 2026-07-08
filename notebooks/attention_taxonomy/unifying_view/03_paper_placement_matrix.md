# Anchor Paper Placement Derivations

This note is not a broad bibliography. A paper appears here only when the
attention operator is written explicitly enough to place it in the taxonomy.

The shared template is:

```math
Q_i
\in
\mathbb{R}^{m_i\times d_q},
\qquad
K_i
\in
\mathbb{R}^{n_i\times d_k},
\qquad
R_i
\in
\mathbb{R}^{n_i\times d_r}.
```

The attention operator is:

```math
A_i
=
\sigma_{\Omega}
\left(
Q_iK_i^\top+B_i
\right),
\qquad
Z_i
=
A_iR_i.
```

Here:

```text
n_i = number of source tokens
m_i = number of target queries
B_i = optional position, graph, modality, or mask bias
sigma_Omega = normalization over the allowed source set
Z_i = statistic that survives this attention step
```

The placement question is not "which architecture name is used?" It is:

```text
what are Q, K, R, support, normalization, and surviving statistic?
```

## Compact Placement

| Paper / method | C/R/G/S placement | Target queries | Source support | Surviving statistic |
|---|---|---|---|---|
| Vaswani et al. Transformer | C | tokens | all tokens or mask | updated token states |
| Set Transformer PMA | R | learned seeds | set tokens | seed-conditioned summaries |
| ABMIL | R | one learned bag query | all instances | one weighted first moment |
| CLAM | R + S | class-specific queries | all instances; top/bottom only in training loss | class-specific weighted first moments |
| DSMIL | R + S | critical instance | all instances | max-instance branch plus critical-instance summary |
| TransMIL | C then R | WSI/class tokens | approximated dense WSI token support | contextual token states and class-token readout |
| GAT | C + G | graph nodes | graph neighbors | edge-weighted node states |
| MCAT | C/R + S | genomic/pathway tokens | pathology tokens | pathway-conditioned pathology summaries |

## Vaswani Et Al.: Self-Attention As Context

For token matrix:

```math
X
\in
\mathbb{R}^{n\times d},
```

scaled dot-product self-attention computes:

```math
Q
=
XW_Q,
\qquad
K
=
XW_K,
\qquad
R
=
XW_V.
```

The attention matrix is:

```math
A
=
\mathrm{softmax}_{\mathrm{row}}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right).
```

The output is:

```math
X'
=
AR.
```

C/R/G/S placement:

```text
C:
    attention updates token states

R:
    not specified by the attention layer itself

G:
    absent unless position embeddings or masks are added

S:
    task supervision arrives through the downstream loss
```

The surviving statistic is not one slide vector. It is a new token field:

```math
\{x_u'\}_{u=1}^{n}.
```

Each token stores a weighted first moment of value vectors conditioned on that
token's query.

## Set Transformer PMA: Learned Query Readout

Pooling by Multihead Attention uses learned seed vectors:

```math
S
\in
\mathbb{R}^{m\times d}.
```

For set tokens `H`:

```math
Q
=
SW_Q,
\qquad
K
=
HW_K,
\qquad
R
=
HW_V.
```

The readout is:

```math
Z
=
\mathrm{softmax}_{\mathrm{row}}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
R.
```

Placement:

```text
C:
    optional if preceded by set-attention blocks

R:
    seed attention reduces the set to m summaries

G:
    none unless coordinates are embedded into tokens

S:
    seed vectors are learned through the task loss
```

The surviving statistic is:

```math
Z
=
\{z_1,\ldots,z_m\},
```

a small set of learned-query weighted moments.

## ABMIL: Single-Query Readout Attention

Attention-based MIL starts from patch embeddings:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i}.
```

A standard gated attention score can be written as:

```math
e_{ij}
=
w^\top
\left[
\tanh(W_t h_{ij})
\odot
\mathrm{sigmoid}(W_s h_{ij})
\right].
```

Weights are:

```math
a_{ij}
=
\frac{\exp(e_{ij})}
{\sum_{\ell=1}^{n_i}\exp(e_{i\ell})}.
```

The bag statistic is:

```math
z_i
=
\sum_{j=1}^{n_i}a_{ij}h_{ij}.
```

Placement:

```text
C:
    identity; patches do not communicate

R:
    one attention-weighted first moment

G:
    ignored unless coordinates are included in h_ij

S:
    bag label shapes the scoring function through the loss
```

The key limitation is visible in the formula: no pairwise patch interaction
survives before pooling.

## CLAM: Class-Specific Readout Plus Training-Time Clustering

CLAM makes the attention score class-specific. For class `c`:

```math
e_{ijc}
=
w_c^\top
\left[
\tanh(W_t h_{ij})
\odot
\mathrm{sigmoid}(W_s h_{ij})
\right].
```

The class-specific attention distribution is:

```math
a_{ijc}
=
\frac{\exp(e_{ijc})}
{\sum_{\ell=1}^{n_i}\exp(e_{i\ell c})}.
```

The class-specific slide statistic is:

```math
z_{ic}
=
\sum_{j=1}^{n_i}a_{ijc}h_{ij}.
```

The top and bottom attended instances enter the instance-clustering loss:

```math
P_{ic}^{+}
=
\mathrm{TopK}_{j}(a_{ijc}),
\qquad
P_{ic}^{-}
=
\mathrm{BottomK}_{j}(a_{ijc}).
```

This affects supervision and feature geometry during training. It is not an
extra inference-time slide statistic.

Placement:

```text
C:
    identity or frozen encoder features

R:
    class-specific attention readout

G:
    generally absent at the attention operator

S:
    bag labels plus top/bottom pseudo-instance constraints
```

The survivor is:

```math
\{z_{ic}\}_{c=1}^{C},
```

not the clustering loss itself.

## DSMIL: Max Branch Plus Critical-Instance Attention

DSMIL first scores instances:

```math
s_{ijc}
=
f_c(h_{ij}).
```

For class `c`, the critical instance is:

```math
j_c^\star
=
\underset{j}{\arg\max}
\ s_{ijc}.
```

That instance becomes the query:

```math
q_{ic}
=
W_Qh_{ij_c^\star}.
```

All instances provide keys and values:

```math
k_{ij}
=
W_Kh_{ij},
\qquad
r_{ij}
=
W_V h_{ij}.
```

Critical-instance attention is:

```math
a_{ijc}
=
\frac{
\exp(q_{ic}^{\top}k_{ij})
}{
\sum_{\ell=1}^{n_i}
\exp(q_{ic}^{\top}k_{i\ell})
}.
```

The bag summary for class `c` is:

```math
z_{ic}
=
\sum_{j=1}^{n_i}a_{ijc}r_{ij}.
```

The final prediction uses both:

```text
max-instance branch:
    max_j s_ijc

bag-attention branch:
    head_c(z_ic)
```

Placement:

```text
C:
    no full patch-patch context update

R:
    critical-instance-conditioned attention readout

G:
    absent unless geometry is encoded upstream

S:
    bag supervision shapes both the max branch and attention branch
```

The failure mode is also specific: if the max branch selects the wrong critical
instance, the query used for bag attention is already biased.

## TransMIL: Context Attention Before Readout

TransMIL moves beyond pure readout MIL by contextualizing WSI tokens before
classification. A simplified attention block is:

```math
X_i
=
[x_{\mathrm{cls}},h_{i1},\ldots,h_{in_i}].
```

The self-attention part is an efficient approximation to:

```math
A_i
\approx
\mathrm{softmax}_{\mathrm{row}}
\left(
\frac{X_iW_QW_K^\top X_i^\top}{\sqrt{d_k}}
\right).
```

Values are mixed as:

```math
X_i'
=
A_iX_iW_V.
```

Position is injected by a positional encoding generator, so the contextual field
is closer to:

```math
\widetilde X_i
=
\Phi_{\mathrm{PPEG}}(X_i').
```

The readout is the final class token:

```math
z_i
=
\widetilde x_{\mathrm{cls}}.
```

Placement:

```text
C:
    efficient transformer-style WSI token context

R:
    class-token readout

G:
    PPEG-style position handling, not raw geometry alone

S:
    bag label or task loss shapes both context and readout
```

The key distinction from ABMIL is that patch states are changed before the
slide statistic is extracted.

## Graph Attention Network: Edge-Masked Context

For a graph:

```math
G_i
=
(V_i,E_i),
```

GAT computes edge scores:

```math
e_{uv}
=
\mathrm{LeakyReLU}
\left(
a^\top[W h_u\Vert W h_v]
\right),
\qquad
(u,v)\in E_i.
```

Weights are normalized over neighbors:

```math
a_{uv}
=
\frac{\exp(e_{uv})}
{\sum_{r\in\mathcal{N}(u)}\exp(e_{ur})}.
```

The node update is:

```math
h_u'
=
\sigma
\left(
\sum_{v\in\mathcal{N}(u)}
a_{uv}W h_v
\right).
```

Placement:

```text
C:
    graph message passing updates node states

R:
    separate graph or slide pooling is still needed

G:
    adjacency determines allowed support

S:
    task loss shapes edge scoring
```

The graph is not a decoration. It decides which messages are impossible before
attention scores are even computed.

## MCAT: Cross-Modal Co-Attention

MCAT uses one modality to query another. Let pathology tokens be:

```math
H_i^{p}
=
\{h_{ij}^{p}\}_{j=1}^{n_i},
```

and genomic or pathway tokens be:

```math
H_i^{g}
=
\{h_{i\ell}^{g}\}_{\ell=1}^{m_i}.
```

A pathology cross-attention map queried by genomic/pathway tokens is:

```math
Q_i
=
H_i^{g}W_Q,
\qquad
K_i
=
H_i^{p}W_K,
\qquad
R_i
=
H_i^{p}W_V.
```

The co-attention matrix is:

```math
A_i^{g\leftarrow p}
=
\mathrm{softmax}_{\mathrm{row}}
\left(
\frac{Q_iK_i^\top}{\sqrt{d_k}}
\right).
```

The pathway-conditioned pathology summaries are:

```math
Z_i^{g\leftarrow p}
=
A_i^{g\leftarrow p}R_i.
```

Placement:

```text
C:
    cross-modal context from pathology into pathway/genomic tokens

R:
    fused multimodal statistic for survival prediction

G:
    slide geometry is not the main support constraint here

S:
    survival loss shapes which cross-modal alignments are useful
```

The surviving object is not a single generic attention heatmap. It is a family
of pathway-conditioned pathology summaries:

```math
\{z_{i\ell}^{g\leftarrow p}\}_{\ell=1}^{m_i}.
```

## What This Placement Buys

The papers differ in the target query:

```text
ABMIL:
    one learned bag query

CLAM:
    class-specific queries

DSMIL:
    critical instance as query

TransMIL:
    each token and class token as queries

GAT:
    graph node as query

MCAT:
    pathway/genomic token as query
```

They also differ in what survives:

```text
weighted first moment:
    ABMIL

class-specific weighted first moments:
    CLAM

max branch plus critical-instance moment:
    DSMIL

contextual token field plus class-token readout:
    TransMIL

edge-weighted node field:
    GAT

pathway-conditioned pathology summaries:
    MCAT
```

This is the useful taxonomy: not "attention method" as a label, but the exact
query, support, normalization, value map, and surviving statistic.
