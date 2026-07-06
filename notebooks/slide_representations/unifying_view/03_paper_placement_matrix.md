# Paper Placement Matrix

This note places anchor methods into the same mathematical template:

```math
H_i
\xrightarrow{G_i,S_i}
\mathcal{X}_i
\xrightarrow{\mathcal{C}}
\widetilde{H}_i
\xrightarrow{\mathcal{R}}
z_i
\xrightarrow{\mathcal{H}}
\widehat{y}_i.
```

Here $G_i$ denotes geometry or relation structure and $S_i$ denotes task
supervision.

## Matrix

| Method | Slide Object | $G_i$ | $\mathcal{C}$ | $\mathcal{R}$ | Surviving Statistic |
|---|---|---:|---:|---:|---:|
| Deep Sets | unordered set | none | instance transform | sum or mean | transformed first moment |
| ABMIL | unordered set | none | attention score per instance | weighted sum | learned weighted first moment |
| CLAM | unordered set plus clustering constraint | none | class-specific attention and instance clustering | class-specific weighted sum | attended class evidence plus feature separation |
| Set Transformer | set with learned complete relations | complete graph | self-attention blocks | PMA or invariant pool | interaction-aware set summary |
| MambaMIL | reordered sequence | learned or constructed order | selective SSM scan | final state or pool | order-compressed trajectory |
| Patch-GCN | coordinate graph | kNN or spatial graph | graph convolution | graph readout plus survival head | contextualized patch statistic |
| HACT | hierarchical graph | cell, tissue, containment graphs | hierarchical message passing | graph-level classifier | multiscale cell-to-tissue summary |
| WiKG | learned dynamic graph | directed learned relations | knowledge-aware graph attention | global pooling | learned relation-aware graph embedding |
| HIPT | image pyramid hierarchy | nested patch and region tokens | hierarchical ViT | slide or region token | multiscale visual summary |
| PANTHER | patch distribution | mixture prototypes | prototype assignment | mixture statistic | morphology prevalence and residuals |
| Yottixel | compact patch mosaic | selected patch set | barcode or mosaic encoding | retrieval score | searchable visual summary |
| SISH | discrete latent slide code | learned index structure | self-supervised encoding | tree search and ranking | retrieval-oriented latent code |
| GigaPath | patch tokens plus coordinates | slide-level pretrained attention | slide encoder | slide embedding | pretrained WSI vector |
| CONCH | image-text latent object | caption alignment | visual-language encoder | prompt or retrieval score | text-aligned morphology embedding |
| PRISM | slide-level multimodal latent | report supervision | slide encoder and generator | prompt/probe/generation | report-shaped slide embedding |
| TITAN | multimodal whole-slide latent | slide and report/caption alignment | slide foundation encoder | retrieval, probe, or generation | general-purpose slide embedding |

## Deep Sets

The object is an unordered set or empirical measure:

```math
\mu_i
=
\frac{1}{n_i}\sum_j\delta_{h_{ij}}.
```

The model has:

```math
z_i
=
\sum_{j=1}^{n_i}\phi_\theta(h_{ij}),
\qquad
\widehat{y}_i
=
\rho_\theta(z_i).
```

Placement:

```text
G:
    none

C:
    instance transform phi

R:
    sum or mean

S:
    bag-level task loss
```

Survives:

```math
\mathbb{E}_{\mu_i}[\phi_\theta(h)].
```

## Attention MIL

Attention MIL scores each instance:

```math
s_{ij}
=
w^\top\tanh(Vh_{ij}),
```

or, in the gated version:

```math
s_{ij}
=
w^\top
\left(
\tanh(Vh_{ij})
\odot
\operatorname{sigmoid}(Uh_{ij})
\right).
```

The attention weights are:

```math
a_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{\ell=1}^{n_i}\exp(s_{i\ell})}.
```

The slide embedding is:

```math
z_i
=
\sum_{j=1}^{n_i}a_{ij}h_{ij}.
```

Placement:

```text
G:
    none

C:
    scoring without pairwise context

R:
    attention-weighted sum

S:
    bag-level cross-entropy, Cox loss, or other slide-level objective
```

Survives:

```math
\mathbb{E}_{j\sim a_i}[h_{ij}].
```

## CLAM

CLAM keeps the attention MIL backbone but makes attention class-specific and
adds instance-level clustering constraints on selected high-attention and
low-attention patches.

A simplified class-specific attention map is:

```math
a_{ij}^{(c)}
=
\operatorname*{softmax}_{j}s_c(h_{ij}).
```

Class-specific slide embeddings are:

```math
z_i^{(c)}
=
\sum_j a_{ij}^{(c)}h_{ij}.
```

The auxiliary instance constraint acts on selected patches:

```math
\ell_{\text{inst}}
=
\ell(g_\theta(h_{ij^+}),1)
+
\ell(g_\theta(h_{ij^-}),0).
```

Placement:

```text
G:
    none unless coordinates are added externally

C:
    class-specific scoring and instance-level feature shaping

R:
    class-specific attention readout

S:
    slide label plus pseudo-instance contrast from attention extremes
```

Survives:

```text
class-conditioned high-evidence patch summaries
```

## Set Transformer

Set Transformer introduces complete-graph attention among instances:

```math
\widetilde{h}_j
=
\sum_{\ell=1}^{n_i}
\operatorname*{softmax}_{\ell}
\left(
\frac{q_j^\top k_\ell}{\sqrt{d}}
\right)
v_\ell.
```

Pooling by multihead attention can be written with seed vectors $s_m$:

```math
z_m
=
\operatorname{Attn}(s_m,\widetilde{H}_i,\widetilde{H}_i).
```

Placement:

```text
G:
    complete learned relation graph

C:
    permutation-equivariant self-attention

R:
    pooling by attention or invariant pooling

S:
    task-level supervision
```

Survives:

```text
set summary after pairwise or higher-order interaction
```

## MambaMIL

MambaMIL turns the bag into a sequence:

```math
\mathcal{X}_i
=
(h_{i\sigma_i(1)},\ldots,h_{i\sigma_i(n_i)}).
```

The order can be constructed or learned:

```math
\sigma_i
=
f_\theta(H_i,C_i).
```

The selective state-space scan is:

```math
s_{t+1}
=
A_t s_t+B_t h_{i\sigma_i(t)},
\qquad
u_t
=
C_t s_t+D_t h_{i\sigma_i(t)}.
```

Placement:

```text
G:
    path induced by sequence order

C:
    selective SSM scan

R:
    final state, attention, or pooled sequence states

S:
    bag-level task loss
```

Survives:

```text
trajectory statistic determined by the scan order
```

## Patch-GCN

Patch-GCN treats patches as spatial point-cloud nodes:

```math
V_i=\{1,\ldots,n_i\},
\qquad
h_v=h_{iv},
\qquad
c_v\in\mathbb{R}^{2}.
```

Edges are coordinate-derived:

```math
(u,v)\in E_i
\quad\Longleftrightarrow\quad
u\in\operatorname{kNN}_k(c_v).
```

Message passing gives:

```math
H_i^{(L)}
=
\operatorname{GCN}_\theta(H_i,A_i).
```

Readout and survival head:

```math
z_i
=
\mathcal{R}(H_i^{(L)},A_i),
\qquad
r_i
=
\mathcal{H}(z_i).
```

Placement:

```text
G:
    spatial kNN graph from patch coordinates

C:
    graph convolution

R:
    graph-level aggregation

S:
    survival objective, commonly Cox-style risk
```

Survives:

```text
first or weighted moment of spatially contextualized patch states
```

## HACT

HACT represents tissue with a hierarchy:

```math
V
=
V_{\text{cell}}
\cup
V_{\text{tissue}}.
```

Edges include:

```math
E
=
E_{\text{cell-cell}}
\cup
E_{\text{tissue-tissue}}
\cup
E_{\text{cell-tissue}}.
```

Cell states and tissue states are contextualized at different scales:

```math
H_{\text{cell}}'
=
\mathcal{C}_{\text{cell}}(H_{\text{cell}},A_{\text{cell}}),
```

```math
H_{\text{tissue}}'
=
\mathcal{C}_{\text{tissue}}
(H_{\text{tissue}},A_{\text{tissue}},H_{\text{cell}}').
```

Placement:

```text
G:
    hierarchical cell-to-tissue graph

C:
    within-level and cross-level message passing

R:
    graph-level classifier

S:
    region or slide label
```

Survives:

```text
cell organization summarized through tissue-level structure
```

## WiKG

WiKG-style dynamic graph methods make relations learned and directed:

```math
A_{uv}
=
g_\theta(h_u,h_v,c_u,c_v).
```

With edge embeddings:

```math
e_{uv}
=
\eta_\theta(h_u,h_v,c_u,c_v).
```

Knowledge-aware attention updates head nodes from neighbors and edges:

```math
h_v'
=
\phi_\theta
\left(
h_v,
\sum_{u\in\mathcal{N}(v)}
\alpha_{uv}\psi_\theta(h_u,e_{uv})
\right).
```

Placement:

```text
G:
    learned directed relation graph

C:
    graph attention using node and edge information

R:
    global pooling

S:
    slide-level classification
```

Survives:

```text
task-learned relational embedding, not necessarily physical adjacency
```

## HIPT

HIPT represents a gigapixel slide through nested visual tokens:

```math
\text{small tokens}
\to
\text{patch tokens}
\to
\text{region tokens}
\to
\text{slide representation}.
```

A simplified hierarchy is:

```math
h_{ij}^{(0)}
=
E_0(x_{ij}^{(0)}),
```

```math
h_{ir}^{(1)}
=
E_1(\{h_{ij}^{(0)}:j\in R_r\}),
```

```math
z_i
=
E_2(\{h_{ir}^{(1)}\}_{r=1}^{R_i}).
```

Placement:

```text
G:
    image pyramid and region membership

C:
    transformer context within and across scales

R:
    hierarchical token readout

S:
    self-supervised pretraining plus downstream task supervision
```

Survives:

```text
multiscale visual summary shaped by the image pyramid
```

## PANTHER

PANTHER treats recurring morphology as mixture components.

Global prototypes:

```math
c_1,\ldots,c_M.
```

Patch responsibilities:

```math
\gamma_{ijm}
=
p(m\mid h_{ij}).
```

Slide-level prototype prevalence:

```math
\widehat\pi_{im}
=
\frac{1}{n_i}
\sum_j\gamma_{ijm}.
```

Placement:

```text
G:
    none unless coordinates are added later

C:
    assignment to morphology prototypes

R:
    mixture weights and optional residual statistics

S:
    unsupervised representation learning, then downstream task head
```

Survives:

```text
distribution of recurring morphology prototypes
```

## Yottixel And SISH

Retrieval systems represent a WSI by a compact searchable object.

Generic form:

```math
z_i
=
T(S_i),
\qquad
\mathcal{N}_K(i)
=
\operatorname*{TopK}_{k}
\operatorname{sim}(z_i,z_k).
```

For patch-mosaic approaches, the representation is a selected subset:

```math
B_i
=
\{b_{i1},\ldots,b_{iM}\}.
```

For discrete latent search, the representation may be:

```math
b_i\in\{0,1\}^{M}
```

or an indexed code.

Placement:

```text
G:
    archive-neighborhood graph

C:
    retrieval over memory

R:
    similarity ranking or neighbor-weighted prediction

S:
    slide labels, self-supervised encoding, or archive metadata
```

Survives:

```text
the part of the slide needed to retrieve similar cases under the chosen metric
```

## GigaPath

GigaPath-style whole-slide foundation models separate tile and slide encoders:

```math
h_{ij}
=
E_{\text{tile}}(x_{ij}),
```

```math
z_i
=
F_{\text{slide}}
\left(
\{h_{ij},c_{ij}\}_{j=1}^{n_i}
\right).
```

Placement:

```text
G:
    tile coordinates and slide-scale token structure

C:
    pretrained slide encoder

R:
    slide embedding

S:
    large-scale pretraining plus downstream adaptation or probing
```

Survives:

```text
pretrained whole-slide latent coordinate
```

## UNI And Virchow

UNI and Virchow are primarily patch-level foundation encoders in the slide
representation pipeline:

```math
h_{ij}
=
E_{\text{FM}}(x_{ij}).
```

The downstream slide representation still requires:

```math
z_i
=
\mathcal{R}_\theta(\{h_{ij}\}_{j=1}^{n_i}).
```

Placement:

```text
G:
    none at patch encoding unless coordinates are added downstream

C:
    pretrained local image representation

R:
    task-specific MIL, graph, transformer, distribution, or slide encoder

S:
    self-supervised pretraining, then downstream slide labels
```

Survives:

```text
local morphology encoded in pretrained patch space, not a complete slide object by itself
```

## CONCH, PRISM, And TITAN

Visual-language and multimodal slide models align image embeddings with text,
captions, or reports.

Image and text embeddings:

```math
z_i
=
F_{\text{image}}(S_i),
\qquad
t_i
=
F_{\text{text}}(R_i).
```

Contrastive alignment:

```math
\ell_i
=
-\log
\frac{\exp(z_i^\top t_i/\tau)}
{\sum_k\exp(z_i^\top t_k/\tau)}.
```

Prompt prediction:

```math
p(y=c\mid S_i)
=
\frac{\exp(z_i^\top t_c/\tau)}
{\sum_r\exp(z_i^\top t_r/\tau)}.
```

Placement:

```text
G:
    visual tokens plus language/report latent space

C:
    multimodal alignment or generative conditioning

R:
    slide embedding, prompt score, retrieval score, or generated report

S:
    paired image-text/report data, captions, and downstream task labels
```

Survives:

```text
visual morphology organized by language or report semantics
```

## Diagnostic Use

For any new paper, fill in:

```text
1. What is the slide object?
2. What is the context graph or order?
3. What states are contextualized?
4. What readout statistic survives?
5. What supervision shapes the representation?
6. What failure mode follows from that structure?
```

If two papers have the same answers, they are mathematically close even if the
implementation names differ.
