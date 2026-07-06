# Adjacency As Context Operator

The cleanest way to unify set, sequence, and graph representations is to ask:

```text
Which instances are allowed to exchange information before readout?
```

Let $G_i=(V_i,E_i)$ be an interaction graph over patch indices. A generic
context layer can be written:

```math
m_j
=
\mathrm{AGG}_{k\in\mathcal{N}_{G_i}(j)}
\psi_\theta(h_j,h_k,e_{jk}),
```

```math
\widetilde{h}_j
=
\phi_\theta(h_j,m_j).
```

Different representation families choose different neighborhoods
$\mathcal{N}_{G_i}(j)$.

## Empty Graph

For no cross-instance context:

```math
E_i=\varnothing,
\qquad
\widetilde{h}_j=\phi_\theta(h_j).
```

Readout is then responsible for all bag interaction:

```math
z_i
=
\mathcal{R}(\{\phi_\theta(h_{ij})\}_{j=1}^{n_i}).
```

Mean MIL, max MIL, and many Deep Sets MIL models live here.

Failure mode:

```text
no explicit pairwise or spatial interaction before aggregation
```

## Complete Graph

Full self-attention uses:

```math
\mathcal{N}_{K_n}(j)=\{1,\ldots,n\}.
```

With:

```math
q_j=W_Qh_j,
\qquad
k_\ell=W_Kh_\ell,
\qquad
v_\ell=W_Vh_\ell,
```

the update is:

```math
\widetilde{h}_j
=
\sum_{\ell=1}^{n}
\alpha_{j\ell}v_\ell,
```

```math
\alpha_{j\ell}
=
\mathrm{softmax}_{\ell}
\left(
\frac{q_j^\top k_\ell}{\sqrt{d}}
\right).
```

This is message passing on the complete graph $K_n$ with learned directed edge
weights $\alpha_{j\ell}$.

Set Transformer uses this idea while preserving permutation equivariance before
the invariant pooling stage.

Failure mode:

```text
all-pairs interaction is expressive but expensive and can ignore physical locality
```

## Path Graph

RNNs and state-space models use the ordered path:

```math
\sigma(1)\to\sigma(2)\to\cdots\to\sigma(n).
```

A state-space recurrence:

```math
s_{t+1}=A_t s_t+B_t h_{\sigma(t)},
\qquad
u_t=C_t s_t+D_t h_{\sigma(t)}
```

passes information through this path.

The path graph analogy applies to scans, recurrences, and local sequence
operators. It does not apply to full self-attention over a sequence. Full
self-attention is still complete-graph message passing, with sequence position
or coordinates injected as features.

Failure mode:

```text
sequence locality may not match tissue locality
```

## Spatial Graph

Patch-GCN-style models choose edges from coordinates:

```math
(u,v)\in E_i
\quad\Longleftrightarrow\quad
u\in\mathrm{kNN}_k(c_v).
```

or:

```math
(u,v)\in E_i
\quad\Longleftrightarrow\quad
\|c_u-c_v\|\le r.
```

Then $L$ message-passing layers produce an $L$-hop contextualization:

```math
h_v^{(L)}
=
F_\theta
\left(
\{h_u^{(0)}:d_{G_i}(u,v)\le L\}
\right).
```

Failure mode:

```text
wrong adjacency means wrong context
```

## Learned Dynamic Graph

A learned graph chooses neighbors from features and possibly coordinates:

```math
A_{uv}
=
\mathrm{softmax}_{u}
g_\theta(h_v,h_u,c_v,c_u).
```

This makes the topology part of the model.

Failure mode:

```text
with only slide-level labels, learned edges may encode shortcuts instead of tissue relations
```

## Hierarchical Graph

A hierarchy chooses several node sets:

```math
V
=
V_{\text{cell}}
\cup
V_{\text{region}}
\cup
V_{\text{slide}}.
```

Edges include within-level and cross-level relations:

```math
E
=
E_{\text{cell-cell}}
\cup
E_{\text{region-region}}
\cup
E_{\text{cell-region}}.
```

This turns context into multiscale message passing.

Failure mode:

```text
errors in segmentation or region assignment propagate across scales
```

## Distribution Statistic

A distribution representation may have no adjacency at all. It chooses a measure:

```math
\mu_i
=
\frac{1}{n_i}\sum_j\delta_{h_{ij}}
```

and a statistic:

```math
z_i=T(\mu_i).
```

The context is implicit in the statistic $T$. For example, prototype assignment
uses a learned morphology codebook:

```math
q_m(h)
=
\mathrm{Assign}(h,c_m).
```

Failure mode:

```text
distribution shape can survive while spatial arrangement disappears
```

## Memory Adjacency

Retrieval builds a graph between the query slide and database items:

```math
(i,k)\in E_{\text{memory}}
\quad
\Longleftrightarrow
\quad
k\in\mathcal{N}_K(i).
```

The neighbors are not tissue neighbors. They are archive neighbors under a
similarity metric.

Failure mode:

```text
nearest cases may be nearest by nuisance rather than biology
```

## Foundation Geometry

Foundation models define context before downstream training through a pretrained
map:

```math
F_{\text{FM}}:S_i\mapsto z_i.
```

The induced neighbor relation is:

```math
k\in\mathcal{N}_K(i)
\quad
\Longleftrightarrow
\quad
z_k
\text{ is close to }
z_i
\text{ under }
d_{\text{FM}}.
```

Failure mode:

```text
pretraining geometry may not match the downstream clinical question
```

## Equivalence Table

| Interaction Structure | Model Family | Context Geometry |
|---|---|---|
| $E=\varnothing$ | mean MIL, max MIL, simple Deep Sets | none before readout |
| $E=K_n$ | set attention, Set Transformer | all-pairs learned relations |
| path from $\sigma$ | RNN, SSM, Mamba-style scan | order-local context |
| kNN or radius graph | Patch-GCN-style GNN | spatially local context |
| learned dynamic graph | WiKG-style graph MIL | task-learned relational context |
| typed hierarchy | HACT-style graph | multiscale tissue context |
| parent map $\pi$ | HIPT-style hierarchy | compositional scale context |
| statistic $T(\mu)$ | PANTHER or histogram-style distribution | morphology distribution context |
| memory graph | Yottixel/SISH-style retrieval | archive-neighborhood context |
| pretrained geometry | UNI, Virchow, GigaPath, CONCH, PRISM, TITAN | inherited latent context |

## The Design Question

Every context operator answers:

```text
What should count as a neighbor?
```

Every readout answers:

```text
After neighbors have exchanged information, what statistic of the contextualized slide survives?
```
