# Hybrid MIL C/R/G/S Design Matrix And Open Axes

## 1. Unified factorization

A hybrid model can be represented as

```math
\mathcal H_{\omega}
\circ
\mathcal R_{\phi}
\circ
\mathcal C_{\theta}
\left(
H_i,G_i,S_i
\right),
```

where S_i denotes any intermediate supervision or training-time target. The
notation keeps the forward geometry G separate from the loss channel S.

## 2. Mapped paper placements

`text
Paper       Context C                       Readout R                Geometry G
--------------------------------------------------------------------------------
DSMIL       critical-instance and          max plus instance         feature
            bag-level attention            evidence head             similarity
TransMIL    Nystrom self-attention         transformer bag head      PPEG/grid
Patch-GCN   coordinate graph convolution   slide attention           spatial kNN
WiKG        dynamic knowledge graph        mean or max interaction   feature graph
HACT        cell and tissue GNNs           assignment sum and sum    cell/tissue
                                            concatenation              hierarchy
HIPT        nested transformer blocks      [CLS] at each scale       image pyramid
DTFD-MIL    Tier-1 and Tier-2 attention    selection or attention     random
                                            at two levels              pseudo-bags
MambaMIL    Mamba or SR-Mamba scan         scalar attention mean      chosen order
`

The table is a decomposition of the forward maps, not a claim that the papers
use identical losses or training protocols.

## 3. Surviving-statistic matrix

`text
Composition                         First information made available
--------------------------------------------------------------------------------
graph then attention                relationally contextualized weighted moment
transformer then [CLS]              learned nonlinear token summary
hierarchy then attention            product of local and coarse routing
state space then mean               order-conditioned first moment
dynamic graph then max              extreme of learned relational features
pseudo-bag selection then attention selected candidate statistics plus coarse moment
`

The first row of this table is not equivalent to attention over raw patches:
context changes the feature being weighted.

## 4. Open design axes

The mapped literature leaves combinations that are mathematically well-defined
but underexplored:

```math
\begin{array}{c|c|c|c}
\text{context} & \text{geometry} & \text{readout} & \text{question}\\
\hline
\text{state-space} & \text{coordinate graph} & \text{prototype} &
\text{can order and space be calibrated?}\\
\text{graph} & \text{hierarchy} & \text{hazard vector} &
\text{which scale carries time-specific risk?}\\
\text{transformer} & \text{distribution} & \text{top-k} &
\text{does selection preserve rare morphology?}\\
\text{attention} & \text{learned routing} & \text{MMD summary} &
\text{can routing and distribution be identified separately?}
\end{array}
```

An empty cell is not automatically a contribution. It is a design hypothesis
that needs a statistical reason and a failure-mode analysis.

## 5. Minimal hybrid specification

A new method should provide

```math
\left(
C_\theta,
R_\phi,
G,
S,
\mathcal L
\right)
```

and answer:

`text
- what object the slide is before context;
- which interactions are available before compression;
- which statistic reaches the task head;
- what symmetry is preserved;
- what failure mode is traded for the claimed gain;
- which paper-specific component is genuinely new.
`

## 6. Final principle

Hybrid MIL is not the union of every module. It is an ordered composition whose
noncommuting operators define the inductive bias.

The useful comparison is therefore

```math
\left[
\text{object}
\;\middle|\;
\text{context}
\;\middle|\;
\text{readout}
\;\middle|\;
\text{surviving statistic}
\;\middle|\;
\text{supervision}
\right].
```

That bracket is the mathematical signature of a whole-slide learner.
`
