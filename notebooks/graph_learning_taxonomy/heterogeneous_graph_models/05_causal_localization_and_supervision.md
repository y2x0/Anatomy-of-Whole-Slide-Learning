# Causal Localization And Supervision

Chan et al. train the WSI graph model with slide-level labels and add a
leave-one-node localization score.

## Slide-Level Prediction

After PL pooling:

```math
z_i
=
\mathcal{R}_{\mathrm{PL}}
\left(
\widetilde H_i,
\tau_i
\right).
```

The classifier outputs:

```math
\widehat p_i
=
\mathrm{softmax}
\left(
z_i W_{\mathrm{cls}}
+
b_{\mathrm{cls}}
\right).
```

For `K_cls` classes:

```math
y_i
\in
\{0,1\}^{K_{\mathrm{cls}}}.
```

The classification loss is:

```math
\mathcal{L}_i
=
-
\sum_{c=1}^{K_{\mathrm{cls}}}
y_{ic}
\log
\widehat p_{ic}.
```

The paper uses this for classification-type tasks, including:

```text
TCGA-COAD:
    cancer staging and normal/tumor classification

TCGA-BRCA:
    cancer staging and normal/tumor classification

TCGA-ESCA:
    cancer typing
```

## Causal Contribution Score

The paper uses a leave-one-node score inspired by Granger causality.

Let:

```math
M_\theta(G_i)
```

be the model prediction on the full graph, and:

```math
M_\theta(G_i\setminus\{v\})
```

be the prediction after removing node `v`.

In these notes:

```math
G_i\setminus\{v\}
```

means the fixed constructed graph with node `v` and its incident edges removed.
It does not mean rebuilding feature-kNN support or recomputing edge attributes
after deletion, unless explicitly stated by an implementation.

The paper writes:

```math
\Delta_{iv}
=
\mathcal{L}
\left(
y_i,
M_\theta(G_i)
\right)
-
\mathcal{L}
\left(
y_i,
M_\theta(G_i\setminus\{v\})
\right).
```

This is a node-removal perturbation score.

## Sign Convention

If removing node `v` makes prediction worse, then:

```math
\mathcal{L}
\left(
y_i,
M_\theta(G_i\setminus\{v\})
\right)
>
\mathcal{L}
\left(
y_i,
M_\theta(G_i)
\right).
```

Under the paper's displayed formula:

```math
\Delta_{iv}
<
0.
```

Therefore a positive helpful-node importance convention is:

```math
I_{iv}^{\mathrm{help}}
=
\mathcal{L}
\left(
y_i,
M_\theta(G_i\setminus\{v\})
\right)
-
\mathcal{L}
\left(
y_i,
M_\theta(G_i)
\right).
```

Equivalently:

```math
I_{iv}^{\mathrm{help}}
=
-\Delta_{iv}.
```

The paper's raw score is the loss contrast `Delta`. The quantity
`I_help` is an interpretive plotting convention for heatmaps where larger means
more helpful.

## What This Localizes

The score localizes patch nodes:

```math
v
\leftrightarrow
x_{iv}.
```

It does not directly localize nuclei, because nuclei only provide pseudo-labels
for patch nodes.

The heatmap therefore lives at patch resolution:

```math
v
\mapsto
I_{iv}^{\mathrm{help}}.
```

## Causality Caveat

The method removes graph nodes and measures prediction change. This is stronger
than a pure attention heatmap, but it is not clinical causality in the
interventional biological sense.

Removing a node changes:

```text
the node set
incoming messages
outgoing messages
PL pooling composition
possibly type-wise counts
```

So:

```math
\Delta_{iv}
```

is a graph-model perturbation effect, not proof that the corresponding tissue
region causes disease.

## C/R/G/S Placement

```text
\mathcal{G}:
    localization perturbs nodes in the constructed graph

\mathcal{C}:
    HEAT recomputes context on G without node v

\mathcal{R}:
    PL pooling changes because one node is removed from its type cluster

\mathcal{S}:
    true slide label y_i is used in the perturbation loss contrast
```
