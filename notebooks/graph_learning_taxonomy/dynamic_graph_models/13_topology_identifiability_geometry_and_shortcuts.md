# Topology Identifiability, Geometry, And Shortcuts

This note studies what slide labels cannot identify about WiKG topology and how rank instability, non-spatial compatibility, encoder nuisance, and global mean injection affect edge interpretation.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## 1. Topology Is Weakly Identified

The training objective observes slide labels:

```math
\mathcal{D}
=
\{(X_b,y_b)\}_{b=1}^{B}.
```

It does not observe a target edge set:

```math
E_b^{\star}.
```

The prediction factors as:

```math
\widehat y_b
=
\mathcal{H}_{\theta}
\circ
\mathcal{R}_{\theta}
\circ
\mathcal{C}_{\theta}
\left(
H_b;
A_{\theta}(H_b),
R_{\theta}(H_b)
\right).
```

Two parameterizations can induce different graphs while producing the same
predictions on the observed slides:

```math
A_{\theta}(H_b)
\ne
A_{\theta'}(H_b),
```

```math
\widehat y_b(\theta)
=
\widehat y_b(\theta')
```

for all training slides `b`.

Therefore predictive fit does not uniquely identify a tissue-relation graph.
An edge visualization is a model-internal relational hypothesis unless it is
validated against an external biological or spatial criterion.

## 2. Hard Top-K Creates Rank-Boundary Instability

For target `v`, define the top-k margin:

```math
\gamma_v
=
s_{v,(K)}-s_{v,(K+1)}.
```

If:

```math
\gamma_v
\gg
0,
```

the selected support is locally stable.

If:

```math
\gamma_v
\approx
0,
```

a small perturbation to patch features or head-tail projections can exchange
the `K`th and `(K+1)`th sources:

```math
\mathcal{N}_K(v;H)
\ne
\mathcal{N}_K(v;H+\Delta H).
```

The downstream message can then jump from:

```math
m_v
=
\sum_{u\in\mathcal{N}_K(v;H)}
\pi_{vu}t_u
```

to a sum over a different discrete set. Attention weights can be smooth while
the support on which they are normalized is unstable.

## 6. Learned Relation Is Not Physical Geometry

Ignoring biases, WiKG scores:

```math
s_{vu}
=
\frac{
g_vM_{HT}g_u^{\top}
}{\sqrt d},
```

where:

```math
M_{HT}
=
W_HW_T^{\top}.
```

No patch coordinates appear in this score. Therefore two distant patches can
be adjacent, and two physically neighboring patches can be disconnected.

This is useful when the intended relation is morphological or task-predictive.
It fails when the claim being made requires physical adjacency, invasion-front
continuity, gland organization, or another explicitly spatial relation.

Formally:

```math
A_{\theta}[v,u]
=
1
```

does not imply:

```math
\|c_v-c_u\|_2
\text{ is small}.
```

## 7. Encoder Shortcuts Become Graph Shortcuts

The graph is inferred from patch embeddings. If the encoder preserves stain,
scanner, tissue-processing, or cohort artifacts, then head-tail similarity can
connect patches by those nuisance signals:

```math
g_v
=
g_v^{\mathrm{morph}}
+
g_v^{\mathrm{nuis}}.
```

Then:

```math
g_vM_{HT}g_u^{\top}
```

can be large because of nuisance agreement rather than biological interaction.

Weak slide supervision may reward such topology whenever nuisance structure is
correlated with the label. Learning the graph end to end does not by itself
make its edges biologically meaningful.

## 8. Pre-Graph Mean Injection Couples Every Node

The released implementation transforms each node as:

```math
\widetilde g_v
=
\frac{1}{2}
\left(
g_v+\overline g
\right),
```

where:

```math
\overline g
=
\frac{1}{n}
\sum_{u=1}^{n}g_u.
```

Therefore every head-tail score already depends on the entire slide:

```math
s_{vu}
=
s_{vu}
\left(
g_v,g_u,\overline g
\right).
```

This global coupling can stabilize slide context, but it weakens a purely
pairwise interpretation of the learned edge. Adding or removing an unrelated
patch changes the mean and can change all pair scores:

```math
\frac{\partial s_{vu}}
{\partial g_a}
\not\equiv
0
```

as a function of the model parameters, even for:

```math
a
\notin
\{u,v\}.
```

The learned relation is bag-conditioned, not solely endpoint-conditioned.
