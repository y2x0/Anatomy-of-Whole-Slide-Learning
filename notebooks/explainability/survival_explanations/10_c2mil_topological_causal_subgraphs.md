# C2MIL Topological Causal Subgraphs

## 1. Differentiable Subgraph Mask

For graph nodes or patches, C2MIL samples a Bernoulli causal mask:

```math
m_{ij}\sim\mathrm{Bernoulli}(p_{ij}),
\qquad
C_i=G_i\odot m_i,
\qquad
S_i=G_i\odot(1-m_i).
```

A straight-through estimator allows hard selection in the forward pass while
passing a differentiable approximation during training.

## 2. Survival Fidelity

The selected subgraph should preserve predictive information:

```math
\mathcal L_{\mathrm{fid}}
=d\left(
F(G_i),F(C_i)
\right).
```

The complementary subgraph should be less predictive under the paper's causal
contrastive and ratio objectives. The mask is an explanation of the learned
survival computation under this training criterion.

## 3. Minimality

A causal-ratio penalty discourages selecting the whole graph:

```math
\mathcal L_{\mathrm{ratio}}
=\lambda\frac{|C_i|}{|G_i|}.
```

Minimality trades against fidelity. A tiny mask may be sparse but unstable or
omit diffuse prognostic morphology.

## 4. Attention Heatmaps

C2MIL visualizes causal-subgraph probabilities and attention maps. The causal
mask is closer to a graph intervention hypothesis than ordinary attention, but
its causal interpretation still depends on the structural model and invariance
assumptions.

