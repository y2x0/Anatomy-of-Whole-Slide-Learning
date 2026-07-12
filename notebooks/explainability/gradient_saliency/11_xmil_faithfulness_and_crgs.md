# xMIL Faithfulness And C/R/G/S

Primary anchor:

- Schramowski et al. "xMIL: Insightful Explanations for Multiple Instance
  Learning in Histopathology." 2024.
  https://arxiv.org/abs/2406.04280

## Most-Relevant-First Perturbation

Sort instance explanation scores:

```math
e_{(1)}
\ge
e_{(2)}
\ge
\cdots
\ge
e_{(K)}.
```

After dropping the first `m` patches, evaluate target probability:

```math
p_c^{(m)}
=
F_c
\left(
X_{\setminus I_m}
\right).
```

The area under perturbation curve can be approximated by:

```math
\mathrm{AUPC}
=
\frac{1}{K+1}
\sum_{m=0}^{K}
p_c^{(m)}.
```

Lower AUPC means high-ranked patches cause a faster score decrease under the
deletion protocol.

## Evaluation Boundary

Deletion changes bag size, attention normalization, and context. AUPC measures
faithfulness under this intervention, not a perturbation-independent ground
truth.

## Toy Evidence Functions

xMIL evaluates positive, negative, and contextual evidence using controlled bag
tasks. A signed evidence target can be written:

```math
\epsilon_k^{(c)}
\in
\left\{
-1,0,1
\right\}.
```

This enables direct ranking evaluation unavailable in real biomarker tasks.

## C/R/G/S Matrix

| Method | Context `C` | Readout `R` | Geometry `G` | Explanation target `S` |
|---|---|---|---|---|
| raw gradient | full differentiable model | selected input coordinates | local derivative | infinitesimal score sensitivity |
| gradient-times-input | full differentiable model | featurewise Taylor sum | signed local product | zero-reference local credit |
| Integrated Gradients | model along baseline path | path-integrated feature sum | conservative line integral | baseline-relative score difference |
| Grad-CAM | downstream network from chosen feature map | channel gradient average and ReLU | coarse activation linearization | positive class localization |
| guided backprop | modified ReLU backward graph | high-resolution positive backward signal | non-gradient visualization | activation-aligned detail |
| xMIL-LRP | MIL context and architecture-specific rules | conserved instance relevance sum | signed relevance flow | selected output decomposition |

## Surviving Statistics

Raw saliency survives first derivatives. IG survives path-integrated
derivatives. Grad-CAM survives channel-average gradients and positive activation
mass. xMIL-LRP survives rule-dependent conserved relevance.

## WSI Failure Matrix

| Failure | Affected methods | Mathematical source |
|---|---|---|
| saturation | raw gradient, gradient-times-input | endpoint derivative near zero |
| off-manifold baseline path | Integrated Gradients | interpolation through unrealizable embeddings |
| coarse localization | Grad-CAM | chosen activation grid and channel averaging |
| loss of negative evidence | guided backprop, Grad-CAM display | backward or final ReLU clipping |
| detached patch encoder | pixel-gradient methods | no computational path to pixels |
| propagation-rule dependence | LRP and xMIL-LRP | architecture-specific relevance allocation |
| context recomputation under deletion | AUPC evaluation | perturbation changes the entire bag map |

## Unified Failure Principle

```math
\text{target score}
\longrightarrow
\text{chosen layer or baseline}
\longrightarrow
\text{backward or relevance rule}
\longrightarrow
\text{spatial aggregation}
\longrightarrow
\text{rendered map}.
```

Every step changes the explanatory object. A gradient-looking heatmap is not a
single method family and cannot be interpreted without naming all five choices.
