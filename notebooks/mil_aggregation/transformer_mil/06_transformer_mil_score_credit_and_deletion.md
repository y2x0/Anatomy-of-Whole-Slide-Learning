# Transformer MIL Score Credit and Deletion

## 1. Scope and central question

This note separates four objects that are often placed under the single word
explanation:

1. attention routing;
2. local gradient or saliency;
3. path-based credit through the computation graph;
4. prediction change after an intervention.

The central question is:

> What does a transformer-MIL patch score actually explain?

For a slide prediction, the answer depends on the target scalar, the
intervention baseline, the treatment of positions and padding, and whether the
full model is recomputed after a patch is altered.

The pathology anchors are:

- Ilse et al., Attention-based Deep Multiple Instance Learning:
  https://arxiv.org/abs/1802.04712
- Lu et al., CLAM:
  https://arxiv.org/abs/2004.09666
- Amores et al., Additive MIL:
  https://arxiv.org/abs/2206.01794
- Schreiber et al., xMIL:
  https://arxiv.org/html/2406.04280v1

The general attribution anchors are:

- Jain and Wallace, Attention is not Explanation:
  https://arxiv.org/abs/1902.10186
- Wiegreffe and Pinter, Attention is not not Explanation:
  https://arxiv.org/abs/1908.04626
- Sundararajan et al., Integrated Gradients:
  https://arxiv.org/abs/1703.01365
- Lundberg and Lee, SHAP:
  https://arxiv.org/abs/1705.07874

The transformer-specific context is TransMIL:

https://arxiv.org/abs/2106.00908

The goal is not to reject attention maps. It is to state the claim each score
supports and the additional experiment required for stronger claims.

## 2. Slide function and target choice

Let slide i contain n_i patch features and an optional coordinate map:

```math
H_i
=
\begin{bmatrix}
h_{i1}^{\mathsf T}\\
\vdots\\
h_{in_i}^{\mathsf T}
\end{bmatrix}
\in\mathbb R^{n_i\times d},
\qquad
C_i=(c_{i1},\ldots,c_{in_i}).
```

Let the trained model be

```math
f_\theta(H_i,C_i)
\in\mathbb R^C.
```

An explanation must specify a scalar target. For class c, common choices are

```math
F_i^{(c)}
=
\left[f_\theta(H_i,C_i)\right]_c,
```

```math
F_i^{(c,\mathrm{prob})}
=
\mathrm{softmax}\!\left(f_\theta(H_i,C_i)\right)_c,
```

or a log-odds contrast

```math
F_i^{(c,c')}
=
\left[f_\theta(H_i,C_i)\right]_c
-
\left[f_\theta(H_i,C_i)\right]_{c'}.
```

For survival, the target may be a risk score, a discrete hazard at horizon k,
or a survival probability:

```math
F_i^{(\mathrm{risk})}=\eta_i,
\qquad
F_i^{(\mathrm{haz},k)}=h_i(\tau_k),
\qquad
F_i^{(\mathrm{surv},k)}=S_i(\tau_k).
```

The same patch can have positive credit for one target and negative credit for
another. A class-agnostic heatmap is not a well-defined object until the target
is chosen.

## 3. Transformer computation graph

For a class-token transformer, let the initial sequence be

```math
Z_i^{(0)}
=
\begin{bmatrix}
q^{(0)\mathsf T}\\
H_i
\end{bmatrix}.
```

One layer produces

```math
Z_i^{(\ell+1)}
=
\mathcal B_\ell\!\left(
\mathcal C_\ell\!\left(Z_i^{(\ell)},C_i\right)
\right),
```

where C is the context operator and B includes attention, residual paths,
normalization, and MLP transformations. For TransMIL, the context path also
contains square completion, Nyström attention, and PPEG.

The final scalar is

```math
F_i
=
\psi_\theta\!\left(Z_{i,0}^{(L)}\right).
```

The computational graph contains edges from every patch to:

- its own value projection;
- its key and query projections;
- the softmax denominator of every affected query row;
- other patch rows through self-attention;
- PPEG neighbors when a raster is used;
- the class row;
- residual and normalization paths.

Any score that uses only one attention matrix observes one slice of this graph.

## 4. Attention routing score

For one head and layer ell, define

```math
Q_i^{(\ell)}=Z_i^{(\ell)}W_Q^{(\ell)},
\qquad
K_i^{(\ell)}=Z_i^{(\ell)}W_K^{(\ell)},
\qquad
V_i^{(\ell)}=Z_i^{(\ell)}W_V^{(\ell)}.
```

The attention matrix is

```math
A_i^{(\ell)}
=
\mathrm{softmax}_{\mathrm{row}}\!\left(
\frac{Q_i^{(\ell)}K_i^{(\ell)\mathsf T}}
\sqrt{d_h}
\right).
```

The final class-to-patch routing score is

```math
a_{ij}^{(\ell)}
=
\left[A_i^{(\ell)}\right]_{0j}.
```

It is a nonnegative normalized coefficient for one layer and one head. For
multihead attention, there is one row per head:

```math
a_{ij}^{(\ell,h)}
=
\left[A_i^{(\ell,h)}\right]_{0j}.
```

An aggregation across heads may be a mean, maximum, norm, or learned
projection. These aggregations answer different questions:

```math
\bar a_{ij}
=
\frac{1}{H}\sum_{h=1}^{H}a_{ij}^{(h)},
```

```math
a_{ij}^{\max}
=
\max_h a_{ij}^{(h)},
```

```math
a_{ij}^{\ell_2}
=
\left(\sum_{h=1}^{H}\left(a_{ij}^{(h)}\right)^2\right)^{1/2}.
```

No aggregation is intrinsically the correct explanation. It must be fixed
before comparing slides or methods.

## 5. Why attention is not complete credit

The class-row attention output for one head is

```math
o_{i0}
=
\sum_{j=0}^{n_i}a_{ij}v_{ij}.
```

Holding the scores and values fixed, a patch's direct value coefficient is a_ij.
But the actual differential is

```math
do_{i0}
=
\sum_{j=0}^{n_i}
\left[
(da_{ij})v_{ij}
+
a_{ij}\,dv_{ij}
\right].
```

The score differential is coupled through softmax:

```math
da_{ij}
=
a_{ij}
\left(
d\ell_{ij}
-
\sum_{r=0}^{n_i}a_{ir}\,d\ell_{ir}
\right),
```

where ell is the class-row logit. One patch can therefore change every class
attention coefficient by changing one denominator.

The value differential contains the value projection:

```math
dv_{ij}
=
W_V\,dh_{ij}.
```

The logit differential contains both query and key paths:

```math
d\ell_{ij}
=
\frac{1}{\sqrt{d_h}}
\left[
(dq_i)^{\mathsf T}k_{ij}
+
q_i^{\mathsf T}(dk_{ij})
\right].
```

Because the class query can itself depend on the complete preceding sequence,
the patch can affect its own score, other scores, and the value content
simultaneously.

Thus

```math
\mathrm{attention\ weight}
\ne
\mathrm{prediction\ credit}
```

unless the architecture is restricted so that scores and values are fixed
features and the output is a linear weighted sum.

## 6. Local gradient and saliency

For target F_i and patch feature h_ij, the local gradient is

```math
g_{ij}
=
\frac{\partial F_i}{\partial h_{ij}}
\in\mathbb R^d.
```

Common scalarizations are

```math
s_{ij}^{\mathrm{grad}}
=
\left\|g_{ij}\right\|_2,
```

```math
s_{ij}^{\mathrm{abs}}
=
\sum_{r=1}^{d}\left|g_{ij,r}\right|,
```

```math
s_{ij}^{\mathrm{gxi}}
=
\left\|h_{ij}\odot g_{ij}\right\|_1.
```

The gradient is local: it describes the first-order response to an infinitesimal
feature perturbation at the current slide.

For a perturbation delta h_ij,

```math
F_i(H_i+\Delta H_i)
\approx
F_i(H_i)
+
\left\langle g_{ij},\Delta h_{ij}\right\rangle.
```

This approximation can be poor for deleting a patch, replacing it with a
background vector, or crossing a ReLU or attention-saturation boundary.

Gradient signs matter. A positive directional derivative means increasing a
feature direction increases the target locally; it does not mean the patch is
globally necessary.

## 7. Integrated gradients

Choose a baseline bag H_i^0 and interpolate feature-wise:

```math
H_i(\alpha)
=
H_i^0+\alpha(H_i-H_i^0),
\qquad
0\le\alpha\le 1.
```

Integrated gradients for patch j are

```math
\mathrm{IG}_{ij}
=
(h_{ij}-h_{ij}^{0})
\odot
\int_0^1
\frac{\partial F_i(H_i(\alpha),C_i)}
\partial h_{ij}}
d\alpha.
```

The completeness relation is

```math
\sum_{j=1}^{n_i}
\mathbf 1^{\mathsf T}\mathrm{IG}_{ij}
=
F_i(H_i,C_i)
-
F_i(H_i^0,C_i),
```

when the path and numerical integration are implemented consistently.

The baseline is part of the explanation. A zero feature vector, a cohort mean,
a sampled background patch, and a masked-token embedding define different
counterfactual questions.

For WSI bags, a single baseline can be invalid because it may create an
unrealistic slide. A coordinate-preserving baseline can instead replace selected
patches while retaining the same grid and tissue mask.

## 8. Attention rollout and path products

A common rollout construction augments each attention matrix with a residual:

```math
\widetilde A_i^{(\ell)}
=
\lambda_\ell A_i^{(\ell)}
+
(1-\lambda_\ell)I,
```

followed by row normalization if needed. The product across layers is

```math
R_i^{(L)}
=
\widetilde A_i^{(L-1)}
\cdots
\widetilde A_i^{(0)}.
```

The class-to-patch row R_0j is a path-routing score under a simplified linear
model. It is not the exact gradient because it ignores:

- value directions;
- score derivatives;
- MLP Jacobians;
- normalization derivatives;
- PPEG channel filters;
- classifier weights;
- sign cancellations.

Rollout can be useful as a structural diagnostic: it shows how normalized
routing paths compose. It should not be labeled an exact causal attribution.

## 9. Deletion interventions

### 9.1 Feature deletion

Let M be a binary patch mask and define the intervened bag

```math
H_i^{(M)}
=
\begin{bmatrix}
m_1h_{i1}^{\mathsf T}\\
\vdots\\
m_{n_i}h_{in_i}^{\mathsf T}
\end{bmatrix}.
```

The deletion effect is

```math
\Delta_i^{\mathrm{del}}(M)
=
F_i(H_i,C_i)
-
F_i(H_i^{(M)},C_i).
```

For a single patch j, set m_j to zero and all other mask entries to one:

```math
\Delta_{ij}^{\mathrm{del}}
=
F_i(H_i,C_i)
-
F_i(H_i^{(-j)},C_i).
```

### 9.2 Removal versus masking

Removing a row changes sequence length:

```math
H_i^{\mathrm{remove}(-j)}
\in\mathbb R^{(n_i-1)\times d}.
```

Masking preserves length:

```math
H_i^{\mathrm{mask}(-j)}
\in\mathbb R^{n_i\times d}.
```

These are different interventions. In a sequence transformer, removal can
change positional indexing. In TransMIL, it can change the square size and
the repeated-token completion set. Masking preserves the original raster but
requires a defined mask value and a convention for whether masked tokens still
participate in attention or PPEG.

### 9.3 Coordinate-preserving deletion

If physical positions matter, the cleanest single-patch test keeps the
coordinate map and occupancy grid fixed:

```math
(h_{ij},c_{ij},m_{ij}=1)
\longmapsto
(b_{ij},c_{ij},m_{ij}=0),
```

where b_ij is a baseline feature. The model then receives the same slide
geometry and a controlled replacement at one location.

This test is closer to feature necessity. It is not necessarily a realistic
biological counterfactual because the replacement may not correspond to any
plausible tissue.

## 10. Deletion curves

Rank patches by an explanation score s_ij. Let P_k be the top k patches and
let D_k be the remaining patches. A deletion curve is

```math
F_i^{\mathrm{del}}(k)
=
F_i\!\left(H_i^{(D_k)},C_i^{(D_k)}\right).
```

For a positive target, a faithful score should often make the target fall
quickly when high-ranked patches are removed. The area under the curve is

```math
\mathrm{AUC}_{\mathrm{del}}
=
\frac{1}{K}
\sum_{k=0}^{K-1}
\frac{
F_i^{\mathrm{del}}(k)+F_i^{\mathrm{del}}(k+1)
}{2}.
```

The direction of "better" depends on the target and baseline. A low deletion
curve can indicate concentrated evidence, but it can also indicate an
unrealistic deletion intervention or a score that identifies a redundant
cluster member.

An insertion curve starts from a baseline and adds patches in ranked order:

```math
F_i^{\mathrm{ins}}(k)
=
F_i\!\left(H_i^{(P_k)},C_i^{(P_k)}\right).
```

Deletion and insertion should be reported together. A score can perform well
on one and poorly on the other because the two paths visit different
off-manifold inputs.

## 11. Interaction and redundancy

Single-patch deletion assumes an additive interpretation that transformer MIL
does not generally satisfy. For two patches j and k, define the pair synergy

```math
\Gamma_{i,jk}
=
F_i(H_i)
-
F_i(H_i^{(-j)})
-
F_i(H_i^{(-k)})
+
F_i(H_i^{(-j,-k)}).
```

If Gamma is positive, the joint removal changes the prediction more than the
sum of the two isolated removal effects under this sign convention. If Gamma is
negative, the patches may be redundant or mutually compensating.

For a group G, the group deletion effect is not generally additive:

```math
\Delta_i^{\mathrm{del}}(G)
\ne
\sum_{j\in G}\Delta_{ij}^{\mathrm{del}}.
```

This matters for WSI morphology. Neighboring patches can jointly define a
structure that no single patch contains. A heatmap that ranks isolated patches
can therefore understate a spatially distributed explanation.

The Shapley value averages marginal contributions over coalitions:

```math
\phi_{ij}
=
\sum_{S\subseteq N_i\setminus\{j\}}
\frac{|S|!\,(n_i-|S|-1)!}{n_i!}
\left[
F_i(S\cup\{j\})-F_i(S)
\right].
```

Exact Shapley computation is exponential in bag size. Sampling or grouping
patches changes the estimand and must be reported.

## 12. Transformer-specific intervention effects

### 12.1 Softmax redistribution

Deleting patch j changes the denominator of every query row. Even if the
deleted value was unimportant, the remaining coefficients change:

```math
a_{ir}^{(-j)}
=
\frac{\exp(\ell_{ir})}
\sum_{s\ne j}\exp(\ell_{is})
\ne
a_{ir}.
```

The deletion effect therefore includes redistribution, not only removal of one
value vector.

### 12.2 PPEG neighborhood change

If deletion is implemented by compacting the sequence before PPEG, the local
neighborhood of many patches changes. Let N_r be the original local stencil and
N_r^(-j) the stencil after compaction:

```math
\mathcal N_r^{(-j)}
\ne
\mathcal N_r
\quad\text{for many }r.
```

The measured score then includes a geometry intervention. To estimate feature
credit, use a fixed grid and a mask or baseline replacement.

### 12.3 Nyström landmark change

If landmarks are sampled or selected from the current token set, deletion can
change the landmark set:

```math
\mathcal L(H_i^{(-j)})
\ne
\mathcal L(H_i).
```

The prediction change includes both patch deletion and approximation-selection
effects. For a controlled test, fix landmark indices in the original grid or
use a deterministic selection rule whose changed inputs are explicitly recorded.

### 12.4 Class-token feedback

Removing a patch can alter the class state in an early layer, which then changes
all later patch states. A final attention map can consequently change even for
patches that were not deleted:

```math
\Delta A_i^{(L)}
=
A_i^{(L)}(H_i)-A_i^{(L)}(H_i^{(-j)}).
```

This is expected transformer behavior, not a bookkeeping error.

## 13. Signed target credit

Attention coefficients are nonnegative, but target credit is signed. For a
binary logit F_i, a patch can suppress the target:

```math
\Delta_{ij}^{\mathrm{del}}<0
\quad\text{or}\quad
\Delta_{ij}^{\mathrm{del}}>0
```

depending on the deletion convention and target sign. A positive attention
weight does not imply positive evidence.

For a class contrast, the signed contribution can be measured by

```math
\Delta_{ij}^{(c,c')}
=
\left(F_i^{(c)}-F_i^{(c')}\right)
-
\left(F_i^{(c,-j)}-F_i^{(c',-j)}\right).
```

For survival, explain the risk direction explicitly. A patch can raise a risk
score while lowering survival probability at a fixed horizon:

```math
\eta_i^{(-j)}-\eta_i
\quad\text{and}\quad
S_i^{(-j)}(\tau)-S_i(\tau)
```

have opposite signs under a monotone survival head. A heatmap without target
and sign is incomplete.

## 14. Approximation-aware credit

Let A be exact attention and A-hat be the Nyström approximation. The output
difference is

```math
\delta O_i
=
(A_i-\widehat A_i)V_i.
```

For a class-row target with head classifier w, the approximate logit error is

```math
\delta F_i
=
w^{\mathsf T}
\left[(A_i-\widehat A_i)V_i\right]_0.
```

When a patch is deleted, four quantities may change:

1. the exact scores;
2. the exact values;
3. the landmark factors;
4. the pseudoinverse approximation.

Define exact and approximate deletion effects:

```math
\Delta_{ij}^{\mathrm{exact}}
=
F_i^{\mathrm{exact}}(H_i)
-
F_i^{\mathrm{exact}}(H_i^{(-j)}),
```

```math
\Delta_{ij}^{\mathrm{approx}}
=
F_i^{\mathrm{approx}}(H_i)
-
F_i^{\mathrm{approx}}(H_i^{(-j)}).
```

The approximation-induced explanation error is

```math
\varepsilon_{ij}^{\mathrm{expl}}
=
\Delta_{ij}^{\mathrm{approx}}
-
\Delta_{ij}^{\mathrm{exact}}.
```

This quantity should be measured on small bags where exact attention is
available before interpreting approximate-attention heatmaps.

## 15. Faithfulness metrics

### 15.1 Deletion and insertion

Use the ranked curves defined above and report the confidence intervals over
slides. Per-slide curves can be dominated by a few pathological bag sizes.

### 15.2 Infidelity

For perturbation random variable Delta H, infidelity measures mismatch between
the explanation inner product and the actual output change:

```math
\mathrm{Infid}(e)
=
\mathbb E_{\Delta H}
\left[
e(H)^{\mathsf T}\Delta H
-
\left(
F(H)-F(H-\Delta H)
\right)
\right]^2.
```

The perturbation distribution is part of the metric. Independent Gaussian
feature noise and coordinate-preserving patch replacement test different
properties.

### 15.3 Sensitivity

For explanation e, sensitivity can be written as the maximum attribution change
under a small input perturbation:

```math
\mathrm{Sens}(e)
=
\sup_{\|\Delta H\|\le\epsilon}
\|e(H+\Delta H)-e(H)\|.
```

High sensitivity means the heatmap is unstable even if the prediction is stable.

### 15.4 Randomization tests

Randomize model parameters or labels and recompute explanations. A valid
method should respond to the randomized model rather than reproducing the same
input-driven visual structure. The test is diagnostic, not a proof of
faithfulness.

## 16. Baseline and off-manifold failure

Deletion is an intervention only relative to a replacement. A zero vector may
be far outside the feature distribution:

```math
\mathrm{dist}\!\left(b_{ij},\mathcal X_{\mathrm{train}}\right)
\gg
\mathrm{dist}\!\left(h_{ij},\mathcal X_{\mathrm{train}}\right).
```

Then a large deletion effect can reflect out-of-distribution behavior rather
than patch necessity.

Useful baseline families include:

- a local coordinate-matched background feature;
- a conditional generative replacement;
- a cohort mean conditioned on stain or tissue type;
- a masked token with explicit attention masking;
- a physically plausible counterfactual patch.

Each baseline changes the question. The correct report names it instead of
calling all replacements deletion.

## 17. Explanation resolution

A patch-level score can explain only the model input unit unless an additional
mapping is supplied. If one patch covers a physical region R_ij, then the
heatmap is a piecewise constant function:

```math
E_i(x)
=
\sum_{j=1}^{n_i}e_{ij}\mathbf 1\{x\in R_{ij}\}.
```

The resolution is bounded by patch size and overlap. A smooth-looking
interpolation can make an explanation appear more precise than the model input.

For overlapping patches, a pixel-level projection requires an aggregation rule:

```math
E_i(x)
=
\frac{
\sum_{j:x\in R_{ij}}w_{ij}(x)e_{ij}
}{
\sum_{j:x\in R_{ij}}w_{ij}(x)+\epsilon_0
}.
```

This visualization is another readout operator. It should not be confused with
the patch-level model credit.

## 18. Paper-specific comparison

### CLAM

CLAM attention weights rank instances under a gated attention mechanism, with
class-specific branches in the multibranch setting. They are useful for
localization, but the ranking is still a routing statistic unless validated by
deletion or another intervention.

### Additive MIL

An additive model can expose a mathematically cleaner spatial credit assignment
when the slide score decomposes as

```math
F_i
=
b+\sum_{j=1}^{n_i}\phi(h_{ij},c_{ij}).
```

In this special case, the per-patch terms are exact additive contributions to
the model score. A transformer-MIL model with patch-patch context generally does
not have this decomposition because the contribution of one patch depends on
the other patches.

### xMIL

xMIL-style layerwise relevance propagates relevance through a neural MIL
computation graph. Its result is a different credit rule from raw attention and
from deletion. The correct comparison is whether each method's relevance
conservation or perturbation claim matches the score it outputs.

### TransMIL

TransMIL combines class-token readout, PPEG, and approximate self-attention.
Its last class row can be visualized, but a faithful patch intervention must
state whether square padding, grid positions, and landmark selection are held
fixed.

## 19. Sanity checks

### Check A: target declaration

For every heatmap, record the scalar target, logit or probability scale, sign,
layer, head aggregation, and baseline.

### Check B: attention versus gradient

Compute rank correlations among attention, gradient norm, gradient-times-input,
and deletion. Disagreement is expected and informative.

```math
\rho_{\mathrm{attn,del}}
=
\mathrm{corr}\!\left(
\{a_{ij}\}_j,
\{\Delta_{ij}^{\mathrm{del}}\}_j
\right).
```

### Check C: fixed-grid deletion

Compare removal, compacted masking, and coordinate-preserving replacement.
Large differences show that the intervention changes geometry rather than only
feature evidence.

### Check D: group deletion

Delete spatially coherent groups and compare against randomly matched groups.
This tests whether a morphology is distributed rather than concentrated in one
patch.

### Check E: approximation audit

On small bags, compare exact and Nyström logits, attention matrices, and
deletion rankings. Do not infer explanation fidelity from predictive accuracy
alone.

### Check F: randomization

Randomize labels and model weights. A method that continues to produce
biologically plausible-looking maps has failed a basic sanity check.

### Check G: duplicate stress

Duplicate a top-ranked patch without adding new tissue. If the prediction
changes substantially, the model and explanation are sensitive to multiplicity.

## 20. C/R/G/S placement

| Component | Transformer-MIL score-credit realization |
|---|---|
| Context operator | Self-attention, PPEG, residuals, normalization, and MLPs; each creates direct and indirect paths from patches to the slide state. |
| Readout operator | Class-token projection to the selected target scalar. |
| Geometry | Attention and deletion can be set-invariant, raster-conditioned, or coordinate-preserving depending on padding, masking, and intervention design. |
| Surviving statistic | Attention routing, local derivative, path relevance, or intervention effect; these are different surviving statistics and should not be collapsed into one heatmap. |

The explanation pipeline is

```math
\text{patch bag}
\xrightarrow{\text{model}}
\text{target scalar}
\xrightarrow{\text{score or intervention}}
\text{patch credit}
\xrightarrow{\text{spatial projection}}
\text{WSI visualization}.
```

Every arrow adds assumptions. A spatial heatmap is not just the model output
with colors; it is an additional mathematical construction.

## 21. Bottom line

For transformer MIL, the final attention row is a normalized routing statistic.
A gradient is a local sensitivity. Integrated gradients are path-averaged
sensitivities relative to a baseline. Deletion is an intervention whose meaning
depends on whether sequence length, geometry, padding, and approximation
selection are held fixed. Shapley-style scores include interactions but are
computationally expensive.

The honest hierarchy is

```math
\boxed{
\text{attention}
\ne
\text{gradient}
\ne
\text{path relevance}
\ne
\text{deletion}
\ne
\text{causal necessity}
}
```

No single score deserves the unqualified label "the explanation." A serious WSI
report should specify the target, score definition, intervention baseline,
geometry convention, approximation state, and faithfulness test. Without those
details, a heatmap shows where the model routed or reacted under one convention,
not what the slide biologically means.

## References

- Ilse et al., "Attention-based Deep Multiple Instance Learning,"
  https://arxiv.org/abs/1802.04712
- Lu et al., "Data-efficient and weakly supervised computational pathology on
  whole-slide images," CLAM, https://arxiv.org/abs/2004.09666
- Amores et al., "Additive MIL: A Framework for Interpretable Multiple Instance
  Learning," https://arxiv.org/abs/2206.01794
- Schreiber et al., "xMIL: Insightful Explanations for Multiple Instance
  Learning in Histopathology," https://arxiv.org/html/2406.04280v1
- Shao et al., "TransMIL: Transformer based Correlated Multiple Instance
  Learning for Whole Slide Image Classification," https://arxiv.org/abs/2106.00908
- Jain and Wallace, "Attention is not Explanation,"
  https://arxiv.org/abs/1902.10186
- Wiegreffe and Pinter, "Attention is not not Explanation,"
  https://arxiv.org/abs/1908.04626
- Sundararajan et al., "Axiomatic Attribution for Deep Networks,"
  https://arxiv.org/abs/1703.01365
- Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions,"
  https://arxiv.org/abs/1705.07874
