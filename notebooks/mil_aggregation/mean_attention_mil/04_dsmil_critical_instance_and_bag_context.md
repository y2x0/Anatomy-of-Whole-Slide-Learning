# DSMIL Critical Instance and Bag Context

## 1. Scope and source

DSMIL is a dual-stream MIL aggregator. Its first stream selects a
class-critical instance with a max operation. Its second stream uses that
instance as a reference and aggregates information from every patch according
to query similarity.

The primary source is Li et al., Dual-stream Multiple Instance Learning
Network for Whole Slide Image Classification:

https://arxiv.org/abs/2011.08939

The source paper's aggregator has three mathematical ingredients:

1. an instance classifier and max-pooling stream;
2. a query transform for the critical patch and all patches;
3. an information transform whose values are weighted by critical-query
   similarity.

The final binary score averages the instance-stream and bag-stream scores.
This is not ordinary ABMIL. ABMIL learns a pointwise score for each patch.
DSMIL first chooses a witness and then defines a reference-conditioned
weighted first moment.

## 2. Bag and feature notation

Let slide i provide n_i patch embeddings:

```math
H_i
=
\begin{bmatrix}
h_{i1}^{\mathsf T}\\
\vdots\\
h_{in_i}^{\mathsf T}
\end{bmatrix}
\in\mathbb R^{n_i\times d}.
```

The feature extractor is a map f from the original patch to its embedding:

```math
h_{ij}=f_\theta(x_{ij}).
```

The aggregator itself receives H_i. The self-supervised feature-learning
component in the DSMIL paper is upstream of this map and should not be
mistaken for an additional bag-level attention term.

For binary classification, let W_0 be the instance classifier row and W_b be
the bag classifier row. The instance and bag streams produce scalar scores.

## 3. Instance max stream

The instance score for patch j is

```math
c_{ij}^{\mathrm{inst}}
=
W_0h_{ij}.
```

The critical score is the maximum:

```math
c_i^{\mathrm{max}}
=
\max_{1\le j\le n_i}c_{ij}^{\mathrm{inst}}.
```

The critical index is

```math
j_i^\star
\in
\mathrm{arg\,max}_{1\le j\le n_i}
c_{ij}^{\mathrm{inst}}.
```

Choose one deterministic tie rule in implementation. The critical embedding is

```math
h_i^\star
=
h_{i j_i^\star}.
```

This stream preserves an extreme class score:

```math
\left(H_i\mapsto c_i^{\mathrm{max}}\right)
=
\left(
\text{instance scores}
\mapsto
\text{maximum}
\right).
```

It matches the sparse-positive MIL assumption more directly than a mean: one
high-scoring patch can drive the instance branch.

## 4. Critical-index discontinuity

The max value is continuous in the vector of instance scores, but the selected
index is not. For two patches:

```math
c_{i1}^{\mathrm{inst}}
>
c_{i2}^{\mathrm{inst}}
\quad\Longrightarrow\quad
j_i^\star=1,
```

while an arbitrarily small perturbation can switch the index near a tie:

```math
\left|
c_{i1}^{\mathrm{inst}}
-
c_{i2}^{\mathrm{inst}}
\right|
\approx 0.
```

The max score changes smoothly as the winning score changes, but the bag
stream query changes from a transform of h_i1 to a transform of h_i2:

```math
q_i^\star
=
q(h_i^\star)
\quad\longrightarrow\quad
\widetilde q_i^\star
=
q(\widetilde h_i^\star).
```

That switch can change every bag attention weight at once.

Away from ties, the derivative of the max score with respect to patch features
is

```math
\frac{\partial c_i^{\mathrm{max}}}{\partial h_{ij}}
=
\begin{cases}
W_0,&j=j_i^\star,\\
0,&j\ne j_i^\star.
\end{cases}
```

The instance stream is therefore sparse in its direct gradient. The bag stream
can still send gradients to all patches through the weighted information sum.

## 5. Query and information transforms

DSMIL maps every instance into a query and an information vector:

```math
q_{ij}
=
W_qh_{ij},
\qquad
v_{ij}
=
W_vh_{ij}.
```

The dimensions can be written as

```math
W_q\in\mathbb R^{d_q\times d},
\qquad
W_v\in\mathbb R^{d_v\times d},
\qquad
q_{ij}\in\mathbb R^{d_q},
\qquad
v_{ij}\in\mathbb R^{d_v}.
```

The critical query is

```math
q_i^\star
=
q_{i j_i^\star}.
```

The model uses the same query transform for the critical instance and the
other instances. The critical patch is not a separate learned token; it is an
observed instance selected by the max branch.

## 6. Reference-conditioned attention

The raw similarity between patch j and the critical patch is

```math
\ell_{ij}
=
\left\langle q_{ij},q_i^\star\right\rangle.
```

The normalized relation weight is

```math
\alpha_{ij}^{\mathrm{crit}}
=
\frac{\exp(\ell_{ij})}
{\sum_{k=1}^{n_i}\exp(\ell_{ik})}.
```

The bag embedding is

```math
b_i
=
\sum_{j=1}^{n_i}
\alpha_{ij}^{\mathrm{crit}}v_{ij}
\in\mathbb R^{d_v}.
```

The bag-stream score is

```math
c_i^{\mathrm{bag}}
=
W_b b_i.
```

This is a single-query nonlocal operation. Every patch is compared with the
critical query, but patches do not independently query every other patch as in
full self-attention.

## 7. The source paper's final score

For the binary aggregator described in the paper, the final score averages the
two streams:

```math
c_i
=
\frac{1}{2}
\left(
c_i^{\mathrm{max}}
+
c_i^{\mathrm{bag}}
\right).
```

Substitution gives

```math
c_i
=
\frac{1}{2}
\left[
W_0h_i^\star
+
W_b
\sum_{j=1}^{n_i}
\alpha_{ij}^{\mathrm{crit}}v_{ij}
\right].
```

The factor one-half is a fixed fusion rule in the source formulation. A
learned fusion would define a different model:

```math
c_i^{(\lambda)}
=
\lambda c_i^{\mathrm{max}}
+
(1-\lambda)c_i^{\mathrm{bag}}.
```

Unless lambda is explicitly learned or tuned, it should not be presented as a
free mixture.

## 8. What the second stream is and is not

The bag stream resembles attention pooling, but its weights are constrained by
the selected critical patch:

```math
\alpha_i^{\mathrm{crit}}
=
\mathrm{softmax}\!\left(
\left[
\left\langle q_{i1},q_i^\star\right\rangle,\ldots,
\left\langle q_{in_i},q_i^\star\right\rangle
\right]
\right).
```

ABMIL instead computes

```math
\alpha_{ij}^{\mathrm{ABMIL}}
=
\mathrm{softmax}_j\!\left(s(h_{ij})\right).
```

DSMIL's score for patch j depends on a selected other patch. The dependence is
global and reference-conditioned:

```math
\frac{\partial\ell_{ij}}{\partial h_{i\ell}}
\ne 0
\quad\text{when }\ell=j_i^\star.
```

For the critical patch itself, changing h_i-star affects every relation logit:

```math
\frac{\partial\ell_{ij}}{\partial h_i^\star}
=
W_q^{\mathsf T}q_{ij}
\quad\text{up to the selected-query parameterization}.
```

The critical patch is therefore both a value in the max stream and a query
source for the bag stream.

## 9. Raw similarity is not a metric

The source calls the query inner product a distance measurement in the
aggregation discussion, but the mathematical object is a similarity score:

```math
\ell_{ij}
=
\left\langle q_{ij},q_i^\star\right\rangle.
```

The normalized weight is reference-conditioned. Even though the raw inner
product is symmetric,

```math
\left\langle q_{ij},q_{ik}\right\rangle
=
\left\langle q_{ik},q_{ij}\right\rangle,
```

the normalized relation need not be:

```math
\frac{\exp(\langle q_{ij},q_{ik}\rangle)}
{\sum_r\exp(\langle q_{ir},q_{ik}\rangle)}
\ne
\frac{\exp(\langle q_{ik},q_{ij}\rangle)}
{\sum_r\exp(\langle q_{ir},q_{ij}\rangle)}.
```

It is not a metric because it need not satisfy nonnegativity, identity of
indiscernibles, or the triangle inequality. The correct description is
critical-query softmax similarity.

## 10. Permutation invariance

The max stream is invariant to patch permutation:

```math
\max_j W_0(P_iH_i)_j
=
\max_j W_0h_{ij}.
```

The selected feature is the same feature value, though its row index changes.
The bag stream sums all relation-weighted information vectors:

```math
b_i
=
\sum_j\alpha_{ij}^{\mathrm{crit}}v_{ij}.
```

Since the critical query is determined by a permutation-invariant maximum and
the sum is commutative,

```math
b_i(P_iH_i)=b_i(H_i).
```

The final score is therefore a set function, apart from exact max ties and
implementation-dependent tie-breaking.

No physical adjacency is used at this bag level. A spatial shuffle that
preserves the feature multiset leaves the score unchanged.

## 11. Multi-class DSMIL

For C classes, let the instance classifier produce a score vector:

```math
c_{ij}^{\mathrm{inst}}
=
W_0h_{ij}
\in\mathbb R^C.
```

The class-specific critical index is

```math
j_i^\star(c)
\in
\mathrm{arg\,max}_j
\left[c_{ij}^{\mathrm{inst}}\right]_c.
```

The class-specific critical query is

```math
q_i^\star(c)
=
q_{i,j_i^\star(c)}.
```

The bag weights and embedding are

```math
\alpha_{ij}^{(c)}
=
\frac{
\exp\!\left(
\langle q_{ij},q_i^\star(c)\rangle
\right)
}{
\sum_{k=1}^{n_i}
\exp\!\left(
\langle q_{ik},q_i^\star(c)\rangle
\right)
},
```

```math
b_i^{(c)}
=
\sum_j\alpha_{ij}^{(c)}v_{ij}.
```

The bag-stream class score is

```math
c_i^{\mathrm{bag},(c)}
=
\left(w_b^{(c)}\right)^{\mathsf T}b_i^{(c)}.
```

The max stream class score is

```math
c_i^{\mathrm{max},(c)}
=
\max_j\left[c_{ij}^{\mathrm{inst}}\right]_c.
```

The final class score is

```math
c_i^{(c)}
=
\frac{1}{2}
\left(
c_i^{\mathrm{max},(c)}
+
c_i^{\mathrm{bag},(c)}
\right).
```

Each class can therefore select a different critical patch and define a
different reference-conditioned context.

## 12. Critical-instance-centered measure

The bag stream defines a class- or slide-specific measure

```math
\widehat\nu_i^\star
=
\sum_j\alpha_{ij}^{\mathrm{crit}}\delta_{v_{ij}}.
```

The bag embedding is its first moment:

```math
b_i
=
\mathbb E_{\widehat\nu_i^\star}[V].
```

Unlike ABMIL, the measure depends on the selected critical feature:

```math
\widehat\nu_i^\star
=
\widehat\nu_i^\star(H_i,h_i^\star).
```

The surviving statistic is therefore

```math
\left(
c_i^{\mathrm{max}},
b_i
\right),
```

not only a single weighted first moment.

## 13. Critical query and coherent morphology

Suppose a morphology class forms a coherent feature cluster around prototype
mu. If the selected critical query q_i-star is representative, then

```math
\langle q_{ij},q_i^\star\rangle
\quad\text{is large for patches in the same morphology cluster}.
```

The bag stream collects related patches rather than only the single maximum.
This is the intended inductive bias:

```math
\text{one suspicious witness}
\longrightarrow
\text{similar morphology context}.
```

It differs from a pure max operator, which preserves only the largest score,
and from ABMIL, whose query is a learned global parameter rather than a
slide-specific witness.

The assumption is not guaranteed. If the max patch is an artifact, the
reference context can collect artifact-like patches.

## 14. Gradient structure

Away from a max tie, the max stream has gradient only through the critical
patch:

```math
\frac{\partial c_i^{\mathrm{max}}}{\partial h_{ij}}
=
\mathbf 1\{j=j_i^\star\}W_0.
```

The bag stream has direct value gradients:

```math
\frac{\partial b_i}{\partial v_{ij}}
=
\alpha_{ij}^{\mathrm{crit}}I_{d_v}
\quad\text{when weights are held fixed}.
```

It also has routing gradients:

```math
\frac{\partial b_i}{\partial \ell_{ij}}
=
\alpha_{ij}^{\mathrm{crit}}
\left(
v_{ij}-b_i
\right).
```

The critical query changes all logits:

```math
\frac{\partial\ell_{ij}}{\partial q_i^\star}
=
q_{ij}.
```

Thus a perturbation to the critical patch affects:

- its max-stream score;
- its query vector;
- every bag-stream attention weight;
- the bag embedding;
- the final fused score.

The selected index itself has no ordinary derivative away from ties. This is
the hard-selection bottleneck.

## 15. Critical-instance instability

Consider two nearly tied candidates j and ell. If j is selected:

```math
b_i^{(j)}
=
\sum_r
\frac{\exp(\langle q_{ir},q_{ij}\rangle)}
{\sum_k\exp(\langle q_{ik},q_{ij}\rangle)}
v_{ir}.
```

If ell becomes selected:

```math
b_i^{(\ell)}
=
\sum_r
\frac{\exp(\langle q_{ir},q_{i\ell}\rangle)}
{\sum_k\exp(\langle q_{ik},q_{i\ell}\rangle)}
v_{ir}.
```

The two outputs can be far apart even when h_ij and h_i-ell are close in
instance score:

```math
\left|
c_{ij}^{\mathrm{inst}}
-
c_{i\ell}^{\mathrm{inst}}
\right|
\ll 1,
\qquad
\left\|
b_i^{(j)}-b_i^{(\ell)}
\right\|_2
\gg 0.
```

An explanation map can therefore change globally after a local score
perturbation.

## 16. Relation to max and ABMIL

DSMIL contains an explicit max-like branch:

```math
c_i^{\mathrm{max}}
=
\max_j c_{ij}^{\mathrm{inst}}.
```

It also contains a critical-conditioned weighted mean:

```math
b_i
=
\sum_j\alpha_{ij}^{\mathrm{crit}}v_{ij}.
```

ABMIL has a learned pointwise score:

```math
z_i^{\mathrm{ABMIL}}
=
\sum_j
\mathrm{softmax}_j(s(h_{ij}))h_{ij}.
```

DSMIL is not equivalent to ABMIL unless the critical query is fixed or the
query similarity can be represented by a pointwise score independent of the
selected instance. In the actual construction, the selected instance is
slide-dependent.

DSMIL is not equivalent to max pooling because the bag stream retains
information from all patches. It is not full self-attention because only the
critical query compares against all patch queries.

## 17. Bag-size behavior

The critical-query weights are normalized:

```math
\sum_j\alpha_{ij}^{\mathrm{crit}}=1.
```

Adding unrelated patches changes the denominator and can reduce the mass of
existing patches. If background queries have similarity ell_b and related
queries have similarity ell_p, r related patches among n_i total receive

```math
\alpha_{\mathrm{related}}
=
\frac{r\exp(\ell_p)}
{r\exp(\ell_p)+(n_i-r)\exp(\ell_b)}.
```

The critical query protects a morphology only if the similarity gap is large
enough to overcome background multiplicity.

The max stream is less sensitive to background count when background scores
remain below the critical score, but its probability of encountering an
artifact increases with n_i.

## 18. Multiscale WSI use

The DSMIL paper combines features from multiple magnifications in a pyramidal
feature-learning strategy. At the aggregator interface, all selected patch
embeddings must be represented in a common width or fused before the dual
streams.

For scale m, let

```math
H_i^{(m)}
\in\mathbb R^{n_i^{(m)}\times d_m}.
```

If scale-specific maps are projected to d:

```math
\widetilde h_{ij}^{(m)}
=
W_mh_{ij}^{(m)}.
```

The critical selection can be performed within one scale or after concatenated
features. These are different operators:

```math
\mathrm{arg\,max}_{m,j}
c_{ij}^{(m)}
\quad\text{versus}\quad
\mathrm{arg\,max}_{j}
c_{ij}^{(\mathrm{fused})}.
```

The first chooses a scale and patch jointly. The second lets a fused feature
determine the witness. A multiscale result should state which is used.

## 19. Failure modes

### 19.1 False critical instance

A high-scoring artifact becomes the query and can amplify similar artifacts
through the bag stream.

### 19.2 Critical tie

Near-ties cause a discontinuous query switch and globally different weights.

### 19.3 Background denominator

Many moderately similar background patches can dilute coherent positive mass.

### 19.4 Query-key collapse

If all q vectors are nearly identical, the bag weights approach uniform:

```math
\left\|q_{ij}-q_{ik}\right\|_2\to 0
\quad\Longrightarrow\quad
\alpha_{ij}^{\mathrm{crit}}\to\frac{1}{n_i}.
```

The bag stream then behaves like mean pooling over information vectors.

### 19.5 Overconcentration

If one patch has a much larger query similarity, the bag stream behaves like a
second max:

```math
\max_j\ell_{ij}-\ell_{i\ell}\to+\infty
\quad\Longrightarrow\quad
\alpha_{ij}^{\mathrm{crit}}\to 1.
```

### 19.6 Wrong relation interpretation

The normalized relation is not a symmetric distance and does not prove
physical adjacency. It is similarity to a selected feature-space reference.

### 19.7 Coordinate erasure

Spatial shuffling leaves the aggregator unchanged when features are fixed. A
critical-instance map can localize a patch but cannot model arrangement.

## 20. Sanity checks

### Check A: max stability

Perturb instance scores and record critical-index stability:

```math
\mathrm{Stab}_i
=
\mathbf 1\{j_i^\star=j_i^{\star\prime}\}.
```

Report stability across seeds and small feature perturbations.

### Check B: query-switch effect

Force each of the top candidate patches to act as the critical query and
measure the bag embedding distance:

```math
D_i(j,\ell)
=
\left\|b_i^{(j)}-b_i^{(\ell)}\right\|_2.
```

Large distances near score ties indicate high global sensitivity.

### Check C: max versus bag score

Report the two stream scores separately:

```math
c_i^{\mathrm{max}},
\qquad
c_i^{\mathrm{bag}},
\qquad
\frac{1}{2}
\left(c_i^{\mathrm{max}}+c_i^{\mathrm{bag}}\right).
```

A final score without its stream decomposition hides which inductive bias
dominated.

### Check D: coordinate shuffle

Shuffle physical coordinates without changing features. The score should stay
fixed for DSMIL's set aggregator; any change occurs in downstream rendering.

### Check E: background duplication

Duplicate unrelated patches and measure critical score, bag score, and final
score separately. This identifies denominator and max-exposure effects.

### Check F: similarity calibration

Measure the distribution of critical-query similarities and the effective
sample size:

```math
n_{\mathrm{eff},i}^{\mathrm{crit}}
=
\frac{1}{\sum_j(\alpha_{ij}^{\mathrm{crit}})^2}.
```

## 21. C/R/G/S placement

| Component | DSMIL realization |
|---|---|
| Context operator | Critical-query nonlocal similarity: every patch query is compared with the query of the max-scored critical instance. |
| Readout operator | Average of a max instance score and a critical-query weighted information sum. |
| Geometry | Set-invariant feature-space relation; no physical WSI adjacency unless encoded upstream. |
| Surviving statistic | An extreme class score plus a reference-conditioned weighted first moment of information vectors. |

The forward map is

```math
\text{patch embeddings}
\xrightarrow{\text{instance classifier}}
\text{critical index}
\xrightarrow{\text{query transform}}
\text{reference-conditioned weights}
\xrightarrow{\text{information sum}}
\text{bag score}
\xrightarrow{\text{average with max score}}
\text{slide score}.
```

DSMIL therefore sits between max MIL and attention MIL:

```math
\boxed{
\text{DSMIL}
=
\text{max witness}
\;+\;
\text{critical-query weighted moment}
}
```

## 22. Bottom line

DSMIL does not learn one independent attention score per patch and stop there.
It first chooses a class-critical instance with an instance classifier, then
uses that instance as a slide-specific query for a nonlocal weighted sum.

The critical patch survives as an explicit max statistic and as the reference
that shapes the bag statistic. This can collect coherent morphology more
effectively than a uniform mean or a single global attention query. It also
creates a sharp failure mode: a wrong or unstable critical instance changes all
bag weights at once.

The mathematically honest description is:

```math
\boxed{
\text{critical-instance selection}
\longrightarrow
\text{reference-conditioned context}
\longrightarrow
\text{dual-stream score}
}
```

The selected patch is a learned witness, not a ground-truth lesion label. A
DSMIL heatmap should be accompanied by critical-index stability, stream-wise
scores, query-similarity support, and coordinate-shuffle controls.

## References

- Li et al., "Dual-Stream Multiple Instance Learning Network for Whole Slide
  Image Classification," https://arxiv.org/abs/2011.08939
- Ilse et al., "Attention-based Deep Multiple Instance Learning,"
  https://arxiv.org/abs/1802.04712
- Lu et al., "Data-efficient and weakly supervised computational pathology on
  whole-slide images," https://arxiv.org/abs/2004.09666
