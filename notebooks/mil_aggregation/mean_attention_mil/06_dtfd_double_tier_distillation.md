# DTFD-MIL Double-Tier Distillation

Source:

Zhang et al., DTFD-MIL: Double-Tier Feature Distillation Multiple Instance
Learning for Histopathology Whole Slide Image Classification, CVPR 2022.

[Paper on arXiv](https://arxiv.org/abs/2203.12081)

[CVPR open-access version](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_DTFD-MIL_Double-Tier_Feature_Distillation_Multiple_Instance_Learning_for_Histopathology_Whole_CVPR_2022_paper.pdf)

This note derives the construction in the paper rather than treating
double-tier MIL as a generic name for any hierarchical model. The central
operation is a random partition of one slide into pseudo-bags, followed by a
Tier-1 AB-MIL model, a paper-specific feature-distillation rule, and a Tier-2
AB-MIL model over the distilled features.

The most important boundary is supervision. The parent slide label is copied
onto every pseudo-bag, but that copied target is not an observed local label.
A positive slide can contain pseudo-bags with no positive patch. DTFD-MIL uses
the resulting noisy auxiliary targets because the smaller pseudo-bags create
more optimization units and because Tier 2 can aggregate the distilled
features again at the slide level.

## 1. Scope and notation

Let slide n contain K_n extracted patch features. The feature extractor is
treated as fixed in the basic derivation:

```math
X_n
=
\{x_{n,k}\}_{k=1}^{K_n},
\qquad
h_{n,k}
=
H_\phi(x_{n,k})
\in
\mathbb R^D.
```

The observed slide label is binary:

```math
Y_n
\in
\{0,1\}.
```

The latent patch labels are not observed:

```math
Z_{n,k}
\in
\{0,1\}.
```

The classical positive-instance MIL assumption is:

```math
Y_n
=
\mathbf 1
\left\{
\sum_{k=1}^{K_n}Z_{n,k}
>
0
\right\}.
```

This assumption is useful for reasoning about label noise, but it is not a
training target available to DTFD-MIL. The paper trains with slide labels and
constructs pseudo-bags from the observed patch feature collection.

The notation below distinguishes four levels:

| Symbol | Meaning |
| --- | --- |
| X_n | patch inputs from slide n |
| h_n,k | extracted feature for patch k |
| X_n,m | m-th pseudo-bag of inputs |
| f-hat_n,m | distilled feature forwarded to Tier 2 |
| y-hat_n | final Tier-2 slide prediction |

The paper uses an approximately even random split into M pseudo-bags. We use
M for the number of pseudo-bags and K_n for the original slide bag size.

## 2. Random pseudo-bag construction

For each slide, choose a partition of the patch indices:

```math
\{1,\ldots,K_n\}
=
\biguplus_{m=1}^{M}\mathcal B_{n,m},
\qquad
\mathcal B_{n,m}
\cap
\mathcal B_{n,m'}
=
\varnothing
\quad
\text{for }
m
\ne
m'.
```

The corresponding pseudo-bag is:

```math
X_{n,m}
=
\{x_{n,k}:k\in\mathcal B_{n,m}\},
\qquad
H_{n,m}
=
\{h_{n,k}:k\in\mathcal B_{n,m}\}.
```

The split is approximately balanced:

```math
\left|
|\mathcal B_{n,m}|
-
\frac{K_n}{M}
\right|
\leq
1.
```

The parent label is copied:

```math
\widetilde Y_{n,m}
=
Y_n.
```

The tilde is important. This is a generated training target, not the latent
truth of the pseudo-bag:

```math
Y_{n,m}^{\star}
=
\mathbf 1
\left\{
\sum_{k\in\mathcal B_{n,m}}Z_{n,k}
>
0
\right\}.
```

The pseudo-bag label error event on a positive slide is:

```math
\{Y_n=1,\;Y_{n,m}^{\star}=0,\;\widetilde Y_{n,m}=1\}.
```

DTFD-MIL does not claim that the random partition produces tissue regions.
Unless coordinates or a spatial grouping rule are explicitly used, membership
in the same pseudo-bag is a computational allocation.

## 3. Positive-instance capture probability

Suppose a positive slide has K_n^+ positive patches:

```math
K_n^+
=
\sum_{k=1}^{K_n}Z_{n,k},
\qquad
p_n
=
\frac{K_n^+}{K_n}.
```

Consider a pseudo-bag of size m sampled uniformly without replacement. The
probability that it contains no positive patch is hypergeometric:

```math
\Pr
\left(
Y_{n,m}^{\star}=0
\mid
Y_n=1
\right)
=
\frac{
\binom{K_n-K_n^+}{m}
}{
\binom{K_n}{m}
}.
```

If the slide is large and the sampling fraction is small, the same quantity
has the approximation:

```math
\Pr
\left(
Y_{n,m}^{\star}=0
\mid
Y_n=1
\right)
\approx
(1-p_n)^m.
```

The complement is the probability that a pseudo-bag contains at least one
positive patch:

```math
\Pr
\left(
Y_{n,m}^{\star}=1
\mid
Y_n=1
\right)
=
1
-
\frac{
\binom{K_n-K_n^+}{m}
}{
\binom{K_n}{m}
}.
```

Because the partition is disjoint, the events for different pseudo-bags are
not independent. Nevertheless, linearity of expectation gives the expected
number of truly positive pseudo-bags:

```math
\mathbb E
\left[
\sum_{m=1}^{M}
Y_{n,m}^{\star}
\middle|
Y_n=1
\right]
=
M
\left[
1
-
\frac{
\binom{K_n-K_n^+}{m}
}{
\binom{K_n}{m}
}
\right],
```

where m denotes the common pseudo-bag size when the split is exactly even.

The expected number of inherited positive pseudo-bags is M. Therefore the
expected number of false positive pseudo-bag targets is:

```math
\mathbb E
\left[
\sum_{m=1}^{M}
\left(
\widetilde Y_{n,m}
-
Y_{n,m}^{\star}
\right)
\middle|
Y_n=1
\right]
=
M
\frac{
\binom{K_n-K_n^+}{m}
}{
\binom{K_n}{m}
}.
```

This is a useful diagnostic, not a claim that DTFD estimates this quantity
during training. It makes the M tradeoff visible: increasing M gives more
small pseudo-bags, but decreases m and can increase local label corruption.

### A small counterexample

Take a positive slide with eight patches and one positive patch. Split it into
four pseudo-bags of size two. A randomly chosen pseudo-bag contains the
positive patch with probability:

```math
\Pr
\left(
Y_{n,m}^{\star}=1
\right)
=
1
-
\frac{\binom{7}{2}}{\binom{8}{2}}
=
\frac{1}{4}.
```

Exactly one of the four disjoint pseudo-bags contains the positive patch, but
all four receive the inherited positive target. Three of the four Tier-1
targets are therefore locally false under the positive-instance interpretation.
The second tier can still learn from the set of distilled features, but it
does not retroactively make those three targets correct.

## 4. Why DTFD enlarges the bag count without creating new slides

The original training set contains N observed slide labels. A random split
creates approximately N M pseudo-bag training examples:

```math
\{(X_n,Y_n)\}_{n=1}^{N}
\longrightarrow
\{(X_{n,m},\widetilde Y_{n,m})\}_{n=1,\ldots,N;\;m=1,\ldots,M}.
```

The count increases, but the labels within one parent slide remain dependent:

```math
\widetilde Y_{n,1}
=
\widetilde Y_{n,2}
=
\cdots
=
\widetilde Y_{n,M}
=
Y_n.
```

Thus N M is a computational example count, not an independent sample count.
If a slide-level random effect induces intraclass correlation rho between the
loss contributions of two pseudo-bags from the same parent, a simple
clustered-data design-effect diagnostic is:

```math
\mathrm{DEFF}
\approx
1
+
(M-1)\rho.
```

The corresponding effective count is only a rough diagnostic:

```math
N_{\mathrm{eff}}
\approx
\frac{N M}{1+(M-1)\rho}.
```

This expression is not part of the DTFD-MIL method. It prevents an incorrect
interpretation of pseudo-bag multiplication as the creation of independent
patients. The benefit is primarily optimization and representation learning,
not a license to report an artificially enlarged cohort.

## 5. Tier-1 attention-based MIL

The paper uses classic attention-based MIL as the base model for each tier.
For a pseudo-bag with m_n,m instances, define the gated attention score:

```math
r_{n,m,k}
=
w_1^{\mathsf T}
\left[
\tanh
\left(
V_1 h_{n,k}
\right)
\odot
\sigma
\left(
V_2 h_{n,k}
\right)
\right].
```

Here V_1 and V_2 map the D-dimensional instance feature to an attention
hidden space of dimension A, w_1 is in that hidden space, and the product is
coordinatewise.

Normalize within the pseudo-bag:

```math
\alpha_{n,m,k}
=
\frac{
\exp(r_{n,m,k})
}{
\sum_{j\in\mathcal B_{n,m}}
\exp(r_{n,m,j})
},
\qquad
\sum_{k\in\mathcal B_{n,m}}
\alpha_{n,m,k}
=
1.
```

The Tier-1 pseudo-bag embedding is:

```math
F_{n,m}^{(1)}
=
\sum_{k\in\mathcal B_{n,m}}
\alpha_{n,m,k}h_{n,k}
\in
\mathbb R^D.
```

For a C-class classifier, the Tier-1 logit for class c is:

```math
s_{n,m,c}^{(1)}
=
\left(w_{2,c}^{(1)}\right)^{\mathsf T}
F_{n,m}^{(1)}
+
b_{2,c}^{(1)}.
```

For binary classification, the positive pseudo-bag probability may be written:

```math
y_{n,m}
=
\sigma
\left(
s_{n,m,1}^{(1)}
-
s_{n,m,0}^{(1)}
\right).
```

The key separation is:

```math
\alpha_{n,m,k}
\quad
\text{is a normalized readout coefficient},
```

whereas:

```math
p_{n,m,k}^{c}
\quad
\text{is a class-conditioned signal derived from the classifier logit}.
```

They answer different questions. Attention describes how the current
embedding is formed. The derived class signal asks how a class-specific
classifier responds to a patch feature in the Grad-CAM construction.

## 6. Tier-1 inherited-label loss

The paper trains Tier 1 with cross entropy against the copied parent label:

```math
\mathcal L_1
=
-
\frac{1}{N M}
\sum_{n=1}^{N}
\sum_{m=1}^{M}
\left[
Y_n\log(y_{n,m})
+
(1-Y_n)\log(1-y_{n,m})
\right].
```

The target is written as Y_n because the paper defines the pseudo-bag label
to equal the parent label. In a latent-label analysis, the supervision is
misaligned whenever:

```math
Y_n
\ne
Y_{n,m}^{\star}.
```

For a positive slide, the objective pushes every pseudo-bag toward a positive
prediction, including pseudo-bags that contain no positive patch. This can
make Tier 1 learn slide-correlated background or morphology instead of a
strictly local disease signal. It can also be useful: the noisy auxiliary
classifier is trained on smaller bags and supplies candidate features to Tier 2.

The paper's claim is therefore a tradeoff, not a theorem that inherited
pseudo-bag labels are harmless.

## 7. AB-MIL as a Grad-CAM-compatible network

The DTFD derivation treats AB-MIL as a special case of a feature-extractor
plus global-average-pooling plus classifier architecture. Let h_k be the
instance features in a generic bag of K instances. The AB-MIL embedding is:

```math
F
=
\sum_{k=1}^{K}a_k h_k.
```

To expose the same operation as global average pooling, construct a spatial
feature map with K locations:

```math
\widehat h_{k,d}
=
K a_k h_{k,d},
\qquad
k\in\{1,\ldots,K\},
\quad
d\in\{1,\ldots,D\}.
```

Then:

```math
\frac{1}{K}
\sum_{k=1}^{K}
\widehat h_k
=
\sum_{k=1}^{K}
a_k h_k
=
F.
```

The K factor is what makes the global average of the constructed map equal to
the attention-pooled representation. This artificial spatial map is an
algebraic device. The index k need not be a spatially adjacent image location.

Let s_c be the class-c logit produced after the pooled representation. Define
the Grad-CAM channel weight:

```math
\beta_d^c
=
\frac{1}{K}
\sum_{i=1}^{K}
\frac{
\partial s_c
}{
\partial \widehat h_{i,d}
}.
```

The instance-level class activation score is:

```math
L_{k}^{c}
=
\sum_{d=1}^{D}
\beta_d^c
\widehat h_{k,d}.
```

Substituting the constructed feature map gives:

```math
L_{k}^{c}
=
K a_k
\sum_{d=1}^{D}
\beta_d^c h_{k,d}.
```

This score contains three ingredients:

1. the attention coefficient a_k;
2. the direction of h_k in the class-sensitive channel-weighted space;
3. the gradient of the class logit with respect to the constructed feature map.

The score is not a causal intervention effect. It is a model-derived class
activation statistic used by DTFD to rank candidate instances.

## 8. DTFD instance probability

For each instance k and class c, DTFD applies a softmax across classes:

```math
p_k^c
=
\frac{
\exp(L_k^c)
}{
\sum_{t=1}^{C}
\exp(L_k^t)
}.
```

For binary classification, c is typically the negative or positive class and:

```math
p_k^0
+
p_k^1
=
1.
```

The index in the denominator is a class index, not an instance index. This is
the most common notation error when summarizing the paper:

```math
\sum_{t=1}^{C}
\exp(L_k^t)
\quad
\ne
\quad
\sum_{j=1}^{K}
\exp(L_j^c).
```

The probability p_k^1 is therefore a class-conditioned patch signal. It is
not the probability that exactly one patch among the K patches is positive,
and it is not calibrated against a patch-level ground truth in the DTFD
training setup.

The raw attention weight is also a different object:

```math
\alpha_k
\ne
p_k^1
\quad
\text{in general}.
```

Two patches can receive comparable attention but have different channel
directions relative to the positive classifier. Conversely, a patch with a
large attention weight need not have a large positive class signal.

### What is exact and what is approximate

The paper's construction is exact relative to the Grad-CAM representation
defined by the artificial feature map and the AB-MIL classifier. Its
interpretation as a probability of an underlying positive patch is not exact.
The following statements should not be conflated:

```math
p_k^1
=
\text{softmax-normalized class activation signal},
```

```math
\Pr(Z_k=1\mid X,Y)
=
\text{latent patch posterior under a separate probabilistic model}.
```

DTFD defines the first quantity. It does not identify the second without
additional assumptions and calibration data.

## 9. Four paper-specific distillation operators

Let p_{n,m,k}^1 be the Tier-1 positive class signal for patch k in pseudo-bag
m of slide n. Let alpha_{n,m,k} be its Tier-1 attention weight.

### 9.1 MaxS

Maximum selection chooses the patch with maximum derived positive probability:

```math
k_{n,m}^{+}
=
\arg\max_{k\in\mathcal B_{n,m}}
p_{n,m,k}^{1}.
```

The distilled feature is:

```math
\widehat f_{n,m}^{\mathrm{MaxS}}
=
h_{n,k_{n,m}^{+}}.
```

MaxS is a hard top-one readout. It has low transmission bandwidth and can
preserve a highly discriminative patch. It also inherits every ranking error
of the Tier-1 score.

### 9.2 MaxMinS

MaxMinS chooses both the maximum and minimum positive-probability patches:

```math
k_{n,m}^{+}
=
\arg\max_{k\in\mathcal B_{n,m}}
p_{n,m,k}^{1},
\qquad
k_{n,m}^{-}
=
\arg\min_{k\in\mathcal B_{n,m}}
p_{n,m,k}^{1}.
```

The paper concatenates their features:

```math
\widehat f_{n,m}^{\mathrm{MaxMinS}}
=
\left[
h_{n,k_{n,m}^{+}}
\middle\Vert
h_{n,k_{n,m}^{-}}
\right]
\in
\mathbb R^{2D}.
```

If the Tier-2 implementation expects width D, an explicit projection or
dimension-matched Tier-2 head is required. Concatenation does not preserve
the original feature width automatically.

The motivation given by DTFD is geometric rather than probabilistic. MaxS can
push the Tier-2 decision boundary too tightly toward selected positive
examples. Including a low-scoring feature gives Tier 2 a contrasting point
from the same pseudo-bag and can loosen that boundary.

The minimum-scoring feature is not guaranteed to be a true negative. It can be
normal tissue, an artifact, an outlier, or simply a patch that the imperfect
Tier-1 model failed to recognize.

### 9.3 MAS

Maximum-attention selection uses the raw Tier-1 attention coefficient:

```math
k_{n,m}^{\mathrm{attn}}
=
\arg\max_{k\in\mathcal B_{n,m}}
\alpha_{n,m,k}.
```

The distilled feature is:

```math
\widehat f_{n,m}^{\mathrm{MAS}}
=
h_{n,k_{n,m}^{\mathrm{attn}}}.
```

MAS is deliberately different from MaxS. It asks which patch received the
largest contribution to the Tier-1 embedding under the attention module,
whereas MaxS asks which patch received the largest class-conditioned signal
after the Grad-CAM construction.

### 9.4 AFS

Aggregated feature selection forwards the Tier-1 attention embedding:

```math
\widehat f_{n,m}^{\mathrm{AFS}}
=
F_{n,m}^{(1)}
=
\sum_{k\in\mathcal B_{n,m}}
\alpha_{n,m,k}h_{n,k}.
```

AFS retains a weighted first moment of the pseudo-bag. It is not a selected
patch and it does not preserve the full pseudo-bag distribution. It can be
more stable than hard selection while still depending on the Tier-1
attention geometry.

### 9.5 Operator comparison

| Rule | Statistic passed upward | Hard selection | Main risk |
| --- | --- | --- | --- |
| MaxS | one maximum-score feature | yes | ranking error and overly tight boundary |
| MaxMinS | concatenated maximum and minimum | yes, twice | minimum can be an outlier |
| MAS | one maximum-attention feature | yes | attention is not class probability |
| AFS | attention-weighted first moment | no | rare evidence can be averaged away |

The four rules are not interchangeable names for “distillation.” They impose
different readout operators and therefore different surviving statistics.

## 10. Tier-2 attention-based MIL

Collect the M distilled features from a parent slide:

```math
\widehat D_n
=
\left\{
\widehat f_{n,m}
\right\}_{m=1}^{M}.
```

Tier 2 applies another AB-MIL attention operator. Let t be its gated score:

```math
u_{n,m}
=
w_3^{\mathsf T}
\left[
\tanh
\left(
V_3\widehat f_{n,m}
\right)
\odot
\sigma
\left(
V_4\widehat f_{n,m}
\right)
\right].
```

Normalize across pseudo-bags:

```math
\gamma_{n,m}
=
\frac{
\exp(u_{n,m})
}{
\sum_{r=1}^{M}
\exp(u_{n,r})
}.
```

The Tier-2 slide representation is:

```math
F_n^{(2)}
=
\sum_{m=1}^{M}
\gamma_{n,m}
\widehat f_{n,m}.
```

For a binary Tier-2 head:

```math
s_n^{(2)}
=
\left(w_4^{(2)}\right)^{\mathsf T}
F_n^{(2)}
+
b_4^{(2)},
```

```math
\widehat y_n
=
\sigma
\left(
s_n^{(2)}
\right).
```

The final prediction is a prediction of the parent slide. The Tier-2
attention coefficient gamma is a weight over distilled pseudo-bag features,
not a patch-level probability.

## 11. Tier-2 loss and the two optimization problems

The Tier-2 slide loss is:

```math
\mathcal L_2
=
-
\frac{1}{N}
\sum_{n=1}^{N}
\left[
Y_n\log(\widehat y_n)
+
(1-Y_n)\log(1-\widehat y_n)
\right].
```

The paper presents the overall training process as separate minimizations over
the Tier-1 and Tier-2 parameters:

```math
\theta_1^{\star}
\in
\arg\min_{\theta_1}
\mathcal L_1,
\qquad
\theta_2^{\star}
\in
\arg\min_{\theta_2}
\mathcal L_2.
```

The paper's displayed equation writes this schematically as:

```math
\mathcal L
=
\arg\min_{\theta_1}\mathcal L_1
+
\arg\min_{\theta_2}\mathcal L_2.
```

This should not automatically be rewritten as a single end-to-end objective
with an arbitrary weight lambda. A jointly differentiated variant would be a
different optimization statement:

```math
\mathcal L_{\mathrm{joint}}
=
\mathcal L_2
+
\lambda\mathcal L_1,
\qquad
\lambda
\geq
0.
```

That variant may be sensible in another implementation, but it is not the
same claim as the paper's separate Tier-1 and Tier-2 formulation. Hard
selection also creates a non-smooth path between the two tiers.

## 12. Gradient and selection boundaries

The Tier-1 attention embedding is differentiable with respect to h when the
attention network is differentiable:

```math
F^{(1)}
=
\sum_k\alpha_k h_k.
```

MaxS instead has the form:

```math
\widehat f^{\mathrm{MaxS}}
=
h_{k^{+}},
\qquad
k^{+}
=
\arg\max_k p_k^1.
```

Away from ties, the selected feature is locally constant as a function of the
scores, but a small score perturbation can switch the selected index:

```math
p_a^1
>
p_b^1
\longrightarrow
k^{+}=a,
```

```math
p_a^1
<
p_b^1
\longrightarrow
k^{+}=b.
```

At the tie boundary, the hard selection map is discontinuous as a map from
scores to selected feature identity. The final Tier-2 output can therefore
change sharply even when the Tier-1 scores change only slightly.

For MaxMinS, two switching boundaries coexist:

```math
\left(
k^{+},k^{-}
\right)
=
\left(
\arg\max_k p_k^1,
\arg\min_k p_k^1
\right).
```

AFS avoids this discrete identity switch at the distillation step, but its
gradient still passes through the attention normalization and the Tier-1
feature map.

## 13. Computation and the role of M

Let m be the approximate pseudo-bag size:

```math
m
\approx
\frac{K_n}{M}.
```

If Tier 1 processes all M pseudo-bags, the total number of patch-feature
visits is still approximately K_n:

```math
\sum_{m=1}^{M}
|\mathcal B_{n,m}|
=
K_n.
```

The first-tier attention normalization is performed over sets of size about
m instead of one set of size K_n. The second tier normalizes over M distilled
features:

```math
\text{Tier-1 normalization scale}
\sim
m,
\qquad
\text{Tier-2 normalization scale}
\sim
M.
```

This gives the architecture a two-level computational shape. It does not
remove feature-extraction cost, and it does not make the original WSI a
smaller image.

In the paper's reported tables, DTFD-MIL uses five pseudo-bags for CAMELYON16
and eight pseudo-bags for the TCGA lung setting. These are experimental
settings, not a universal law:

```math
M_{\mathrm{CAMELYON16}}=5,
\qquad
M_{\mathrm{TCGA\ lung}}=8.
```

The choice of M changes both the computational granularity and the
pseudo-bag-label noise. It should be reported together with the split rule,
patch count, and whether the random partition is resampled.

## 14. Why a random split is not a spatial hierarchy

Let c_{n,k} be the physical coordinate of patch k. A spatial hierarchy would
construct pseudo-bags using a map that depends on coordinates:

```math
\mathcal B_{n,m}
=
\left\{
k:
\varphi(c_{n,k})=m
\right\}.
```

DTFD's random split instead samples a partition independently of coordinates:

```math
\mathcal B_{n,m}
=
\pi_n^{-1}
\left(
\text{index block }m
\right),
```

where pi_n is a random permutation. Two neighboring patches can be placed in
different pseudo-bags, and two distant patches can be placed in the same
pseudo-bag.

Consider two slides with identical feature multiset and different layouts:

```math
\{(h_k,c_k)\}_{k=1}^{K}
\quad
\text{and}
\quad
\{(h_k,c_k')\}_{k=1}^{K},
```

with the same h values but different coordinates c and c-prime. If the model
receives only h and the random partition, its conditional output does not
contain physical layout information:

```math
f_{\mathrm{DTFD}}
\left(
\{h_k\},\pi
\right)
=
f_{\mathrm{DTFD}}
\left(
\{h_k\},\pi
\right)
.
```

If the partition is resampled, the expectation over partitions is still a
function of the feature multiset rather than the original geometry:

```math
\mathbb E_{\pi}
\left[
f_{\mathrm{DTFD}}
\left(
\{h_k\},\pi
\right)
\right]
=
G
\left(
\{h_k\}
\right).
```

Coordinates can be concatenated to h, or a coordinate-aware partition can be
introduced, but then the representation is a modified method. The vanilla
DTFD construction should be placed in set or pseudo-bag MIL, not described as
a spatial hierarchy.

## 15. What survives each tier

The original slide contains K_n features. Tier 1 produces one embedding per
pseudo-bag:

```math
\left|
\{h_{n,k}\}_{k=1}^{K_n}
\right|
=
K_n
\longrightarrow
\left|
\{F_{n,m}^{(1)}\}_{m=1}^{M}
\right|
=
M.
```

Distillation may reduce each pseudo-bag even further:

```math
\{h_{n,k}:k\in\mathcal B_{n,m}\}
\longrightarrow
\widehat f_{n,m}.
```

The surviving statistic depends on the rule:

```math
\mathrm{MaxS}
\longrightarrow
\text{one extreme feature},
```

```math
\mathrm{MaxMinS}
\longrightarrow
\text{two opposing extreme features},
```

```math
\mathrm{MAS}
\longrightarrow
\text{one attention-selected feature},
```

```math
\mathrm{AFS}
\longrightarrow
\text{one weighted first moment}.
```

Tier 2 then applies another weighted readout:

```math
\{\widehat f_{n,m}\}_{m=1}^{M}
\longrightarrow
F_n^{(2)}.
```

The architecture therefore preserves a small number of learned summaries, not
the full empirical distribution of patches. It can be more tractable while
discarding within-pseudo-bag diversity.

## 16. Distillation as a bias-variance choice

Suppose the ideal local statistic for pseudo-bag m is a latent positive
evidence vector mu_{n,m}. The distilled feature can be decomposed as:

```math
\widehat f_{n,m}
=
\mu_{n,m}
+
b_{n,m}
+
\varepsilon_{n,m},
```

where b is systematic selection bias and epsilon is selection or sampling
variation.

For AFS:

```math
\widehat f_{n,m}^{\mathrm{AFS}}
=
\sum_k\alpha_{n,m,k}h_{n,k}.
```

This can have lower selection variance than a hard maximum, but a sparse
positive signal can be diluted by many negative features. If the positive
patch has attention alpha-plus and all other features collectively have
attention one minus alpha-plus, then:

```math
\widehat f_{n,m}^{\mathrm{AFS}}
=
\alpha_{n,m,+}h_{n,m,+}
+
\left(
1-\alpha_{n,m,+}
\right)
\overline h_{n,m,-}.
```

The positive direction is attenuated whenever alpha-plus is small.

For MaxS:

```math
\widehat f_{n,m}^{\mathrm{MaxS}}
=
h_{n,k^{+}}.
```

The statistic does not average away a selected rare feature, but its error is
controlled by the probability that the ranking selects the wrong patch:

```math
\Pr
\left(
k^{+}
\notin
\mathcal P_{n,m}
\right),
```

where P_{n,m} is the set of truly positive patches in the pseudo-bag. A
maximum can have high variance and can select an artifact.

MaxMinS adds a contrasting feature. Its benefit depends on the minimum being
informative rather than pathological:

```math
\widehat f_{n,m}^{\mathrm{MaxMinS}}
=
\left[
h_{n,k^{+}}
\middle\Vert
h_{n,k^{-}}
\right].
```

The method gains a larger geometric span but also doubles the opportunity for
an extreme outlier to enter Tier 2.

## 17. AFS versus MaxS on a rare-positive pseudo-bag

Consider a pseudo-bag containing one positive feature h-plus and m minus one
negative features. Let the positive feature have attention alpha-plus.

AFS is:

```math
F^{(1)}
=
\alpha_{+}h_{+}
+
\sum_{k\ne +}
\alpha_k h_k.
```

If the negative weighted mean is h-bar-minus:

```math
F^{(1)}
=
\alpha_{+}h_{+}
+
\left(
1-\alpha_{+}
\right)
\overline h_{-}.
```

For a linear positive direction v, the scalar evidence is:

```math
v^{\mathsf T}F^{(1)}
=
\alpha_{+}v^{\mathsf T}h_{+}
+
\left(
1-\alpha_{+}
\right)
v^{\mathsf T}\overline h_{-}.
```

MaxS instead forwards h-plus only if the Tier-1 ranking is correct:

```math
v^{\mathsf T}
\widehat f^{\mathrm{MaxS}}
=
v^{\mathsf T}h_{+}.
```

This is the basic reason the methods can behave differently on sparse
positives. AFS preserves contextual mixture; MaxS preserves a candidate
extreme. Neither is uniformly superior.

## 18. Tier-2 attention is not an independent truth filter

Tier 2 receives only the distilled features:

```math
\widehat D_n
=
\{\widehat f_{n,1},\ldots,\widehat f_{n,M}\}.
```

It does not see the discarded patches unless the chosen distillation rule
encoded their information. Therefore:

```math
\widehat D_n
\ne
\{h_{n,k}\}_{k=1}^{K_n}
\quad
\text{in general}.
```

If two different slides produce the same distilled collection, Tier 2 must
produce the same output:

```math
\widehat D_n
=
\widehat D_{n'}
\Longrightarrow
\widehat y_n
=
\widehat y_{n'}.
```

This is a sufficient-statistic statement for the implemented second tier,
not a claim that the distilled collection is sufficient for the biological
label. A failed distillation can make information-theoretic recovery
impossible for Tier 2.

## 19. A two-slide collision

Let slide A and slide B have pseudo-bags with different patch distributions,
but suppose the Tier-1 MaxS rule selects the same feature from every
pseudo-bag:

```math
\widehat f_{A,m}^{\mathrm{MaxS}}
=
\widehat f_{B,m}^{\mathrm{MaxS}}
\quad
\text{for all }
m.
```

Then the Tier-2 input collections coincide:

```math
\widehat D_A
=
\widehat D_B.
```

No Tier-2 classifier can distinguish the slides from those inputs alone:

```math
\mathcal H_2(\widehat D_A)
=
\mathcal H_2(\widehat D_B).
```

This is not a flaw unique to DTFD. It is the unavoidable consequence of a
many-to-one readout. The useful question is whether the selected summary
retains the label-relevant information under the data distribution.

## 20. Attribution boundary

There are two learned attention systems:

```math
\alpha_{n,m,k}
\quad
\text{within pseudo-bag},
\qquad
\gamma_{n,m}
\quad
\text{across distilled pseudo-bags}.
```

A heatmap using alpha alone explains only the Tier-1 embedding. A heatmap
using gamma alone identifies a Tier-2 pseudo-bag, not an individual patch.

For AFS, a first-order path heuristic for a patch may combine both levels:

```math
\mathrm{credit}_{n,m,k}
\propto
\gamma_{n,m}
\alpha_{n,m,k}.
```

This is only a routing heuristic. A target-specific derivative must include the
Tier-1 class signal, the Tier-2 head, and the dependence of the distilled
feature on the patch:

```math
\frac{
\partial s_n^{(2)}
}{
\partial h_{n,k}
}
=
\sum_{m=1}^{M}
\frac{
\partial s_n^{(2)}
}{
\partial\widehat f_{n,m}
}
\frac{
\partial\widehat f_{n,m}
}{
\partial h_{n,k}
}.
```

For MaxS and MAS, the derivative through the selected index is piecewise and
does not represent the counterfactual effect of replacing the selected patch.
For MaxMinS, both selected paths must be included. For AFS, the weighted
aggregation path is differentiable except at any upstream score-ranking
operation used to create the attention.

The correct interpretation is therefore:

```math
\text{Tier-1 score}
\to
\text{distillation}
\to
\text{Tier-2 readout}
\to
\text{slide logit}.
```

Any explanation that stops before the final arrow is an explanation of an
intermediate mechanism, not necessarily of the final slide prediction.

## 21. Failure modes

### 21.1 Positive evidence is split away

If positive patches are rare, many inherited positive pseudo-bags contain no
positive patch. Tier 1 can learn an unstable mixture of disease and
slide-correlated background.

### 21.2 Random partition destroys local co-occurrence

The model cannot exploit physical adjacency unless coordinates or
coordinate-aware features enter the inputs. A pseudo-bag is not a tissue
compartment.

### 21.3 MaxS locks onto an artifact

One highly ranked patch controls the entire distilled feature. A stain
artifact or tissue edge can therefore be promoted to Tier 2.

### 21.4 MaxMinS imports a pathological minimum

The minimum class signal is not a certified negative. It can be an outlier or
an uninformative patch whose extreme score is caused by classifier
miscalibration.

### 21.5 MAS confuses attention with class evidence

Attention is trained to construct a bag embedding. It is not required to
localize the positive class, so MAS can select visually salient but
non-diagnostic tissue.

### 21.6 AFS dilutes sparse positives

The weighted first moment can retain much more negative context than positive
evidence when the attention model is diffuse.

### 21.7 Tier-2 attention is mistaken for causal importance

Gamma identifies which distilled pseudo-bag affects the Tier-2 embedding.
It does not prove that every patch inside that pseudo-bag caused the output.

### 21.8 Pseudo-bag count is treated as a free scaling parameter

Increasing M changes pseudo-bag size, label-noise probability, the number of
Tier-1 normalizations, and the granularity of the Tier-2 input. It changes the
statistical problem as well as the compute budget.

### 21.9 Partition randomness is hidden

Different random partitions can select different maxima and minima. A reported
result without partition seeds or repeated splits does not isolate model
variation from partition variation.

### 21.10 Patient count is overstated

NM pseudo-bags do not equal NM independent patient observations. Evaluation
must remain at the slide or patient level, with all pseudo-bags from a parent
slide kept in the same data split.

## 22. Sanity checks

### Check 1: label inheritance

For every positive slide, verify that the implementation can produce a
pseudo-bag with no positive patch under the latent diagnostic simulation:

```math
Y_n=1
\quad\text{and}\quad
\sum_{k\in\mathcal B_{n,m}}Z_{n,k}=0.
```

If the code or explanation says this event is impossible, it has silently
replaced DTFD's weak supervision with local supervision.

### Check 2: class denominator

For every patch signal, verify that the probability denominator sums over
classes:

```math
\sum_{t=1}^{C}
\exp(L_k^t).
```

An instance-index denominator would define a different normalization and
should not be attributed to DTFD.

### Check 3: attention versus probability

Construct two features with equal attention score but different class
directions:

```math
\alpha_a
=
\alpha_b,
\qquad
\left(\beta^1\right)^{\mathsf T}h_a
\ne
\left(\beta^1\right)^{\mathsf T}h_b.
```

The derived class signals can differ even though the attention weights match.

### Check 4: partition permutation

Hold the feature multiset fixed and change only the random partition. If the
distillation rule uses hard selection, the distilled collection may change:

```math
\pi_n
\ne
\pi_n'
\Longrightarrow
\widehat D_n(\pi_n)
\ne
\widehat D_n(\pi_n').
```

This sensitivity should be measured rather than described as noise-free
hierarchical structure.

### Check 5: MaxS and AFS collision

Use a pseudo-bag with one very positive patch and many mildly negative patches.
MaxS should forward the extreme feature, while AFS should retain a mixture.
If both outputs are identical, the implementation may be silently replacing
the requested distillation rule.

### Check 6: MaxMinS width

Verify the Tier-2 input dimension:

```math
\dim
\left(
\widehat f^{\mathrm{MaxS}}
\right)
=
D,
\qquad
\dim
\left(
\widehat f^{\mathrm{MaxMinS}}
\right)
=
2D.
```

An explicit projection is required if the same Tier-2 head is reused.

### Check 7: deletion at both levels

Delete a selected patch and recompute the full pipeline. Then delete an entire
pseudo-bag and recompute the full pipeline. The two interventions test
different mechanisms:

```math
\Delta_{n,k}
=
s_n^{(2)}
-
s_{n,-k}^{(2)},
```

```math
\Delta_{n,m}^{\mathrm{bag}}
=
s_n^{(2)}
-
s_{n,-m}^{(2)}.
```

The first tests patch routing and distillation. The second tests Tier-2
pseudo-bag reliance.

### Check 8: slide-level split

No pseudo-bag from a parent slide may appear in a different train, validation,
or test split:

```math
\mathrm{split}(n,m)
=
\mathrm{split}(n,m')
\quad
\text{for all }
m,m'.
```

Otherwise the virtual bag multiplication creates leakage.

## 23. C/R/G/S placement

| Component | DTFD-MIL interpretation |
| --- | --- |
| Context operator C | Tier 1 aggregates patches inside each random pseudo-bag; Tier 2 aggregates distilled pseudo-bag features across the slide |
| Readout operator R | Tier-1 AB-MIL creates a weighted first moment, then MaxS, MaxMinS, MAS, or AFS distills the feature, followed by Tier-2 AB-MIL |
| Geometry operator G | Vanilla DTFD uses random partition membership; it does not encode physical adjacency or a learned topology |
| Supervision operator S | Slide labels are copied to pseudo-bags; Grad-CAM-derived class signals and hard distillation are generated internal supervision/routing statistics |

The decomposition exposes why DTFD is hybrid but not automatically spatial:

```math
\mathrm{DTFD}
=
\underbrace{
\mathrm{ABMIL}_{\mathrm{local}}
}_{C,R}
\circ
\underbrace{
\mathrm{Distill}_{\mathrm{MaxS/MaxMinS/MAS/AFS}}
}_{R,S}
\circ
\underbrace{
\mathrm{ABMIL}_{\mathrm{global}}
}_{C,R}.
```

The random partition belongs to the computational geometry of the method, not
to the slide's physical geometry.

## 24. Comparison with neighboring operators

### Flat AB-MIL

Flat AB-MIL computes one weighted first moment over all K_n patches:

```math
F_n^{\mathrm{flat}}
=
\sum_{k=1}^{K_n}
\alpha_{n,k}h_{n,k}.
```

DTFD inserts a partition, a local readout, a distillation operator, and a
second readout:

```math
\{h_{n,k}\}_{k=1}^{K_n}
\to
\{F_{n,m}^{(1)}\}_{m=1}^{M}
\to
\{\widehat f_{n,m}\}_{m=1}^{M}
\to
F_n^{(2)}.
```

The extra stages can alter the inductive bias even when both tiers use the
same AB-MIL attention formula.

### Hierarchical spatial MIL

A spatial hierarchy defines parent-child groups using coordinates or regions:

```math
\mathcal B_{n,m}
=
\mathrm{RegionAssign}
\left(
\{c_{n,k}\}
\right).
```

DTFD uses a random allocation:

```math
\mathcal B_{n,m}
=
\mathrm{RandomAssign}
\left(
\{1,\ldots,K_n\}
\right).
```

Calling both “hierarchical” hides the geometry difference.

### Max pooling

Max pooling keeps an extreme score directly:

```math
\mathrm{MaxPool}
\left(
\{r_k\}
\right)
=
\max_k r_k.
```

MaxS keeps the feature associated with a class-derived extreme and then lets
Tier 2 learn another readout:

```math
\mathrm{MaxS}
\left(
\{h_k,p_k^1\}
\right)
=
h_{\arg\max_k p_k^1}.
```

The second tier is what separates DTFD from simply classifying the maximum
patch score.

### Prototype or distribution pooling

Prototype pooling retains distances to learned anchors, and distribution
pooling retains moments or transport summaries. DTFD does neither in its
vanilla form:

```math
\widehat D_n
=
\{\widehat f_{n,m}\}_{m=1}^{M}
\quad
\text{rather than}
\quad
\{\mathrm{dist}(h_{n,k},q_r)\}_{r}
\text{ or }
\mathrm{MeanMeasure}(H_n).
```

AFS is a local weighted mean, but the overall model is not a distribution
representation because it discards the rest of each pseudo-bag before Tier 2.

## 25. What DTFD actually contributes

The method's mathematical move can be summarized as:

```math
\text{one enormous weakly labeled bag}
\longrightarrow
\text{many noisy local bags}
\longrightarrow
\text{one distilled feature per local bag}
\longrightarrow
\text{a second weakly supervised bag}.
```

The useful inductive biases are:

1. smaller Tier-1 normalization domains;
2. more optimization units derived from the same slide labels;
3. a class-conditioned feature-selection mechanism;
4. a second attention readout over local summaries.

The costs are:

1. inherited pseudo-bag label noise;
2. random rather than anatomical grouping;
3. hard-selection instability for MaxS, MaxMinS, and MAS;
4. information loss before Tier 2;
5. dependence between pseudo-bags from the same parent slide.

The paper's empirical comparison of MaxS, MaxMinS, MAS, and AFS is therefore
mathematically meaningful: the variants differ in what statistic survives the
Tier-1 to Tier-2 interface.

## 26. Bottom line

DTFD-MIL is not merely AB-MIL applied twice. Its defining object is the
pseudo-bag interface:

```math
\widehat f_{n,m}
=
\Phi_{\mathrm{distill}}
\left(
\{h_{n,k}\}_{k\in\mathcal B_{n,m}},
\{\alpha_{n,m,k}\},
\{p_{n,m,k}^{1}\}
\right).
```

The parent label is copied downward, but local truth is not known. The Tier-1
model generates class-sensitive or attention-sensitive candidates, and Tier 2
learns how to combine those candidates into a slide prediction.

In the C/R/G/S language:

```math
\boxed{
\text{random pseudo-bag context}
+
\text{AB-MIL and distillation readout}
+
\text{no physical geometry}
+
\text{copied weak supervision}
}
```

The right explanation of DTFD is consequently neither “attention finds the
positive patches” nor “the hierarchy models slide anatomy.” It is a
two-level, weakly supervised set readout whose crucial bottleneck is the
paper-specific distillation rule.
