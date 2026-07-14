# ABMIL as a Gated Attention Measure

## 1. Scope and source

Attention-based deep multiple instance learning replaces the uniform first
moment with a learned weighted first moment. The original ABMIL construction
of Ilse et al. uses a gated attention score for each instance:

https://arxiv.org/abs/1802.04712

For WSI, this is a set-level model. It can rank patches, but its weights are
routing coefficients in a predictive computation, not automatically causal
evidence or a calibrated probability that a patch is diseased.

This note derives:

- the gated score and normalized empirical measure;
- permutation invariance and the mean-pooling subcase;
- gradients through both value and routing paths;
- effective support and temperature;
- prevalence, bag-size, and attention-collapse failure modes;
- class-specific extensions and C/R/G/S placement.

## 2. Bag object

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

ABMIL applies an instance embedding and scalar score before aggregation. Let
the attention hidden width be r:

```math
V\in\mathbb R^{r\times d},
\qquad
U\in\mathbb R^{r\times d},
\qquad
w\in\mathbb R^r.
```

The original gated score for patch j is

```math
s_{ij}
=
w^{\mathsf T}
\left[
\tanh(Vh_{ij})
\odot
\mathrm{sigm}(Uh_{ij})
\right].
```

The two branches have different roles. The tanh branch supplies a signed
content feature. The sigmoid branch gates that feature coordinatewise. Their
Hadamard product is then projected to one scalar.

## 3. Normalized attention measure

The attention weight is

```math
\alpha_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{\ell=1}^{n_i}\exp(s_{i\ell})}.
```

The weights form a probability vector:

```math
\alpha_{ij}\ge 0,
\qquad
\sum_{j=1}^{n_i}\alpha_{ij}=1.
```

Define the learned empirical attention measure

```math
\widehat\nu_i^\theta
=
\sum_{j=1}^{n_i}\alpha_{ij}\delta_{h_{ij}}.
```

The ABMIL slide representation is its first moment:

```math
z_i
=
\int x\,d\widehat\nu_i^\theta(x)
=
\sum_{j=1}^{n_i}\alpha_{ij}h_{ij}.
```

The complete predictor is

```math
\widehat y_i
=
\psi_\theta(z_i).
```

The attention network changes the measure before taking the moment. Mean
pooling uses the uniform empirical measure; ABMIL learns a bag-dependent
reweighting of that measure.

## 4. Permutation invariance

Let P_i be a permutation matrix. The score vector is rowwise:

```math
s(P_iH_i)
=
P_i s(H_i).
```

Softmax commutes with the same permutation:

```math
\alpha(P_iH_i)
=
P_i\alpha(H_i).
```

The weighted sum is invariant:

```math
\sum_j\alpha_j(P_iH_i)(P_iH_i)_j
=
\sum_j\alpha_j(H_i)h_{ij}.
```

Thus ABMIL is a set model as long as the patch score is pointwise and no
order-dependent preprocessing enters the feature matrix.

Coordinates can be appended to h or used by the patch encoder, but then the
model remains permutation invariant over coordinate-feature pairs, not
invariant to coordinate transformations.

## 5. Mean pooling as a parameter subcase

If every score is equal within a bag,

```math
s_{i1}=\cdots=s_{in_i},
```

then

```math
\alpha_{ij}
=
\frac{1}{n_i}
```

and

```math
z_i
=
\frac{1}{n_i}\sum_jh_{ij}.
```

Equal scores can occur through zero attention parameters or through a feature
region where the gated score is constant. Mean pooling is therefore contained
in the ABMIL family, but generic ABMIL is not mean pooling.

The distinction is a learned density ratio. Relative to the uniform measure,
the weight multiplier is

```math
\frac{d\widehat\nu_i^\theta}{d\widehat\mu_i}(h_{ij})
=
n_i\alpha_{ij}.
```

The average multiplier equals one:

```math
\frac{1}{n_i}\sum_j n_i\alpha_{ij}=1.
```

Attention does not change the number of patches. It changes their normalized
mass.

## 6. Gating geometry

Define the ungated and gate vectors:

```math
a_{ij}
=
\tanh(Vh_{ij}),
\qquad
g_{ij}
=
\mathrm{sigm}(Uh_{ij}).
```

Then

```math
s_{ij}
=
w^{\mathsf T}(a_{ij}\odot g_{ij}).
```

The score gradient with respect to a patch is

```math
\nabla_{h_{ij}}s_{ij}
=
V^{\mathsf T}
\left[
\left(1-a_{ij}^{\odot 2}\right)
\odot
g_{ij}
\odot
w
\right]
+
U^{\mathsf T}
\left[
a_{ij}
\odot
g_{ij}
\odot
\left(1-g_{ij}\right)
\odot
w
\right].
```

The first term moves the content branch. The second moves the gate branch.
When tanh saturates, the first derivative becomes small. When sigmoid
saturates near zero or one, the gate derivative becomes small. A patch can
therefore receive a low score gradient because of saturation even if its
attention weight is high.

The scalar score is bounded in magnitude by

```math
|s_{ij}|
\le
\|w\|_1
```

because tanh and sigmoid are bounded, although the exact bound can be tighter
when the two branches are coupled.

## 7. Attention as a soft selection distribution

The log weight has the form

```math
\log\alpha_{ij}
=
s_{ij}
-
\log\sum_{\ell=1}^{n_i}\exp(s_{i\ell}).
```

The log-sum-exp term couples all patches. Adding the same constant c to every
score leaves the weights unchanged:

```math
\alpha_{ij}(s_i+c\mathbf 1)
=
\alpha_{ij}(s_i).
```

Only relative score differences matter. A slide-specific shift in the score
network is not identifiable from the weights.

For two patches:

```math
\frac{\alpha_{i1}}{\alpha_{i2}}
=
\exp(s_{i1}-s_{i2}).
```

An attention weight is therefore a relative ranking quantity. It is not an
absolute confidence without a specified comparison set.

## 8. Temperature and effective support

Introduce temperature tau:

```math
\alpha_{ij}(\tau)
=
\frac{\exp(s_{ij}/\tau)}
{\sum_{\ell}\exp(s_{i\ell}/\tau)}.
```

As tau tends to infinity:

```math
\alpha_{ij}(\tau)
\longrightarrow
\frac{1}{n_i}.
```

As tau tends to zero, mass concentrates on score maximizers:

```math
\alpha_i(\tau)
\longrightarrow
\mathrm{Uniform}\!\left(
\mathrm{ArgMax}(s_i)
\right)
```

when ties are handled symmetrically.

The attention entropy is

```math
\mathcal H_i(\tau)
=
-\sum_j\alpha_{ij}(\tau)\log\alpha_{ij}(\tau).
```

The effective sample size is

```math
n_{\mathrm{eff},i}(\tau)
=
\frac{1}{\sum_j\alpha_{ij}(\tau)^2}.
```

Mean pooling has n_eff equal to n_i. A one-patch attention collapse has n_eff
equal to one. Reporting only the top patch hides this continuum.

## 9. Gradient of the weights

The softmax Jacobian is

```math
\frac{\partial\alpha_{ij}}{\partial s_{i\ell}}
=
\alpha_{ij}
\left(
\mathbf 1\{j=\ell\}
-
\alpha_{i\ell}
\right).
```

The derivative of the pooled representation with respect to one score is

```math
\frac{\partial z_i}{\partial s_{ij}}
=
\alpha_{ij}
\left(h_{ij}-z_i\right).
```

This equation is central. Increasing a patch score moves the representation
toward that patch relative to the current attention mean. A high weight does
not imply a high marginal effect if h_ij is already close to z_i.

The full derivative with respect to patch h_ij has a direct value path and a
routing path:

```math
\frac{\partial z_i}{\partial h_{ij}}
=
\alpha_{ij}I_d
+
\alpha_{ij}
\left(h_{ij}-z_i\right)
\left(\nabla_{h_{ij}}s_{ij}\right)^{\mathsf T}
+
\sum_{\ell\ne j}
\frac{\partial z_i}{\partial s_{i\ell}}
\frac{\partial s_{i\ell}}{\partial h_{ij}}.
```

For a pointwise score, the final sum vanishes for ell not equal to j. The
softmax denominator is already represented in the h_ij minus z_i term. If
contextual patch scores are used, cross-patch terms reappear.

## 10. Attention weight versus predictive credit

Let F_i be a scalar target. The target derivative is

```math
\frac{\partial F_i}{\partial h_{ij}}
=
\left(\nabla_{z_i}F_i\right)^{\mathsf T}
\frac{\partial z_i}{\partial h_{ij}}
```

before accounting for any downstream stochastic or normalization operation.

The direct coefficient alpha_ij is nonnegative. The target derivative can be
signed:

```math
\left(\nabla_{z_i}F_i\right)^{\mathsf T}
\left(h_{ij}-z_i\right)
\quad\text{can be positive or negative}.
```

A high-weight patch can reduce a target logit if its value points against the
head direction. A low-weight patch can have large score sensitivity if moving
it changes the denominator or the pooled direction.

The correct statement is:

```math
\text{ABMIL weight}
=
\text{normalized routing mass}
\ne
\text{signed target contribution}.
```

Deletion, gradients, and attention weights should be compared rather than
identified.

## 11. Rare-positive behavior

Let one positive patch p compete with n_i minus one background patches b, with
scores s_p and s_b. Its attention weight is

```math
\alpha_p
=
\frac{\exp(s_p)}
{\exp(s_p)+(n_i-1)\exp(s_b)}.
```

Define the score gap delta equal to s_p minus s_b. Then

```math
\alpha_p
=
\frac{\exp(\Delta)}
{\exp(\Delta)+n_i-1}.
```

To keep alpha_p above a target q, the score gap must satisfy

```math
\Delta
\ge
\log\left(
\frac{q(n_i-1)}{1-q}
\right).
```

The required separation grows like log n_i. A larger WSI makes a fixed score
gap less selective because the softmax denominator contains more background.

If r_i positive patches share score s_p:

```math
\alpha_{P}
=
\frac{r_i\exp(s_p)}
{r_i\exp(s_p)+(n_i-r_i)\exp(s_b)}.
```

The relevant quantity is total positive mass, not the weight of one patch.

## 12. Background multiplicity

Duplicate every background patch r times while leaving the positive patch
unchanged. The positive weight becomes

```math
\alpha_p^{(r)}
=
\frac{\exp(s_p)}
{\exp(s_p)+r(n_i-1)\exp(s_b)}.
```

Thus attention is not invariant to uniform replication of only one population.
It is invariant to uniform replication of every patch because all weights and
the value sum scale together, but composition changes alter the measure.

This creates a WSI size and tissue-fraction shortcut. A slide with more
background tokens can downweight a positive patch even when its local feature
is unchanged.

## 13. Attention collapse and diffuse attention

### 13.1 Collapse

If one score dominates by a large gap:

```math
s_{i j^\star}-s_{i\ell}\to+\infty
\quad\text{for all }\ell\ne j^\star,
```

then

```math
\alpha_{ij^\star}\to 1,
\qquad
\alpha_{i\ell}\to 0.
```

The representation approaches h_i j-star. This can behave like a learned max
operator and can be useful for a single critical instance. It is brittle to
artifacts, duplicate patches, and score noise.

### 13.2 Diffusion

If scores are nearly equal:

```math
\max_j|s_{ij}-\bar s_i|\to 0,
```

then alpha approaches the uniform mean. Rare-positive evidence is diluted and
the model loses its adaptive advantage.

### 13.3 Score saturation

The gating branches can saturate, reducing score gradients:

```math
\left|\tanh(Vh_{ij})\right|\to 1
\quad\text{or}\quad
\mathrm{sigm}(Uh_{ij})\to 0,1.
```

The attention weights can look decisive even while the score network is hard
to update.

## 14. Head interaction

Let the task head be linear:

```math
F_i
=
w_{\mathrm{cls}}^{\mathsf T}z_i+b.
```

Then

```math
F_i
=
\sum_j\alpha_{ij}
\left(w_{\mathrm{cls}}^{\mathsf T}h_{ij}\right)+b.
```

The score network controls routing while the classifier controls target
direction. The patch contribution is the product of both:

```math
c_{ij}
=
\alpha_{ij}
\left(w_{\mathrm{cls}}^{\mathsf T}h_{ij}\right).
```

Ranking by alpha and ranking by c can disagree. A high attention patch with a
negative class projection can suppress the target.

For a nonlinear head, this additive equation no longer holds globally, though
it remains a local linearization around z_i.

## 15. Class-specific attention

For C classes, use one score vector per class:

```math
s_{ij}^{(c)}
=
\left(w^{(c)}\right)^{\mathsf T}
\left[
\tanh(V^{(c)}h_{ij})
\odot
\mathrm{sigm}(U^{(c)}h_{ij})
\right].
```

The class-specific weights are

```math
\alpha_{ij}^{(c)}
=
\frac{\exp(s_{ij}^{(c)})}
{\sum_{\ell}\exp(s_{i\ell}^{(c)})}.
```

This is a different model from one shared attention vector followed by a
multiclass head. It can represent different instance rankings for different
targets, but it increases parameter count and can produce incompatible
heatmaps across classes.

The class-specific representation is

```math
z_i^{(c)}
=
\sum_j\alpha_{ij}^{(c)}h_{ij}.
```

If a shared head is applied to every z_i^(c), the class index enters through
the representation. If each class has its own head, the class-specific routing
and readout are both target-dependent.

## 16. Attention as a learned measure change

The uniform empirical measure is

```math
\widehat\mu_i
=
\frac{1}{n_i}\sum_j\delta_{h_{ij}}.
```

The attention measure is

```math
\widehat\nu_i^\theta
=
\sum_j\alpha_{ij}\delta_{h_{ij}}.
```

The Radon-Nikodym ratio on observed atoms is

```math
r_{ij}
=
\frac{d\widehat\nu_i^\theta}{d\widehat\mu_i}(h_{ij})
=
n_i\alpha_{ij}.
```

The weighted first moment is

```math
z_i
=
\mathbb E_{\widehat\nu_i^\theta}[X].
```

The second moment under the learned measure is

```math
M_i^{(2)}
=
\sum_j\alpha_{ij}h_{ij}h_{ij}^{\mathsf T}.
```

ABMIL still returns only one first moment of the reweighted distribution. It
does not preserve the complete attention-weighted distribution unless more
statistics are added to the head.

## 17. Relation to set-function universality

A Deep Sets representation has the form

```math
f(\{h_j\})
=
\rho\!\left(\sum_j\phi(h_j)\right).
```

ABMIL has a normalized, data-dependent weighting:

```math
f_{\mathrm{ABMIL}}(H)
=
\rho\!\left(
\frac{\sum_j\exp(s(h_j))h_j}
{\sum_j\exp(s(h_j))}
\right).
```

The normalization removes total score mass. Two bags with the same normalized
weighted moment can still collide even if their unnormalized sums differ.

ABMIL is expressive as a set function, but its particular parameterization
chooses one weighted vector as the interface to the head. Universality of a
broader set-function class is not a guarantee that one finite-width ABMIL
instance is injective or task-sufficient.

## 18. Coordinates and geometry

If the score is pointwise in h only, a coordinate permutation paired with the
same feature permutation leaves the output unchanged:

```math
f_{\mathrm{ABMIL}}(PH_i,PC_i)
=
f_{\mathrm{ABMIL}}(H_i,C_i)
```

If coordinates are concatenated:

```math
\widetilde h_{ij}
=
\begin{bmatrix}
h_{ij}\\
\varphi(c_{ij})
\end{bmatrix},
```

the model can learn absolute coordinate-dependent scores. It still does not
see pairwise adjacency unless adjacency is encoded in h or in the score.

If the patch encoder contains a local convolutional context, ABMIL's
pointwise score is applied to contextualized features. The bag-level operator
remains a weighted first moment, but the instance vector already contains
information from a neighborhood.

## 19. Sanity checks

### Check A: uniform-score recovery

Set attention parameters to produce equal scores and verify

```math
\left\|
z_i-\frac{1}{n_i}\sum_jh_{ij}
\right\|_2
\approx 0.
```

This checks the mean-pooling subcase.

### Check B: permutation recovery

Randomly permute patch rows and verify the output and weight multiset:

```math
f(PH_i)=f(H_i),
\qquad
\mathrm{sort}(\alpha(PH_i))
=
\mathrm{sort}(\alpha(H_i)).
```

The ordered heatmap should permute with the patches.

### Check C: score-gap sweep

Construct one positive and n minus one background patches with controlled gap
delta. Compare observed positive mass with

```math
\alpha_p(\Delta)
=
\frac{\exp(\Delta)}
\exp(\Delta)+n-1.
```

This reveals how bag size changes the selection threshold.

### Check D: entropy and effective support

Report entropy, participation ratio, top-k cumulative mass, and the maximum
weight. A single top-patch map cannot distinguish collapse from broad support.

### Check E: weight versus deletion

Rank patches by alpha and by target deletion effect. Large disagreement is
expected when value directions or interactions matter.

### Check F: duplication stress

Duplicate background patches, positive patches, and all patches separately.
The three curves identify composition sensitivity versus uniform replication
invariance.

### Check G: saturation gradients

Measure the norm of the content and gate score gradients. High attention with
near-zero score gradient indicates saturation, not necessarily robust evidence.

## 20. C/R/G/S placement

| Component | ABMIL realization |
|---|---|
| Context operator | Pointwise gated score network; no patch-patch context at the bag level. |
| Readout operator | Softmax-normalized weighted first moment, with optional class-specific score branches. |
| Geometry | Permutation invariant over patch-feature pairs; physical arrangement is absent unless encoded into features or scores. |
| Surviving statistic | One learned attention-weighted mean; effective support, score entropy, and target direction determine which instances dominate it. |

The forward map is

```math
\{h_{i1},\ldots,h_{in_i}\}
\xrightarrow{\text{gated score}}
\{s_{ij}\}
\xrightarrow{\text{softmax}}
\{\alpha_{ij}\}
\xrightarrow{\text{weighted first moment}}
z_i
\xrightarrow{\text{head}}
\widehat y_i.
```

The key comparison with mean pooling is

```math
\text{uniform measure}
\longrightarrow
\text{learned measure}
\longrightarrow
\text{one surviving moment}.
```

## 21. Bottom line

ABMIL is a permutation-invariant learned reweighting of the empirical patch
measure. Its gated score can focus on rare evidence and reduce background
dilution, but the softmax denominator makes attention competition depend on bag
size and composition. The representation is still one weighted first moment.

Attention weights are nonnegative routing mass. Predictive credit is signed and
depends on the value direction, classifier, denominator redistribution, and
any contextual encoder upstream. A convincing WSI explanation therefore pairs
attention with target-specific deletion or gradient tests.

The honest summary is:

```math
\boxed{
\text{ABMIL}
=
\text{gated pointwise score}
\;+\;
\text{normalized measure change}
\;+\;
\text{weighted first moment}
}
```

It is more adaptive than mean pooling but still has a sharp information
bottleneck: all target-relevant evidence must survive in one attention-weighted
vector.

## References

- Ilse et al., "Attention-based Deep Multiple Instance Learning,"
  https://arxiv.org/abs/1802.04712
- Campanella et al., "Clinical-grade computational pathology using weakly
  supervised deep learning on whole slide images," Nature Medicine 2019.
  https://www.nature.com/articles/s41591-019-0508-1
- Zaheer et al., "Deep Sets," https://arxiv.org/abs/1703.06114
- Jain and Wallace, "Attention is not Explanation,"
  https://arxiv.org/abs/1902.10186
