# Transformer MIL Complexity and Sampling

## 1. Scope

Whole-slide learning is constrained by two different budgets:

1. the computational cost of contextualizing a large bag;
2. the statistical cost of seeing only part of that bag.

These budgets are often discussed together under the word efficient, but they
are mathematically different. A low-rank attention approximation can preserve
all input tokens while changing pairwise interactions. Patch sampling can leave
the attention operator unchanged while changing the distribution of evidence
that reaches it.

The central question is:

> When a transformer-MIL model is made affordable, which interactions and which
> instances survive?

The main pathology anchor is TransMIL:

https://arxiv.org/abs/2106.00908

The sampling comparisons use the paper map's anchors:

- Campanella et al., clinical-scale WSI MIL:
  https://www.nature.com/articles/s41591-019-0508-1
- Cluster-to-Conquer:
  https://arxiv.org/abs/2103.10626
- Zhu et al., How Effective Can Dropout Be in Multiple Instance Learning?,
  ICML 2025:
  https://proceedings.mlr.press/v267/zhu25q.html
- Wu et al., Distributed Parallel Gradient Stacking:
  https://openreview.net/forum?id=ss5JNmJDkW

The goal is to derive the cost and estimator rather than call every reduction
linear, sparse, or unbiased without specifying the relevant quantity.

## 2. Bag and sequence notation

Let slide i contain n_i patch embeddings of width d:

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

For a class-token transformer, the sequence length is

```math
L_i=n_i+1.
```

The class row is included in attention cost but is negligible relative to n_i
for very large bags. A variable-length WSI batch contains lengths
L_1 through L_B. Padding all slides to the maximum length L_max creates

```math
\mathrm{waste}
=
\sum_{i=1}^{B}
\left(L_{\max}-L_i\right).
```

The useful token fraction is

```math
\mathrm{Util}
=
\frac{\sum_{i=1}^{B}L_i}
{B\,L_{\max}}.
```

Low utilization can dominate the theoretical attention saving. Packed
sequences, bucketing, or distributed gradient stacking change this systems
term without changing the model's mathematical context operator.

## 3. Exact self-attention cost

### 3.1 Projection cost

For one head, the query, key, and value projections cost

```math
\mathcal O(L_i d d_h)
\quad\text{each}.
```

For H heads with H d_h equal to d, all projections together cost

```math
\mathcal O(L_i d^2).
```

The output projection has the same order.

### 3.2 Score and value-mixing cost

The score matrix is

```math
S_i
=
\frac{Q_iK_i^{\mathsf T}}{\sqrt{d_h}}
\in\mathbb R^{L_i\times L_i}.
```

Its multiplication costs

```math
\mathcal O(L_i^2d).
```

Multiplying the normalized score matrix by values has the same order:

```math
\mathcal O(L_i^2d).
```

Thus one dense attention block has leading arithmetic cost

```math
\mathcal O(L_i d^2+L_i^2d).
```

When L_i is much larger than d, the quadratic term dominates.

### 3.3 Memory cost

Storing the attention logits or probabilities for one head costs

```math
\mathcal O(L_i^2).
```

Across H heads:

```math
\mathcal O(HL_i^2).
```

Backpropagation may retain projections, logits, softmax intermediates, and
activation states. The training memory constant is therefore substantially
larger than the inference memory required for one output.

The exact statement is:

```math
\mathrm{Cost}_{\mathrm{exact}}
=
\mathcal O(L_i d^2+L_i^2d),
\qquad
\mathrm{Memory}_{\mathrm{exact}}
=
\mathcal O(HL_i^2+L_i d).
```

The first term in memory is the WSI bottleneck.

## 4. TransMIL square completion cost

TransMIL chooses

```math
N_i
=
\left\lceil\sqrt{n_i}\right\rceil,
\qquad
\delta_i=N_i^2-n_i,
\qquad
L_i^{\mathrm{TransMIL}}
=
N_i^2+1.
```

The repeated-token completion does not change the asymptotic order, but it
changes the constant and the exact sequence length. The attention inflation
factor relative to a hypothetical no-padding sequence is

```math
\rho_i^{\mathrm{pad}}
=
\frac{(N_i^2+1)^2}{(n_i+1)^2}.
```

Since

```math
0\le\delta_i<2\sqrt{n_i}+1,
```

the relative padding fraction vanishes asymptotically, but for small bags it
can be material. More importantly, the repeated rows are data-bearing tokens,
not empty slots, so the cost inflation is also a multiplicity change.

If the final remainder is repeated from the first tokens, the number of times
a token appears in the completed sequence is

```math
m_{ij}
=
1+\mathbf 1\{j\le\delta_i\},
```

in the common case where delta is smaller than n_i. The attention operator sees
this multiplicity.

## 5. PPEG cost

For a square grid of side N_i and width d, a depthwise convolution with kernel
K has cost

```math
\mathcal O(N_i^2 d K^2).
```

The TransMIL PPEG module uses kernels 7, 5, and 3:

```math
\mathrm{Cost}_{\mathrm{PPEG}}
=
\mathcal O\!\left(
N_i^2d(7^2+5^2+3^2)
\right).
```

This is linear in the number of completed patch tokens. It is not the
quadratic bottleneck. PPEG can nevertheless change the output for every patch
near an altered or repeated cell, so low arithmetic cost does not mean low
statistical effect.

## 6. Nyström attention cost

### 6.1 Factorization

Let m be the number of landmarks. TransMIL forms factors with shapes

```math
A_{L,m}\in\mathbb R^{L_i\times m},
\qquad
A_{m,m}\in\mathbb R^{m\times m},
\qquad
A_{m,L}\in\mathbb R^{m\times L_i}.
```

The approximate mixing matrix is

```math
\widehat A
=
A_{L,m}A_{m,m}^{+}A_{m,L}.
```

### 6.2 Arithmetic

The three score products and value mixing have leading order

```math
\mathcal O(L_i m d+m^2d).
```

An iterative pseudoinverse with T iterations adds

```math
\mathcal O(Tm^3)
```

in the dense landmark core. The full per-head cost is therefore more honestly
written as

```math
\mathrm{Cost}_{\mathrm{Nys}}
=
\mathcal O(L_i d^2+L_i m d+m^2d+Tm^3).
```

For fixed m, d, and T, this is linear in L_i. If m is chosen proportional to
L_i, the quadratic advantage disappears. If m is too small, statistical
approximation error can dominate.

### 6.3 Memory

The attention factors require

```math
\mathcal O(L_i m+m^2)
```

per head rather than O(L_i^2). This is the main reason the approximation makes
larger bags possible.

### 6.4 Landmark budget as a representation bottleneck

The landmark count is not only a runtime knob. It defines the interface through
which global values are mixed:

```math
\widehat O
=
A_{L,m}
\left(A_{m,m}^{+}A_{m,L}V\right).
```

The center term is an m-row summary. A rare pattern that is weakly represented
in this summary can be lost before the class token reads it.

## 7. Windowed and hierarchical comparisons

For a windowed attention partition with W windows of size w, the score cost is

```math
\mathcal O(Ww^2d)
=
\mathcal O(L_iwd).
```

This is linear in L_i for fixed window size, but one layer has no direct
cross-window edges. Shifted windows or cross-window pooling are required for
global reach.

For a hierarchy with R regions and average m patches per region, local attention
plus region attention costs approximately

```math
\mathcal O(Rm^2d+R^2d).
```

The comparison to flat attention is

```math
Rm^2+R^2<n_i^2
```

only when the region partition is sufficiently favorable. A badly chosen
partition or a region count close to n_i can remove the saving. A hierarchy
also changes the surviving statistic by compressing local groups before global
context is formed.

For a graph with E edges, message passing has approximate cost

```math
\mathcal O(Ed)
```

per layer, but the number of layers required for long-range reach depends on
graph diameter and oversquashing. Complexity alone does not identify the
preserved information.

## 8. Sampling as a change of object

Let the full bag be a finite population

```math
\mathcal U_i
=
\{1,\ldots,n_i\}.
```

Let I_i be a sampled subset of size m_i. The sampled input is

```math
H_i^{(I)}
=
\{h_{ij}:j\in I_i\}.
```

The full model computes

```math
F_i^{\mathrm{full}}
=
F_\theta(H_i,C_i),
```

while the sampled pipeline computes

```math
F_i^{\mathrm{sample}}
=
F_\theta(H_i^{(I)},C_i^{(I)}).
```

In general,

```math
\mathbb E_I\!\left[F_i^{\mathrm{sample}}\right]
\ne
F_i^{\mathrm{full}}.
```

This is not a failure of the sampler. It is the expected consequence of
applying a nonlinear function to a random subset. The sampled model has a
different input distribution and potentially a different inductive bias.

## 9. Uniform sampling and capture probability

### 9.1 Without replacement

Uniform sampling without replacement selects m patches from n. If a slide
contains r positive patches, the probability of selecting at least one is

```math
\mathbb P(\mathrm{capture})
=
1-
\frac{\binom{n-r}{m}}{\binom{n}{m}}.
```

For r much smaller than n and m much smaller than n,

```math
\mathbb P(\mathrm{capture})
\approx
1-\exp\!\left(-\frac{mr}{n}\right).
```

The expected number of positive patches is

```math
\mathbb E[R_I]
=
m\frac{r}{n}.
```

If r/n is extremely small, increasing m may be expensive before capture
probability becomes reliable. Uniform sampling is therefore a poor default for
rare positives unless the required coverage is quantified.

### 9.2 With replacement

Independent sampling with replacement has capture probability

```math
\mathbb P(\mathrm{capture})
=
1-\left(1-\frac{r}{n}\right)^m.
```

Repeated samples can waste budget and create multiplicity shortcuts. Without
replacement is usually preferable for coverage when the full candidate pool is
available.

### 9.3 Stratified sampling

Partition the bag into strata S_1 through S_G, such as tissue regions or
clusters. Sample m_g tokens from stratum g. The capture probability for a
positive stratum is controlled by its allocation:

```math
\mathbb P(\mathrm{capture\ in\ }g)
=
1-
\frac{\binom{|S_g|-r_g}{m_g}}{\binom{|S_g|}{m_g}}.
```

Stratification improves coverage only when the strata correlate with the
target-relevant signal. A bad tissue filter can make the estimator confidently
wrong.

## 10. Unbiased estimators for linear pooling

Sampling can be unbiased for a linear population statistic if inclusion
probabilities are used correctly. Let pi_ij be the inclusion probability of
patch j. The Horvitz-Thompson estimator of a population sum is

```math
\widehat T_i^{\mathrm{HT}}
=
\sum_{j\in I_i}
\frac{h_{ij}}{\pi_{ij}}.
```

For the population mean:

```math
\widehat\mu_i^{\mathrm{HT}}
=
\frac{1}{n_i}
\sum_{j\in I_i}
\frac{h_{ij}}{\pi_{ij}}.
```

Its expectation equals the full mean under known positive inclusion
probabilities:

```math
\mathbb E_I\!\left[\widehat\mu_i^{\mathrm{HT}}\right]
=
\frac{1}{n_i}\sum_{j=1}^{n_i}h_{ij}.
```

The variance can be large when pi_ij is small:

```math
\mathrm{Var}\!\left(\widehat\mu_i^{\mathrm{HT}}\right)
\propto
\sum_j\frac{1-\pi_{ij}}{\pi_{ij}}
\|h_{ij}-\mu_i\|_2^2
\quad\text{under simple designs}.
```

For a nonlinear transformer, applying inverse-probability weights to tokens
does not generally recover the full-transformer output. It changes the
operator and may create unstable logits.

## 11. Attention pooling under sampling

For a full bag, a scalar attention pooling readout is

```math
z_i
=
\frac{\sum_{j=1}^{n_i}\exp(s_{ij})v_{ij}}
{\sum_{j=1}^{n_i}\exp(s_{ij})}.
```

Under a sampled subset, the naive estimator is

```math
\widehat z_i
=
\frac{\sum_{j\in I_i}\exp(s_{ij})v_{ij}}
{\sum_{j\in I_i}\exp(s_{ij})}.
```

Even if numerator and denominator are separately estimated without bias, their
ratio is generally biased:

```math
\mathbb E_I\!\left[\widehat z_i\right]
\ne
z_i.
```

An inverse-probability version is

```math
\widehat z_i^{\mathrm{IPW}}
=
\frac{
\sum_{j\in I_i}\pi_{ij}^{-1}\exp(s_{ij})v_{ij}
}{
\sum_{j\in I_i}\pi_{ij}^{-1}\exp(s_{ij})
}.
```

This is a ratio estimator. It can reduce sampling bias but is not exactly
unbiased at finite sample size. Its variance can explode when high-score
instances have low inclusion probability.

For self-attention, the situation is harder because every query-key score and
every denominator changes when the subset changes. There is no simple
inverse-probability correction that reproduces the full dense transformer in
general.

## 12. Top-k and score-based sampling

Let s_ij be a preliminary score and let I_i be the top-k set:

```math
I_i^{\mathrm{topk}}
=
\mathrm{arg\,topk}_{j\in\mathcal U_i}s_{ij}.
```

This is not a probability sample. Its inclusion probability is effectively one
for selected patches and zero for the rest, conditional on the score model.

Top-k sampling optimizes a different object:

```math
F_\theta\!\left(
\mathrm{TopK}_{s}(H_i)
\right)
\quad\text{rather than}\quad
F_\theta(H_i).
```

It can work when the task is driven by a few high-scoring instances. It fails
when:

- the preliminary score is poorly calibrated;
- positives are distributed across many moderate-score patches;
- context is required before a patch can be recognized;
- the score model attends to artifacts;
- the top-k operation is unstable under small perturbations.

Because selection and prediction are coupled, the training gradient is often
piecewise or requires a surrogate. Let the selected set change at a score
boundary. Then the mapping has a discontinuous derivative:

```math
\frac{\partial I_i^{\mathrm{topk}}}{\partial s_{ij}}
\quad\text{is undefined at ties and zero away from selection boundaries}.
```

Soft top-k relaxations smooth this boundary but optimize a relaxed operator,
not exact discrete selection.

## 13. Cluster-aware sampling

Let a slide be partitioned into G clusters with sizes n_g. A cluster sampler
allocates m_g tokens. The sample fraction in cluster g is

```math
f_g=\frac{m_g}{n_g}.
```

Uniform per-cluster allocation has

```math
m_g=\frac{m}{G},
```

while proportional allocation has

```math
m_g=m\frac{n_g}{n}.
```

These preserve different quantities. Proportional allocation estimates global
mass. Equal cluster allocation protects small clusters from being drowned out.

If cluster g contains a positive fraction q_g, the expected positive sample
count is

```math
\mathbb E[R_g]
=
m_g q_g.
```

The allocation problem can be written as a constrained variance-minimization
problem:

```math
\min_{m_1,\ldots,m_G}
\sum_{g=1}^{G}\frac{w_g^2\sigma_g^2}{m_g}
\quad\text{subject to}\quad
\sum_{g=1}^{G}m_g=m,
```

where w_g is the target population weight and sigma_g is within-cluster
variability. The continuous optimum is Neyman allocation:

```math
m_g
\propto
w_g\sigma_g.
```

In WSI, sigma_g and target relevance are unknown. Cluster-aware methods replace
them with learned or unsupervised proxies, which introduces estimation error.

## 14. Dropout and stochastic instance removal

Let b_ij be a Bernoulli retention variable:

```math
b_{ij}\sim\mathrm{Bernoulli}(1-p_{ij}).
```

The stochastic bag is

```math
H_i^{(b)}
=
\{h_{ij}:b_{ij}=1\}.
```

For linear sum pooling, inverse retention scaling can preserve the expected
sum:

```math
\widehat T_i
=
\sum_{j=1}^{n_i}
\frac{b_{ij}}{1-p_{ij}}h_{ij},
\qquad
\mathbb E_b[\widehat T_i]
=
\sum_{j=1}^{n_i}h_{ij}.
```

For a nonlinear transformer,

```math
\mathbb E_b\!\left[F_\theta(H_i^{(b)})\right]
\ne
F_\theta(H_i)
\quad\text{in general}.
```

The stochastic model is trained under a distribution of partial bags. This can
regularize reliance on a single instance or improve robustness, but it can also
erase the only positive patch.

If the positive count is r and each patch is retained independently with
probability 1-p, the probability of losing every positive is

```math
\mathbb P(\mathrm{lose\ all\ positives})
=
p^r.
```

For r equal to one, the loss probability is exactly p. The dropout rate must
therefore be interpreted relative to the positive-instance prevalence.

## 15. Rare-positive variance

Let Y_ij be a binary indicator that patch j contains target evidence. Under
uniform sampling, the sample prevalence estimator is

```math
\widehat q_i
=
\frac{1}{m_i}
\sum_{j\in I_i}Y_{ij}.
```

For simple random sampling without replacement,

```math
\mathrm{Var}(\widehat q_i)
=
\frac{1}{m_i}
\left(1-\frac{m_i}{n_i}\right)
\frac{n_i}{n_i-1}
q_i(1-q_i).
```

When q_i is small, the estimator is often zero even when the slide contains
positive patches. A transformer cannot recover evidence that was never included
in its context.

For a rare positive count r_i, the more relevant event is capture, not mean
variance:

```math
\mathbb P(\mathrm{capture})
=
1-
\frac{\binom{n_i-r_i}{m_i}}
\binom{n_i}{m_i}.
```

Sampling policies should report both capture probability estimates and
downstream predictive variance.

## 16. Complexity versus support

The main operators can be compared by their pairwise support and cost:

| Operator | Pairwise support per layer | Leading cost |
|---|---|---|
| Dense attention | All token pairs | O(L squared d) |
| Nyström attention | Dense factorized route through m landmarks | O(L m d + m squared d + T m cubed) |
| Windowed attention | Within-window pairs | O(L w d) |
| Graph message passing | Graph edges | O(E d) |
| Hierarchical attention | Within-region plus region pairs | O(R m squared d + R squared d) |
| Uniform sampling | All pairs among retained tokens | O(m squared d) after selection |

The table is incomplete without a statistical column:

```math
\text{surviving evidence}
=
\text{input coverage}
\cap
\text{context support}
\cap
\text{readout capacity}.
```

A method can have low arithmetic cost and poor rare-positive coverage, or high
nominal support and poor approximation of useful long-range interactions.

## 17. Distributed variable-length batches

For a batch with lengths L_i, naive dense padding cost is

```math
\mathcal C_{\mathrm{pad}}
\propto
B L_{\max}^2d.
```

The ideal per-slide packed cost is

```math
\mathcal C_{\mathrm{packed}}
\propto
d\sum_{i=1}^{B}L_i^2.
```

Their ratio is

```math
\rho_{\mathrm{batch}}
=
\frac{\sum_iL_i^2}{B L_{\max}^2}
\le 1.
```

Length bucketing improves this ratio by grouping similar slides. Distributed
gradient stacking can improve throughput by accumulating variable-length
examples without constructing one rectangular tensor. It changes execution,
not the slide-level mathematical objective, provided gradients are normalized
by the intended sample or token weighting.

If gradients are averaged per token instead of per slide, the effective
objective changes:

```math
\mathcal L_{\mathrm{token}}
=
\frac{\sum_i\sum_{j=1}^{n_i}\ell_{ij}}
\sum_i n_i
\ne
\frac{1}{B}\sum_i\mathcal L_i
=
\mathcal L_{\mathrm{slide}}
\quad\text{in general}.
```

Long slides can dominate training unless the reduction convention is explicit.

## 18. Sampling and supervision

Suppose the slide label is y_i and the model sees sampled evidence. The
training objective is

```math
\mathcal L_{\mathrm{sample}}
=
\mathbb E_{I_i\sim\pi_i}
\left[
\ell\!\left(
F_\theta(H_i^{(I_i)}),y_i
\right)
\right].
```

The full-bag objective is

```math
\mathcal L_{\mathrm{full}}
=
\ell\!\left(
F_\theta(H_i),y_i
\right).
```

Because the loss is nonlinear,

```math
\mathbb E_{I_i}
\left[
\ell(F_\theta(H_i^{(I_i)}),y_i)
\right]
\ne
\ell\!\left(
\mathbb E_{I_i}[F_\theta(H_i^{(I_i)})],y_i
\right).
```

Sampling can therefore act as data augmentation, regularization, and
measurement error simultaneously. The trained model may learn to predict from
partial evidence rather than to approximate the full-slide function.

## 19. Sanity checks

### Check A: length sweep

Evaluate the same slide with increasing retained patch counts. Report
prediction convergence:

```math
\varepsilon_i(m)
=
\left\|
F_\theta(H_i^{(I_m)})
-
F_\theta(H_i^{(I_{m_{\max}})})
\right\|_2.
```

The subset sequence should be nested or independently replicated, and the
sampling seed should be recorded.

### Check B: capture audit

If positive or annotated regions are available, estimate the probability of
capturing at least one positive patch. Do not infer it from mean sample size.

### Check C: seed variance

For K sampling seeds, report

```math
\widehat{\mathrm{Var}}_I(F_i)
=
\frac{1}{K-1}
\sum_{k=1}^{K}
\left(F_i^{(k)}-\overline F_i\right)^2.
```

Large seed variance means the reported slide prediction is a sampler-dependent
random variable.

### Check D: exact versus Nyström

On bags small enough for dense attention, compare logits, gradients, and
deletion rankings across exact and approximate operators.

### Check E: padding stress

Use bag sizes immediately below and above square numbers. Hold the available
feature population fixed as much as possible and measure the jump caused by
the completion remainder.

### Check F: allocation ablation

Compare uniform, proportional, stratified, cluster-aware, and score-based
sampling at equal patch budgets. Report capture, compute, predictive mean, and
predictive variance.

### Check G: objective weighting

Verify whether loss reduction is per slide, per patch, or per token. Equal
batch size does not imply equal slide contribution for variable-length bags.

## 20. C/R/G/S placement

| Component | Complexity and sampling realization |
|---|---|
| Context operator | Dense, landmark, windowed, graph, or hierarchical context; each defines a different pairwise support and cost. |
| Readout operator | Class-token or pooling readout applied to the full or sampled context state. |
| Geometry | Sampling can preserve, distort, or remove physical coordinates; square completion adds a raster convention and repeated-token multiplicity. |
| Surviving statistic | The evidence and interactions retained under the compute budget; a sampler changes the input object, while an attention approximation changes the context operator. |

The full design map is

```math
\text{full WSI bag}
\xrightarrow{\text{sampling or compression}}
\text{observed token object}
\xrightarrow{\text{context operator}}
\text{contextual states}
\xrightarrow{\text{readout}}
\text{slide statistic}.
```

The two approximation errors should be separated:

```math
\underbrace{
F_\theta(H_i)-F_\theta(H_i^{(I)})
}_{\text{sampling error}}
\qquad\text{and}\qquad
\underbrace{
F_\theta^{\mathrm{exact}}(H_i^{(I)})
-
F_\theta^{\mathrm{approx}}(H_i^{(I)})
}_{\text{context approximation error}}.
```

Reporting only runtime hides both.

## 21. Bottom line

Dense attention is quadratic in sequence length because it forms all pairwise
token scores. TransMIL's Nyström module reduces this cost through an
m-landmark interface, but its linear-in-length claim is conditional on fixed
landmark count, width, and pseudoinverse iterations. PPEG is linear in token
count and not the quadratic bottleneck.

Sampling is different. It changes the bag presented to the model. Uniform
sampling has a quantifiable rare-positive capture probability; score-based
sampling has selection bias; cluster-aware sampling trades global mass
preservation for coverage; stochastic dropout changes the training distribution.
Inverse-probability weighting can repair some linear pooling statistics, not a
general nonlinear transformer.

The honest summary is:

```math
\boxed{
\text{compute reduction}
\ne
\text{statistical preservation}
\ne
\text{full-slide equivalence}
}
```

Every efficient WSI transformer should report the retained token count,
attention approximation, square-padding rule, sampling distribution, rare
positive capture, batch reduction convention, and seed variance. Without those
quantities, a speed claim says little about which slide evidence the model
actually learned to use.

## References

- Shao et al., "TransMIL: Transformer based Correlated Multiple Instance
  Learning for Whole Slide Image Classification," https://arxiv.org/abs/2106.00908
- Campanella et al., "Clinical-grade computational pathology using weakly
  supervised deep learning on whole slide images," Nature Medicine 2019.
  https://www.nature.com/articles/s41591-019-0508-1
- Gadermayr et al., "Cluster-to-Conquer: A Framework for Comprehensive
  Histopathology Image Classification," https://arxiv.org/abs/2103.10626
- Zhu et al., "How Effective Can Dropout Be in Multiple Instance Learning?,"
  ICML 2025. https://proceedings.mlr.press/v267/zhu25q.html
- Wu et al., "Distributed Parallel Gradient Stacking: Solving Whole Slide
  Image Stacking Challenge in Multiple Instance Learning,"
  https://openreview.net/forum?id=ss5JNmJDkW
- Ilse et al., "Attention-based Deep Multiple Instance Learning,"
  https://arxiv.org/abs/1802.04712
- Lee et al., "Set Transformer: A Framework for Attention-based Permutation-
  Invariant Neural Networks," https://arxiv.org/abs/1810.00825
