# Mean Pooling as a First-Moment Operator

## 1. Scope

Mean pooling is often introduced as the simplest multiple-instance learning
baseline:

> Encode every patch, average the patch embeddings, and classify the slide.

The simplicity is useful only if its mathematical consequences are stated
precisely. Mean pooling is not a neutral way to remove the instance axis. It
chooses the empirical first moment as the surviving statistic. Everything that
cannot be recovered from that moment is unavailable to the task head, even if
the patch encoder is expressive.

This note analyzes:

- the empirical-measure interpretation of mean pooling;
- permutation and geometry invariance;
- the sampling variance of a slide representation;
- why rare positives can be diluted;
- what survives under a linear or nonlinear head;
- how mean pooling differs from sum, max, and attention;
- the exact C/R/G/S placement.

The pathology anchors are the weakly supervised WSI systems represented by
Campanella et al.:

https://www.nature.com/articles/s41591-019-0508-1

and the general MIL architecture in Ilse et al.:

https://arxiv.org/abs/1802.04712

The set-function comparison uses Deep Sets:

https://arxiv.org/abs/1703.06114

## 2. Bag object

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

The patch list is an implementation representation. The corresponding
empirical measure is

```math
\widehat\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\delta_{h_{ij}},
```

where delta at a feature is a point mass. The mean pooled slide vector is the
first moment of this measure:

```math
\bar h_i
=
\int x\,d\widehat\mu_i(x)
=
\frac{1}{n_i}\sum_{j=1}^{n_i}h_{ij}
\in\mathbb R^d.
```

The complete model is

```math
\widehat y_i
=
\psi_\theta(\bar h_i),
```

where psi is a task head. The aggregation map is linear in the patch features;
the complete slide predictor need not be linear because psi may be an MLP.

## 3. Algebraic properties

### 3.1 Permutation invariance

Let P_i be any n_i by n_i permutation matrix. Then

```math
\overline{PH}
=
\frac{1}{n_i}\mathbf 1^{\mathsf T}PH
=
\frac{1}{n_i}\mathbf 1^{\mathsf T}H
=
\bar h_i.
```

Mean pooling is therefore exactly invariant to patch order. It cannot use
coordinates or adjacency unless they have already been encoded into each
patch vector.

### 3.2 Translation equivariance in feature space

For any vector a,

```math
\frac{1}{n_i}\sum_j(h_{ij}+a)
=
\bar h_i+a.
```

The operator preserves feature translations. The task head can cancel or
amplify this change depending on its parameters.

### 3.3 Convex-hull constraint

The pooled vector lies in the convex hull of the patch embeddings:

```math
\bar h_i
\in
\mathrm{conv}\{h_{i1},\ldots,h_{in_i}\}.
```

Mean pooling cannot extrapolate outside the patch-feature convex hull. A
linear classifier on the mean sees only a single point in that hull.

### 3.4 Multiplicity sensitivity and count blindness

If every patch is duplicated r times, the mean is unchanged:

```math
\frac{1}{rn_i}
\sum_{j=1}^{n_i}\sum_{q=1}^{r}h_{ij}
=
\bar h_i.
```

Thus mean pooling is invariant to uniform replication of the complete bag. It
does not preserve slide patch count. This can be desirable when count reflects
scanning resolution, but harmful when lesion burden is encoded by prevalence or
area.

Nonuniform duplication changes the mean. If patch j is duplicated r_j times:

```math
\bar h_i^{(r)}
=
\frac{\sum_jr_jh_{ij}}{\sum_jr_j}.
```

The operator therefore preserves a normalized composition, not absolute mass.

## 4. What statistic survives?

The first coordinate moment is preserved:

```math
m_i^{(1)}
=
\mathbb E_{\widehat\mu_i}[X]
=
\bar h_i.
```

The second raw moment is not determined by the first:

```math
m_i^{(2)}
=
\mathbb E_{\widehat\mu_i}[XX^{\mathsf T}]
=
\frac{1}{n_i}\sum_jh_{ij}h_{ij}^{\mathsf T}.
```

The covariance is

```math
\widehat\Sigma_i
=
\frac{1}{n_i}\sum_j
(h_{ij}-\bar h_i)(h_{ij}-\bar h_i)^{\mathsf T}.
```

Two bags can have the same mean and different covariance:

```math
\bar h_i=\bar h_i',
\qquad
\widehat\Sigma_i\ne\widehat\Sigma_i'.
```

The task head cannot distinguish them if it receives only the mean. A richer
patch encoder can map distributional information into the mean, but this is a
learned compression trick, not a guarantee that the distribution survives.

For a scalar feature, the mean also does not determine:

- the maximum;
- the minimum;
- quantiles;
- multimodality;
- the number of rare patches;
- spatial arrangement;
- within-slide variance.

The surviving-statistic claim should therefore be written as:

```math
\boxed{
\text{mean pooling preserves the empirical first moment and discards the
remaining instance distribution unless the encoder hides it in that moment}
}
```

## 5. Exact collisions

### 5.1 Two scalar bags

The bags

```math
\mathcal B_1=\{0,2\},
\qquad
\mathcal B_2=\{1,1\}
```

have equal mean:

```math
\frac{0+2}{2}
=
\frac{1+1}{2}
=1.
```

Their variances differ:

```math
\mathrm{Var}(\mathcal B_1)=1,
\qquad
\mathrm{Var}(\mathcal B_2)=0.
```

No head applied only to the mean can recover this difference.

### 5.2 Rare positive versus diffuse background

Let a positive feature be p and background feature be b. Compare

```math
\mathcal B_1=\{p,b,b,b\},
\qquad
\mathcal B_2=\left\{
\frac{p+3b}{4},
\frac{p+3b}{4},
\frac{p+3b}{4},
\frac{p+3b}{4}
\right\}.
```

Their means are equal. The first bag contains a distinct positive instance;
the second contains no separable positive instance under a patch-level rule.
A mean-only predictor sees the same vector.

### 5.3 Same mean, different spatial layout

Let four features occupy a two by two physical grid:

```math
H
=
\begin{bmatrix}
a\\b\\c\\d
\end{bmatrix},
\qquad
H'
=
\begin{bmatrix}
a\\c\\b\\d
\end{bmatrix}.
```

Their means are equal:

```math
\frac{a+b+c+d}{4}
=
\frac{a+c+b+d}{4}.
```

Mean pooling cannot distinguish the layouts. A convolution, graph, or
coordinate-aware transformer can, provided the geometry is supplied faithfully.

## 6. Gradients and credit

The derivative of the mean with respect to each patch is

```math
\frac{\partial\bar h_i}{\partial h_{ij}}
=
\frac{1}{n_i}I_d.
```

For a scalar target F_i equal to a head applied to the mean:

```math
\frac{\partial F_i}{\partial h_{ij}}
=
\frac{1}{n_i}
\frac{\partial F_i}{\partial\bar h_i}.
```

Every patch receives the same direct Jacobian before accounting for its feature
content. A gradient-times-input score becomes

```math
s_{ij}^{\mathrm{gxi}}
=
\left\|
\frac{1}{n_i}
h_{ij}\odot
\frac{\partial F_i}{\partial\bar h_i}
\right\|_1.
```

Differences in such a score come from feature values and the head direction,
not from a learned instance-specific routing weight.

For a linear head with target logit

```math
F_i=w^{\mathsf T}\bar h_i+b,
```

the exact additive contribution of patch j is

```math
c_{ij}
=
\frac{1}{n_i}w^{\mathsf T}h_{ij}.
```

The logit decomposes exactly:

```math
F_i
=
b+\sum_{j=1}^{n_i}c_{ij}.
```

For a nonlinear head, the first-moment representation is still exact, but a
unique patch-level additive decomposition is not generally defined. One can
use gradients, integrated gradients, or Shapley values, but each introduces a
credit convention.

## 7. Sampling interpretation

Suppose patches are sampled from a slide-level population with distribution
mu_i. The empirical mean estimates the population mean:

```math
\mathbb E_{\mu_i}[X]
=
\mu_i^{(1)}.
```

Under independent samples with covariance Sigma_i,

```math
\mathbb E[\bar h_i]
=
\mu_i^{(1)},
\qquad
\mathrm{Cov}(\bar h_i)
=
\frac{1}{n_i}\Sigma_i.
```

The standard error decreases as n_i to the one-half power:

```math
\mathrm{SE}(\bar h_i)
\propto
n_i^{-1/2}.
```

This statistical interpretation assumes the patches are sampled from a
well-defined population. WSI patches are spatially correlated and often
filtered by tissue masks, so the effective sample size can be lower than n_i.

For correlated patches with covariance Gamma_r at spatial lag r, the variance
of a scalar mean is

```math
\mathrm{Var}(\bar h_i)
=
\frac{1}{n_i^2}
\left[
n_i\Gamma_0
+
2\sum_{r=1}^{n_i-1}(n_i-r)\Gamma_r
\right].
```

Positive spatial correlation makes the estimator noisier than independent
sampling suggests. Counting patches alone overstates information.

## 8. Effective sample size

For scalar observations with autocorrelation rho_r, define

```math
n_{\mathrm{eff}}
=
\frac{n_i}
{1+2\sum_{r=1}^{\infty}\rho_r}.
```

The finite-slide sum is truncated at available lags. If neighboring patches
are strongly correlated, n_eff can be far below n_i. A mean over 100,000
overlapping patches is not necessarily a 100,000-sample estimate.

For weighted pooling with normalized weights alpha_j, an approximate effective
sample size is

```math
n_{\mathrm{eff}}^{(\alpha)}
=
\frac{1}{\sum_j\alpha_j^2}.
```

Mean pooling has alpha_j equal to one over n_i, giving n_eff equal to n_i.
Attention pooling can have a much smaller effective sample size.

## 9. Rare-positive dilution

Suppose r_i patches are positive and n_i minus r_i are background. Let patch
features be p_j for positives and b_j for background. The mean is

```math
\bar h_i
=
\frac{1}{n_i}
\left(
\sum_{j\in P_i}p_j
+
\sum_{j\in B_i}b_j
\right).
```

The total positive contribution has norm bounded by

```math
\left\|
\frac{1}{n_i}\sum_{j\in P_i}p_j
\right\|_2
\le
\frac{r_i}{n_i}
\max_{j\in P_i}\|p_j\|_2.
```

If positive features are aligned and background is centered, signal magnitude
scales with prevalence r_i over n_i. If positives are rare, the first moment
can be dominated by background even when one patch is highly diagnostic.

A linear classifier w sees positive margin contribution

```math
\Delta_i^{+}
=
\frac{1}{n_i}\sum_{j\in P_i}w^{\mathsf T}p_j.
```

To keep this margin fixed as n_i grows while r_i stays fixed, the patch-level
margin must grow linearly with n_i. That is an unrealistic burden for a fixed
encoder.

## 10. Class imbalance and prevalence

Mean pooling estimates composition. If the label depends on presence rather
than prevalence, composition is a mismatch:

```math
y_i
=
\mathbf 1\{r_i\ge 1\}
\quad\text{versus}\quad
\bar h_i
\approx
\frac{r_i}{n_i}p+\left(1-\frac{r_i}{n_i}\right)b.
```

The label stays one while the representation moves toward background as n_i
increases at fixed r_i. This creates a bag-size-dependent margin.

A max or noisy-or operator can preserve presence more directly. Attention can
learn to increase the positive weight, but it must discover the instance-level
structure under weak bag labels.

## 11. Mean versus sum pooling

Sum pooling is

```math
s_i
=
\sum_{j=1}^{n_i}h_{ij}
=
n_i\bar h_i.
```

Mean and sum preserve the same direction only if n_i is supplied or irrelevant.
Sum retains total mass and is sensitive to patch count:

```math
s_i^{(r)}
=
r s_i
```

under uniform replication. Mean removes that scale.

If the head receives both mean and count,

```math
z_i
=
\left[\bar h_i;\log n_i\right],
```

it can recover the sum exactly:

```math
s_i
=
\exp(\log n_i)\bar h_i.
```

This makes count an explicit representation axis rather than an accidental
shortcut.

## 12. Mean versus max pooling

Coordinatewise max pooling is

```math
m_i^{\max}(r)
=
\max_{1\le j\le n_i}h_{ij,r}.
```

The mean has stable dense gradients. Max has gradient routed through argmax
instances:

```math
\frac{\partial m_i^{\max}(r)}{\partial h_{ij,r}}
=
\mathbf 1\{j\in\mathrm{ArgMax}_r\}
\quad\text{under a selected tie rule}.
```

Mean preserves average composition. Max preserves coordinatewise extremes.
Neither preserves the full distribution.

For a rare positive that is extreme in a known feature coordinate, max can avoid
dilution. For noisy artifacts, max can select the wrong patch.

## 13. Mean versus attention

Attention pooling has normalized learned weights:

```math
\alpha_{ij}
=
\frac{\exp(s_\theta(h_{ij}))}
{\sum_{\ell=1}^{n_i}\exp(s_\theta(h_{i\ell}))},
\qquad
z_i^{\mathrm{attn}}
=
\sum_j\alpha_{ij}h_{ij}.
```

Mean is the special case alpha equal to one over n_i. The attention operator
can adapt effective sample size:

```math
n_{\mathrm{eff}}^{(\alpha)}
=
\frac{1}{\sum_j\alpha_{ij}^2}.
```

It can preserve rare evidence if the score learns the correct signal, but it
also introduces attention collapse, score shortcuts, and explanation ambiguity.
Mean has less capacity and fewer failure modes.

## 14. Mean under geometric transformations

Let T be a transformation of coordinates only:

```math
(h_{ij},c_{ij})
\longmapsto
(h_{ij},T(c_{ij})).
```

Mean pooling is invariant:

```math
\bar h_i(H_i,T(C_i))
=
\bar h_i(H_i,C_i).
```

This is a strong geometry quotient. Translation, rotation, adjacency, and
regional arrangement are removed unless the patch encoder itself sees local
context or coordinates.

If coordinates are concatenated to features,

```math
\widetilde h_{ij}
=
\begin{bmatrix}
h_{ij}\\
\varphi(c_{ij})
\end{bmatrix},
```

mean pooling preserves the average coordinate feature, not the arrangement.
Two different spatial layouts can still collide.

## 15. Mean and missing tissue

Let m_ij be an occupancy indicator on a complete candidate grid. A mean over
retained tissue patches is

```math
\bar h_i^{\mathrm{ret}}
=
\frac{\sum_jm_{ij}h_{ij}}{\sum_jm_{ij}}.
```

This is a conditional mean given retention:

```math
\bar h_i^{\mathrm{ret}}
=
\mathbb E[X\mid m=1]
\quad\text{empirically}.
```

It is not the mean over the full slide domain unless missing cells are missing
completely at random relative to the feature and target. Tissue filtering can
therefore create selection bias.

A count or tissue-fraction statistic can be added:

```math
\tau_i
=
\frac{1}{N_i}\sum_jm_{ij}.
```

The pair [mean, tissue fraction] retains more information than the mean alone,
but still not the arrangement of occupied tissue.

## 16. Head capacity does not undo aggregation loss

Let A(H) be mean pooling and psi any deterministic head. If

```math
A(H)=A(H'),
```

then necessarily

```math
\psi(A(H))=\psi(A(H')).
```

No increase in head depth or width can separate the collision. The only way to
recover the lost distinction is to change the patch encoder or aggregation
operator so that the representations differ before the head.

This is the key order-of-operations fact:

```math
\text{encode}
\longrightarrow
\text{mean}
\longrightarrow
\text{head}
\quad\text{cannot recover information discarded by mean}.
```

## 17. Sanity checks

### Check A: duplicate invariance

Duplicate every patch uniformly and verify that mean pooling is unchanged:

```math
\bar h_i^{(r)}-\bar h_i=0.
```

If the complete model changes, the change comes from a count-sensitive head,
normalization, or implementation detail outside the mean.

### Check B: layout invariance

Permute patch coordinates while keeping feature values fixed. Mean pooling
should be unchanged. This verifies that no hidden ordering enters preprocessing.

### Check C: rare-positive sweep

Hold one positive patch fixed and increase background count. Measure the target
margin:

```math
\Delta_i(n)
=
\psi\!\left(
\frac{p+(n-1)b}{n}
\right).
```

The curve reveals dilution directly.

### Check D: covariance collision

Construct bags with equal means and different variance or multimodality. A
mean-only model should produce equal representations up to numerical error.

### Check E: spatial shuffle

Shuffle features across physical coordinates. Mean pooling should not change;
any change indicates geometry entered before aggregation.

### Check F: effective sample size

Estimate feature autocorrelation across spatial lags and report n_eff rather
than only the raw patch count.

## 18. C/R/G/S placement

| Component | Mean-pooling realization |
|---|---|
| Context operator | None at the bag level; each patch is encoded independently unless the patch encoder already contains local context. |
| Readout operator | Uniform average with weight one over n_i for every retained patch. |
| Geometry | Quotiented out at aggregation; coordinates and arrangement survive only if encoded into patch features. |
| Surviving statistic | Empirical first moment, optionally followed by a task head; covariance, maxima, prevalence arrangement, and spatial interactions are not directly retained. |

The forward map is

```math
\{h_{i1},\ldots,h_{in_i}\}
\xrightarrow{\text{uniform measure}}
\widehat\mu_i
\xrightarrow{\text{first moment}}
\bar h_i
\xrightarrow{\text{task head}}
\widehat y_i.
```

The C/R/G/S description is therefore exact:

```math
\boxed{
\text{no bag context}
\;+\;
\text{uniform readout}
\;+\;
\text{first-moment survival}
\;+\;
\text{geometry quotient}
}
```

## 19. Bottom line

Mean pooling is not merely a weak version of attention. It is a precise
statistical choice: represent a slide by its empirical first moment. This gives
permutation invariance, stable gradients, low variance under independent
sampling, and a clean baseline. It also removes spatial arrangement, patch
count, higher moments, and rare-positive identity unless the patch encoder
encodes those distinctions into the mean.

The most important failure is prevalence mismatch. If the label means that a
positive patch exists but the number of background patches varies, the positive
signal shrinks like its prevalence. A large WSI can therefore be harder than a
small one even when the lesion evidence is identical.

The honest summary is:

```math
\boxed{
\text{mean pooling}
=
\text{empirical first moment}
\ne
\text{presence detector}
\ne
\text{spatial representation}
\ne
\text{full slide distribution}
}
```

Mean pooling belongs in every MIL comparison because its assumptions are
transparent. Its weakness is equally transparent: the head can only reason
about what the first moment retained.

## References

- Campanella et al., "Clinical-grade computational pathology using weakly
  supervised deep learning on whole slide images," Nature Medicine 2019.
  https://www.nature.com/articles/s41591-019-0508-1
- Ilse et al., "Attention-based Deep Multiple Instance Learning,"
  https://arxiv.org/abs/1802.04712
- Zaheer et al., "Deep Sets," https://arxiv.org/abs/1703.06114
- Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and
  Segmentation," https://arxiv.org/abs/1612.00593
