# DTFD Pseudo-Bags And Two-Tier Distillation

Source:

`text
Zhang et al., DTFD-MIL: Double-Tier Feature Distillation Multiple Instance
Learning for Histopathology Whole Slide Image Classification, CVPR 2022.
https://arxiv.org/abs/2203.12081
`

## 1. Why pseudo-bags are not just regions

A WSI bag B_i contains n_i extracted patch features:

```math
B_i
=
\{h_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}=H_{\phi}(x_{ij})\in\mathbb R^d.
```

DTFD randomly partitions the instances into R approximately balanced pseudo-bags:

```math
B_i
=
\biguplus_{r=1}^{R}B_{ir},
\qquad
|B_{ir}|\approx n_i/R.
```

The pseudo-bags are virtual training bags, not anatomically annotated regions.
Their labels are inherited from the parent slide:

```math
y_{ir}=y_i.
```

Under positive-instance MIL semantics this creates label noise: a positive slide
can produce a pseudo-bag containing no positive patch. If positive instances
occupy a fraction p_i of the slide and the pseudo-bag has m instances sampled
without replacement, the probability that a positive slide's pseudo-bag contains
no positive instance is approximately

```math
\Pr(\text{empty positive pseudo-bag}\mid y_i=1)
\approx
(1-p_i)^m.
```

This probability is the explicit tradeoff in virtual bag enlargement. Increasing
R decreases m and increases the chance of an inherited but locally false
positive label.

## 2. Tier-1 AB-MIL

Let the Tier-1 attention operator be

```math
a_{irj}
=
\frac{\exp s_{\theta}(h_{irj})}
{\sum_{k\in B_{ir}}\exp s_{\theta}(h_{irk})},
\qquad
z_{ir}
=
\sum_{j\in B_{ir}}a_{irj}h_{irj}.
```

A binary Tier-1 classifier produces

```math
\ell_{ir}^{(1)}
=
w_1^{\mathsf T}z_{ir}+b_1,
\qquad
q_{ir}
=
\sigma\left(\ell_{ir}^{(1)}\right).
```

The slide label supervises every pseudo-bag in Tier 1:

```math
\mathcal L_1
=
-\sum_{i=1}^{N}\sum_{r=1}^{R}
\left[
y_i\log q_{ir}
+
(1-y_i)\log(1-q_{ir})
\right].
```

For a positive slide with sparse positive tissue, this objective does not
identify a true positive patch in every pseudo-bag. It instead trains a noisy
auxiliary classifier whose outputs are used to construct candidate features.

## 3. DTFD's instance-probability derivation

The attention coefficient is not itself a probability of a positive instance.
DTFD derives an instance class signal from the bag classifier. For class c, let
the Tier-1 bag logit be

```math
\ell_{ir,c}^{(1)}
=
w_{1,c}^{\mathsf T}z_{ir}+b_{1,c}.
```

Expanding the pooled representation gives

```math
\ell_{ir,c}^{(1)}
=
\sum_{j\in B_{ir}}
a_{irj}\,
w_{1,c}^{\mathsf T}h_{irj}
+
b_{1,c}.
```

The Grad-CAM-style instance contribution is therefore

```math
g_{irj,c}
=
a_{irj}\,
w_{1,c}^{\mathsf T}h_{irj}.
```

For binary classification, the paper converts the positive and negative
instance signals into a probability by a two-class softmax:

```math
p_{irj,c}
=
\frac{\exp(g_{irj,c})}
{\exp(g_{irj,0})+\exp(g_{irj,1})}.
```

This is a classifier-weighted contribution, not the normalized attention
coefficient. It can rank two patches differently even when their attention
weights are similar, because their directions relative to the class vector
differ.

The derivation also has a boundary: if the attention network's dependence on
h is included in the full derivative, the simple product above is not the whole
Jacobian. The DTFD construction uses the AB-MIL/Grad-CAM decomposition as the
operational instance signal; it should not be read as a causal patch effect.

## 4. Feature distillation choices

For each pseudo-bag, Tier 1 supplies a feature to Tier 2. DTFD evaluates four
selection rules.

Maximum-selection uses the patch with largest derived positive probability:

```math
j_{ir}^{\max}
=
\arg\max_{j\in B_{ir}}p_{irj,1},
\qquad
d_{ir}^{\mathrm{MaxS}}
=
h_{irj_{ir}^{\max}}.
```

Max-min selection concatenates the most positive and most negative candidates:

```math
j_{ir}^{\min}
=
\arg\min_{j\in B_{ir}}p_{irj,1},
\qquad
d_{ir}^{\mathrm{MaxMinS}}
=
\left[
h_{irj_{ir}^{\max}}
\middle\Vert
h_{irj_{ir}^{\min}}
\right].
```

Maximum-attention selection uses the attention score rather than the derived
class probability:

```math
j_{ir}^{\mathrm{MAS}}
=
\arg\max_{j\in B_{ir}}a_{irj},
\qquad
d_{ir}^{\mathrm{MAS}}
=
h_{irj_{ir}^{\mathrm{MAS}}}.
```

Aggregated feature selection forwards the Tier-1 attention embedding itself:

```math
d_{ir}^{\mathrm{AFS}}
=
z_{ir}.
```

The selection rule is part of the readout operator. MaxS and MAS are sparse
selection operators; MaxMinS preserves two opposing extremes; AFS retains a
weighted first moment.

## 5. Tier-2 parent-slide MIL

The distilled pseudo-bag features form a second-level bag:

```math
D_i=\{d_{ir}\}_{r=1}^{R}.
```

Tier 2 applies another attention-based MIL readout:

```math
\beta_{ir}
=
\frac{\exp t_{\psi}(d_{ir})}
{\sum_{s=1}^{R}\exp t_{\psi}(d_{is})},
\qquad
z_i^{(2)}
=
\sum_{r=1}^{R}\beta_{ir}d_{ir}.
```

The final classifier is

```math
\ell_i^{(2)}
=
w_2^{\mathsf T}z_i^{(2)}+b_2,
\qquad
\widehat y_i
=
\sigma\left(\ell_i^{(2)}\right).
```

The full training objective is a weighted sum of both tiers:

```math
\mathcal L_{\mathrm{DTFD}}
=
\mathcal L_2
+
\lambda\mathcal L_1,
\qquad
\lambda\geq 0,
```

where L_2 is the Tier-2 slide loss and L_1 is the inherited-label pseudo-bag
loss. Tier 1 is therefore both a supervised auxiliary model and a feature
proposal mechanism.

## 6. What hierarchy means here

DTFD does not construct a spatial tree. Its hierarchy is induced by a random
partition and a two-stage supervised computation:

```math
\{h_{ij}\}
\longrightarrow
\{d_{ir}\}_{r=1}^{R}
\longrightarrow
z_i^{(2)}
\longrightarrow
\widehat y_i.
```

The child-parent relation is the pseudo-bag membership relation. It is not
interpretable as tissue adjacency unless the partitioning procedure explicitly
uses coordinates, which the core DTFD formulation does not require.

## 7. C/R/G/S placement

`text
C: Tier-1 and Tier-2 gated attention scores.

R: Tier-1 attention or selection distillation, followed by Tier-2 attention.

G: pseudo-bag membership; no anatomical spatial graph is required.

S: parent slide labels supervise both tiers, with inherited pseudo-bag labels
   introducing controlled label noise.

Sparsity is handled by proposing local candidates and then re-aggregating them,
not by proving that each pseudo-bag is semantically positive.
`
