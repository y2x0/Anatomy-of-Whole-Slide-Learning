# PAMIL Dual-Branch Prototype Attention

PAMIL learns global prototypes, compares every WSI instance with them, and
regularizes instance-oriented and prototype-oriented predictive branches to
agree.

## 1. Global Prototype Dictionary

Let

```math
P=[p_1,\ldots,p_K]^{\top}\in\mathbb R^{K\times d},
\qquad
H_i=[h_{i1},\ldots,h_{in_i}]^{\top}\in\mathbb R^{n_i\times d}.
```

Global prototypes are initialized by two-stage K-means over training patch
embeddings, optimized during learning, and projected to nearest observed
instances. After dimensionality reduction, write

```math
T_i=g(H_i),
\qquad
Q=g(P).
```

PAMIL combines a hyperbolic tangent transform of instances with a sigmoid
transform of prototypes. With rows indexing instances and columns indexing
prototypes, the paper's cross-comparison can be written as

```math
S_i
=\tanh(T_iW_q)\,
\sigma(QW_k)^{\top}
\in\mathbb R^{n_i\times K}.
```

The paper prints this product under a different matrix-orientation convention;
the object is the same instance-by-prototype matrix. It is a learned bipartite
relation, not yet a bag prediction.

## 2. Instance Representation Branch

The instance branch maps prototype similarities to class-conditioned instance
scores using a learned prototype-to-category matrix:

```math
A_i=S_iW_c,
\qquad
\widehat A_i
=\mathrm{softmax}_{\mathrm{instance}}(A_i).
```

It pools compressed instance features and applies a classifier with one hidden
layer:

```math
z_i^{\mathrm{inst}}
=(\widehat A_i)^{\top}T_i,
\qquad
\widehat p_i^{\mathrm{inst}}
=f_{\mathrm{cls},1}(z_i^{\mathrm{inst}}).
```

The explanatory object is a category-conditioned instance-attention map whose
scores were constructed through prototype relations.

## 3. Prototype Representation Branch

The prototype branch max-pools the instance dimension of the similarity matrix.
For prototype `k`, the primitive surviving statistic is

```math
m_{ik}=\max_j(S_i)_{jk}.
```

After arranging the pooled prototype-to-bag values by output category, denote
the paper's matrix by

```math
\widehat S_i\in\mathbb R^{K\times C}.
```

The second prediction is

```math
\widehat p_i^{\mathrm{proto}}
=f_{\mathrm{cls},2}
(\widehat S_i^{\top}Q).
```

This branch retains existential prototype matches, whereas the instance branch
retains weighted instance features. The final probability vector averages both
branches:

```math
\widehat p_i
=\frac12
(\widehat p_i^{\mathrm{inst}}
+\widehat p_i^{\mathrm{proto}}).
```

## 4. Agreement Is Not Identity

PAMIL trains both predictions with binary cross-entropy and regularizes their
agreement through

```math
\mathcal L_{\mathrm{eq}}
=D_{\mathrm{KL}}
(\widehat p_i^{\mathrm{proto}}
\,\|\,
\widehat p_i^{\mathrm{inst}}).
```

The full objective also contains the paper's prototype-instance clustering
term. Small KL divergence aligns outputs, not internal explanations. Two
branches can predict the same class while relying on different
instance-prototype paths.

## 5. Explanation Levels

PAMIL exposes

```math
S_i,
\qquad
\widehat A_i,
\qquad
\widehat S_i,
\qquad
\text{projected prototype patches}.
```

These correspond to resemblance, category-conditioned routing,
prototype-to-bag matching, and visual reference. Although the paper calls the
branch quantities contribution or importance, neither is automatically an
additive decomposition of a class logit because both branch classifiers are
nonlinear. Branch-specific deletion and score-credit tests remain necessary.
