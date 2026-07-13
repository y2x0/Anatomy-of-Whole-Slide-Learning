# DiCE Diversity and Determinantal Geometry

## 1. Set-Valued Explanation

DiCE searches jointly for `m` counterfactuals

```math
\mathcal C=\{c_1,\ldots,c_m\}.
```

Its objective combines target loss, proximity, and diversity:

```math
\mathcal L(\mathcal C)
=\frac1m\sum_{i=1}^m
\ell(f(c_i),y_{\star})
+\lambda_1\frac1m\sum_{i=1}^m d(c_i,x)
-\lambda_2\,\mathrm{Diversity}(\mathcal C).
```

Constraints can fix immutable variables and bound feasible ranges.

## 2. DPP Diversity

Define a similarity kernel

```math
K_{ij}
=\frac{1}{1+d(c_i,c_j)}.
```

DiCE uses

```math
\mathrm{Diversity}(\mathcal C)=\det(K).
```

The determinant is the squared volume spanned in the kernel feature space. It
is small when solutions are similar or redundant and larger when they occupy
distinct directions.

## 3. Diversity-Proximity Conflict

Increasing `lambda_2` can force candidates away from the nearest boundary
region. Diversity is useful only among individually valid and feasible
solutions. Otherwise the optimizer can purchase determinant volume with
implausible feature combinations.

## 4. Sparsity Postprocessing

DiCE treats proximity in the objective and can reduce the number of changed
features through post-hoc coordinate restoration while preserving target
validity. This greedy operation is order-dependent and need not find the
minimum-cardinality counterfactual.

## 5. WSI Diversity

For pathology, diversity should distinguish mechanisms rather than merely
pixels. Candidate distances can be defined over:

```math
\text{changed region sets},
\quad
\text{cell-composition vectors},
\quad
\text{morphology descriptors},
\quad
\text{generator latent directions}.
```

Multiple visually different images produced by the same latent shortcut are
not mechanistically diverse explanations.

