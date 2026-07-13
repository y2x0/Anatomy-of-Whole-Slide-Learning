# MoPaDi MIL Gradient Guidance

For slide-level labels, MoPaDi replaces linear patch supervision with a
transformer-based MIL classifier and generates counterfactuals for selected
highly predictive tiles.

## 1. Bag Predictor

Let

```math
F_c(B)=F_c(z_1,\ldots,z_n)
```

denote the patient-level class score after layer normalization, multi-head
cross-attention pooling, self-attention, and the classifier head.

For target class `b`, a local tile direction can be oriented to increase the
target score:

```math
d_{jb}
=\frac{\nabla_{z_j}F_b(B)}
{\|\nabla_{z_j}F_b(B)\|_2}.
```

Equivalently, if the paper's target loss is minimized, use the negative
normalized target-loss gradient. The sign must be chosen by verifying that the
target score increases.

## 2. First-Order Bag Effect

For one edited tile,

```math
F_b(B_{j,\alpha})-F_b(B)
=\alpha
\|\nabla_{z_j}F_b(B)\|_2
+O(\alpha^2)
```

when moving along the normalized target gradient. The direction is local and
bag-conditional because contextual interactions make it depend on every tile.

## 3. Tile Selection

MoPaDi applies generation to top predictive tiles rather than synthesizing an
entire gigapixel slide. This yields patch-level morphological explanations for
a slide-level classifier. It does not construct a globally coherent
counterfactual WSI.

## 4. Frozen-Context Approximation

If each selected tile is manipulated separately, the gradient is computed at
the original bag. Simultaneously replacing several decoded tiles can change
attention and interactions:

```math
\Delta F_b
\ne
\sum_{j\in A}
\nabla_{z_j}F_b(B)^{\top}\Delta z_j
```

outside the local additive approximation.

## 5. Re-encoding Consistency

Let the generated image be `x_cf` and its re-encoded feature be

```math
\widetilde z_{\mathrm{cf}}=E_{\theta}(x_{\mathrm{cf}}).
```

A cycle-consistency diagnostic is

```math
\|\widetilde z_{\mathrm{cf}}-(z+\alpha d)\|_2.
```

Large error means the decoded image did not realize the intended classifier
space intervention.

