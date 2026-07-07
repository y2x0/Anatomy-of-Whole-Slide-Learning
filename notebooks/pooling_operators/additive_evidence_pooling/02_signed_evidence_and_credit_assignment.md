# Signed Evidence And Credit Assignment

Additive pooling gives direct credit assignment because the bag score is a sum.

Let:

```math
r_i
=
\sum_j e_{ij}.
```

The contribution of patch $j$ is:

```math
\Delta_{ij}
=
e_{ij}.
```

No attention normalization is needed.

## Logit Contribution

For binary classification:

```math
p_i
=
\mathrm{sigmoid}(r_i).
```

Patch $j$ changes the log-odds by:

```math
\log\frac{p_i}{1-p_i}
-
\log\frac{p_i^{(-j)}}{1-p_i^{(-j)}}
=
e_{ij}.
```

Thus additive evidence is naturally a logit-space explanation.

## Probability Change Is Nonlinear

The probability change is:

```math
p_i-p_i^{(-j)}
=
\mathrm{sigmoid}(r_i)
-
\mathrm{sigmoid}(r_i-e_{ij}).
```

This depends on the current score $r_i$. The same evidence unit changes
probability more near the decision boundary than in a saturated region.

## Positive And Negative Maps

Define:

```math
e_{ij}^{+}
=
\max(e_{ij},0),
\qquad
e_{ij}^{-}
=
\max(-e_{ij},0).
```

Then:

```math
r_i
=
\sum_j e_{ij}^{+}
-
\sum_j e_{ij}^{-}.
```

A faithful map should show both positive and negative evidence. Showing only
$|e_{ij}|$ loses direction.

## Dense Summary

Additive MIL gives exact credit assignment in the model's score space:

```math
\frac{\partial r_i}{\partial e_{ij}}
=
1.
```

Every patch has a direct, signed, non-normalized contribution.

