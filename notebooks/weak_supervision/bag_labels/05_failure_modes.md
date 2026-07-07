# Failure Modes

Bag labels fail when the bag map is wrong or the latent witness is ambiguous.

## Witness Ambiguity

A positive bag only says:

```math
\sum_j Z_{ij}\ge 1.
```

If many patch subsets satisfy this, the model can choose any predictive witness.

Attention or max selection may converge to:

```math
\widehat j_i
\ne
j_i^{\mathrm{true}}.
```

The bag prediction can still be correct.

## Shortcut Witness

Suppose an artifact feature $A_{ij}$ appears in positive slides:

```math
P(A=1\mid Y=1)
>
P(A=1\mid Y=0).
```

If an artifact patch predicts $Y$, max or attention MIL may select it as the
witness:

```math
\widehat j_i
=
\arg\max_j s_\theta(h_{ij})
\quad
\text{artifact patch}.
```

The slide classifier works until the artifact distribution shifts.

## Prevalence Mismatch

The OR assumption treats these latent configurations identically:

```math
Z_A
=
(1,0,\ldots,0),
\qquad
Z_B
=
(1,1,\ldots,1).
```

If prognosis, grade, or subtype depends on burden:

```math
Y
=
F
\left(
\frac{1}{n}\sum_j Z_j
\right),
```

then max-like MIL is misspecified.

## Negative-Bag Assumption Break

If a negative slide can contain small positive-like regions because of label
thresholding or report noise:

```math
Y_i=0
\not\Longrightarrow
Z_{ij}=0\ \forall j,
```

then the strongest standard MIL constraint fails. Negative bags no longer give
clean instance negatives.

## Attention Interpretation Failure

Bag training identifies predictors of $Y$, not necessarily true patch labels:

```math
a_{ij}
\not=
P(Z_{ij}=1\mid H_i,Y_i).
```

Attention is a readout coefficient unless the model and loss are designed to
make it a calibrated posterior.

## Dense Summary

Bag-label MIL can be correct at the slide level and wrong at the instance level:

```math
\widehat Y_i=Y_i
\quad
\text{does not imply}
\quad
\widehat Z_{ij}=Z_{ij}.
```

That is the central weak-supervision trap.
