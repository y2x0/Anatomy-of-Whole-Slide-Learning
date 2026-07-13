# Noisy-Or Failure Modes

Noisy-or is mathematically precise, but its assumptions are strong.

## 1. Conditional Independence Is False

The formula:

```math
P(y_i=0\mid H_i)
=
\prod_j(1-p_{ij})
```

assumes instance failures are conditionally independent. WSI patches are highly
correlated because nearby tissue produces repeated morphology. Noisy-or can
overcount evidence when many similar patches have moderate probability.

## 2. Saturation

If many patches have nonzero probability:

```math
r_i
=
1-\prod_j(1-p_{ij})
\approx
1.
```

Once saturated, the model loses resolution among positive slides. A mildly
positive slide and a strongly positive slide can both map near
```math
1
```
.

## 3. Instance Probability Non-Identifiability

Many probability vectors give the same bag probability:

```math
1-\prod_j(1-p_{ij})
=
1-\prod_j(1-\widetilde p_{ij}).
```

Slide labels alone do not identify true patch probabilities.

## 4. Negative Bag Pressure

For negative bags:

```math
\mathcal{L}_i^{-}
=
-\sum_j\log(1-p_{ij}).
```

Every patch is pushed toward zero. If labels are noisy or if negative slides
contain unlabeled early disease, the model receives wrong instance-level
pressure everywhere.

## 5. Prevalence Blindness

Noisy-or answers:

```text
is there at least one trigger?
```

It does not answer:

```text
how much tissue is involved?
```

Once the probability saturates, additional positive tissue barely changes the
bag probability.

## Dense Summary

Noisy-or fails when:

```text
instances are correlated,
prevalence matters,
bag labels are noisy,
or patch probabilities are interpreted as true labels.
```

The operator is a probabilistic OR, not a calibrated lesion burden estimator.
