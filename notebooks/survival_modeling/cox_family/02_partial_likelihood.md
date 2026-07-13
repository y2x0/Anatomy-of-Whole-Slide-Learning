# Partial Likelihood

The Cox model avoids specifying the baseline hazard by conditioning on event
order.

Let:

```math
R_i = \{j : X_j \ge X_i\}
```

be the risk set at the observed event time
```math
X_i
```
. It contains everyone still
under observation just before
```math
X_i
```
.

For an uncensored event
```math
\delta_i=1
```
, the Cox model assigns probability:

```math
\Pr(i \text{ fails at } X_i \mid R_i)
=
\frac{\exp(\eta_i)}
{\sum_{j\in R_i}\exp(\eta_j)}.
```

The partial likelihood is:

```math
L_{\mathrm{Cox}}(\theta)
=
\prod_{i:\delta_i=1}
\frac{\exp(\eta_i)}
{\sum_{j\in R_i}\exp(\eta_j)}.
```

The negative log partial likelihood is:

```math
\mathcal{L}_{\mathrm{Cox}}(\theta)
=
-
\sum_{i:\delta_i=1}
\left[
\eta_i
-
\log\sum_{j\in R_i}\exp(\eta_j)
\right].
```

Only observed events contribute numerator terms. Censored samples still matter
because they appear in risk-set denominators before their censoring time.

## Softmax View

At each event time, the Cox loss is a softmax classification problem over the
risk set:

```math
p_{ij}
=
\frac{\exp(\eta_j)}
{\sum_{k\in R_i}\exp(\eta_k)}.
```

The event patient
```math
i
```
 is the positive class. Everyone in
```math
R_i
```
is a
competitor.

This makes the gradient easy to read:

```math
\frac{\partial \mathcal{L}_i}{\partial \eta_j}
=
p_{ij} - \mathbf{1}[j=i],
\qquad j\in R_i.
```

For a linear head
```math
\eta_j=w^\top z_j
```
:

```math
\frac{\partial \mathcal{L}_i}{\partial w}
=
\sum_{j\in R_i}p_{ij}z_j - z_i.
```

So Cox training pushes the event patient's representation above the risk-set
average representation.

## What Censoring Does

If
```math
\delta_i=0
```
, patient
```math
i
```
has no event numerator. But for any earlier
event time
```math
X_k \le X_i
```
, patient
```math
i\in R_k
```
. Therefore censored patients
shape the denominator until they leave observation.

This encodes:

```math
T_i > X_i
```

but not a precise event time.

## Ties

If multiple events happen at the same observed time, the exact partial
likelihood is more complicated. Common approximations include Breslow and Efron
ties.

The simple Breslow form for tied event set
```math
D_t
```
is:

```math
\ell_t
=
\sum_{i\in D_t}\eta_i
-
|D_t|\log\sum_{j\in R_t}\exp(\eta_j).
```

For small clinical datasets with coarse follow-up intervals, tie handling can
matter.

## WSI Consequence

For WSI survival, a batch may contain only a few events. The Cox denominator
couples many slides together through risk sets, so training is sensitive to:

```text
batch construction
event imbalance
censoring distribution
cohort size
time discretization in recorded labels
```

The loss looks local at each event time, but the risk-set coupling is global.

## Key Takeaway

The partial likelihood is a product of conditional event probabilities over
risk sets. It is therefore ranking-like, because each event is favored over
patients who remain at risk at that time, but it is not merely a pairwise
ranking loss: every risk-set member contributes through the shared log-sum-
exp denominator, and the risk sets are determined by the observed follow-up
times.

```text
among everyone still at risk, assign the observed event the largest conditional
event probability under the Cox model.
```

That distinction matters for WSI interpretation. The learned representation is
optimized for a proportional-hazards risk ordering and its induced likelihood,
not for a calibrated survival curve unless a baseline hazard is estimated and
the required modeling assumptions hold.
