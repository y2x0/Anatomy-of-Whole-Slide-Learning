# Competing-Risk Failure Modes

Competing risks are easy to misuse because several valid mathematical objects
sound like "risk."

## 1. Treating Competing Events As Censoring

If event $c'\ne c$ is treated as independent censoring for cause $c$, one
implicitly asks a counterfactual question:

```text
what would happen if the competing event process did not exist?
```

That may be useful for etiology, but it is not the observed probability of cause
$c$ occurring first.

The observed prediction target is:

```math
F_c(t\mid z)
=
\Pr(T\le t,J=c\mid z).
```

## 2. Reading Fine-Gray As Biological Hazard

Fine-Gray estimates a subdistribution hazard:

```math
\alpha_c(t\mid z)
=
\frac{dF_c(t\mid z)/dt}{1-F_c(t\mid z)}.
```

This is not the instantaneous rate of cause $c$ among event-free patients.

It is a regression device for the CIF.

## 3. Independent Binary Heads

Training one binary survival model per cause can violate probability
constraints:

```math
\sum_cF_c(t\mid z)>1.
```

A valid competing-risk model must ensure:

```math
\sum_cF_c(t\mid z)+S(t\mid z)=1.
```

PMF heads and cause-specific hazard systems enforce this naturally when built
correctly.

## 4. Shared Attention Across Causes

If:

```math
z_i=\sum_ja_{ij}h_{ij}
```

is shared for all event types, then all causes see the same morphology summary.

This can hide event-specific signals.

## 5. Event Imbalance

Some event types may be rare:

```math
|\{i:\Delta_i=c\}|\ll n.
```

Cause-specific heads then have weak gradients. Ranking terms can dominate
likelihood terms and produce poorly calibrated CIFs.

## 6. Ambiguous Clinical Labels

Pathology datasets may have:

```text
overall survival
disease-specific survival
progression-free survival
recurrence-free survival
```

These are not interchangeable. Progression-free survival often treats death as
an event, while recurrence-specific analysis may treat death as a competing
event.

## 7. Metric Mismatch

A model may optimize cause-specific likelihood but be evaluated by all-cause
C-index, or optimize CIF ranking but be interpreted as a cause-specific hazard
model.

The output object must match the metric:

```text
hazard score -> cause-specific ranking
CIF -> horizon-specific probability
PMF -> event-time/cause distribution
```

## Diagnostic Questions

1. What are the event types?
2. Is the target $\lambda_c(t)$, $F_c(t)$, or $p_{kc}$?
3. Are competing events removed, retained, or modeled jointly?
4. Do predicted CIFs sum to at most one?
5. Does event type enter only at the final head or earlier in the WSI readout?
6. Are rare causes sufficiently supported?

## Dense Summary

The main error is collapsing:

```math
\lambda_c(t\mid z),
\qquad
\alpha_c(t\mid z),
\qquad
F_c(t\mid z),
\qquad
p_{kc}(z)
```

into one word: risk. They are different mathematical objects.
