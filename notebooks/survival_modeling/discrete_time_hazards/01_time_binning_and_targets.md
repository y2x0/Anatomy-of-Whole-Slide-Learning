# Time Binning And Targets

Discrete-time models replace continuous event time with ordered intervals:

```math
0=\tau_0 < \tau_1 < \cdots < \tau_K.
```

The \(k\)-th interval is:

```math
I_k=(\tau_{k-1},\tau_k].
```

For a patient with observed time \(X_i\), define:

```math
\kappa_i
=
\min\{k:X_i\le \tau_k\}.
```

This is the interval containing the observed event or censoring time.

## Hazard Targets

The model predicts:

```math
h_{ik}
=
\Pr(T_i\in I_k \mid T_i>\tau_{k-1},z_i).
```

If \(\delta_i=1\), the event happened in interval \(\kappa_i\). Then:

```math
y_{ik}=0 \quad k<\kappa_i,
\qquad
y_{i\kappa_i}=1.
```

Intervals after the event are undefined because the patient is no longer at
risk.

If \(\delta_i=0\), the patient is censored in interval \(\kappa_i\). We know:

```math
T_i>X_i.
```

Depending on the convention, the patient contributes survival information up to
the censoring time, but no event label afterward.

## Survival From Hazards

The discrete survival probability through the end of interval \(k\) is:

```math
S_i(\tau_k)
=
\Pr(T_i>\tau_k\mid z_i)
=
\prod_{\ell=1}^{k}(1-h_{i\ell}).
```

The event probability in interval \(k\) is:

```math
\Pr(T_i\in I_k\mid z_i)
=
h_{ik}
\prod_{\ell=1}^{k-1}(1-h_{i\ell}).
```

This gives a full stepwise event-time distribution.

## Choosing Bins

Common choices:

```text
uniform time bins
quantile bins by observed event times
clinically meaningful cut points
dataset-specific follow-up windows
```

The binning is not cosmetic. It defines the task.

A coarse bin makes the model stable but temporally blunt. A fine bin gives more
detail but increases sparsity and censoring ambiguity.

## WSI Interpretation

For whole-slide survival:

```math
h_{ik}=\sigma(w_k^\top z_i+b_k).
```

Each time bin has its own readout vector \(w_k\). Therefore:

```math
w_k^\top z_i
```

is the slide evidence for risk in interval \(k\).

If:

```math
z_i=\sum_j a_{ij}h_{ij},
```

then:

```math
w_k^\top z_i
=
\sum_j a_{ij}w_k^\top h_{ij}.
```

Discrete hazards allow the patch-level risk evidence to be time-indexed.

## Key Takeaway

Discrete-time survival changes the output from:

```text
one risk score
```

to:

```text
a sequence of conditional event probabilities.
```
