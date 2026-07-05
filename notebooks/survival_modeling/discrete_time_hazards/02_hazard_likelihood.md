# Discrete Hazard Likelihood

For interval hazards:

```math
h_{ik}
=
\Pr(T_i\in I_k \mid T_i>\tau_{k-1},z_i).
```

Surviving through interval \(k\) has probability:

```math
1-h_{ik}.
```

## Event Likelihood

If patient \(i\) has an event in interval \(\kappa_i\), then the patient must
survive all previous intervals and fail in interval \(\kappa_i\):

```math
p_i
=
\left[
\prod_{k=1}^{\kappa_i-1}(1-h_{ik})
\right]
h_{i\kappa_i}.
```

The log likelihood is:

```math
\log p_i
=
\sum_{k=1}^{\kappa_i-1}\log(1-h_{ik})
+
\log h_{i\kappa_i}.
```

## Censored Likelihood

If patient \(i\) is censored at interval \(\kappa_i\), the observed information
is survival until censoring:

```math
p_i
=
\prod_{k=1}^{\kappa_i}(1-h_{ik})
```

under the convention that the patient is known event-free through the censoring
interval boundary. If censoring occurs inside the interval, some implementations
only include intervals fully before censoring.

The log likelihood is:

```math
\log p_i
=
\sum_{k=1}^{\kappa_i}\log(1-h_{ik}).
```

## Unified Masked Binary Form

Define an at-risk mask:

```math
m_{ik}=1
\quad\text{if interval } k \text{ is observed for patient } i.
```

Define event target:

```math
y_{ik}=1
\quad\text{if patient } i \text{ fails in interval } k.
```

Then a common discrete hazard loss is:

```math
\mathcal{L}_i
=
-
\sum_{k=1}^K
m_{ik}
\left[
y_{ik}\log h_{ik}
+
(1-y_{ik})\log(1-h_{ik})
\right].
```

This looks like binary cross entropy, but the mask encodes survival/censoring
structure.

## Relation To Survival

Once hazards are predicted:

```math
S_i(\tau_k)
=
\prod_{\ell=1}^{k}(1-h_{i\ell}).
```

The cumulative incidence by interval \(k\) is:

```math
F_i(\tau_k)
=
1-S_i(\tau_k).
```

The expected event time can be approximated by:

```math
\mathbb{E}[T_i\mid z_i]
\approx
\sum_{k=1}^{K}
\bar{\tau}_k
\left[
h_{ik}\prod_{\ell=1}^{k-1}(1-h_{i\ell})
\right],
```

where \(\bar{\tau}_k\) is an interval representative.

## Gradient Shape

If:

```math
h_{ik}=\sigma(g_{ik}),
```

then the masked BCE gradient is:

```math
\frac{\partial \mathcal{L}_i}{\partial g_{ik}}
=
m_{ik}(h_{ik}-y_{ik}).
```

The mask determines which time intervals produce gradient.

Compared with Cox, this is less globally coupled across patients. Each patient
contributes a time-indexed supervised sequence.

## Key Takeaway

Discrete hazard likelihoods make censoring explicit through masks:

```text
observed intervals contribute, unobserved future intervals do not.
```

The price is that time binning becomes part of the model.
