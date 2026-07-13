# Discrete Hazard and Horizon Explanations

## 1. Discrete Output

For bins `tau_1 < ... < tau_K`, a discrete model predicts

```math
h_{ik}=P(T_i\in(\tau_{k-1},\tau_k]\mid T_i>\tau_{k-1},Z_i).
```

The survival probability at bin `k` is

```math
S_i(\tau_k)=\prod_{r=1}^{k}(1-h_{ir}).
```

## 2. Horizon-Specific Credit

If the model has logits `eta_ik` and a linear patch readout per horizon,

```math
\eta_{ik}=\sum_j a_{ijk}w_k^{\top}r_{ij},
```

then patch credit differs by horizon:

```math
\gamma_{ijk}=a_{ijk}w_k^{\top}r_{ij}.
```

If attention is shared, the head remains horizon-specific. A single heatmap
cannot automatically explain every bin.

## 3. Survival-Function Effect

At horizon `tau_m`, a patch intervention changes survival by

```math
\Delta_{ij}^{S}(\tau_m)
=S_i(\tau_m)-S_i^{(-j)}(\tau_m).
```

The same patch can have different effects at early and late horizons because
the product contains all preceding hazards.

## 4. Hazard Versus Survival

Increasing one early hazard can reduce every later survival value. A heatmap
for `h_i,m` must not be labeled a heatmap for `S_i(tau_m)` without specifying
the transformation and sign.

## 5. Censoring

Only bins known to be traversed before censoring contribute to the observed
likelihood. An explanation of the fitted output remains defined for a censored
patient, but evaluation of event-specific faithfulness must respect censoring.

