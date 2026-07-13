# Additive MIL Signed Evidence

## 1. Additive Score

Additive MIL represents a bag score as

```math
F_{ic}=b_c+\sum_{j=1}^{n_i}f_c(h_{ij}).
```

For a linear instance score,

```math
f_c(h_{ij})=w_c^{\top}h_{ij}.
```

## 2. Exact Instance Credit

```math
\gamma_{ijc}=f_c(h_{ij}),
\qquad
F_{ic}-b_c=\sum_j\gamma_{ijc}.
```

Positive and negative evidence are preserved explicitly. This differs from
attention, where normalized routing can obscure sign.

## 3. Shapley Boundary

For an additive set function with zero baseline, the Shapley value equals the
instance term:

```math
\varphi_{ijc}=f_c(h_{ij}).
```

With nonlinear context or interactions, the equality fails and coalition values
must account for interaction terms.

## 4. Failure

Additivity assumes the bag score is decomposable over instances. It can miss
spatial interactions, but its score accounting is more faithful than an
attention-only heatmap when the model is actually additive.

