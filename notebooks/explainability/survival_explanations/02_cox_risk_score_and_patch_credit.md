# Cox Risk Score and Patch Credit

## 1. Cox Output

The proportional-hazards model has

```math
h_i(t)=h_0(t)\exp(\eta_i),
\qquad
\eta_i=w^{\top}z_i.
```

The score `eta_i` orders relative risk; it is not an absolute hazard without
the baseline hazard `h_0`.

## 2. Additive Attention Readout

For attention-pooled patch states,

```math
z_i=\sum_{j=1}^{n_i}a_{ij}r_{ij},
\qquad
\sum_ja_{ij}=1.
```

Then

```math
\eta_i
=\sum_j a_{ij}w^{\top}r_{ij}.
```

The algebraic patch term

```math
\gamma_{ij}^{\mathrm{Cox}}
=a_{ij}w^{\top}r_{ij}
```

is exact signed credit for the current Cox score when the head is linear.

## 3. Attention Is Still Not Credit

`a_ij` alone is nonnegative routing. A patch can receive high attention but have
a negative projection onto `w`, lowering the risk score. Conversely, a low
attention patch can have a large signed projection.

## 4. Deletion Effect

With attention recomputed,

```math
\Delta_{ij}^{\mathrm{risk}}
=\eta_i-\eta_i^{(-j)}
```

includes renormalization and contextual changes. It equals the algebraic term
only when the rest of the computation is held fixed in the relevant way.

## 5. Cox Interpretation Boundary

If `gamma_ij` is positive, the patch raises the model's log-risk score relative
to the chosen zero baseline. It does not prove a tissue feature causes death,
nor does it identify a calibrated absolute survival difference.

