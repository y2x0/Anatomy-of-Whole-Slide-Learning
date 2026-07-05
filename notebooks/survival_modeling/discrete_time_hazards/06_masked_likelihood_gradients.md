# Masked Likelihood And Gradients

Discrete survival can be written as masked binary cross entropy, but the mask is
not a trick. It is the likelihood's representation of censoring.

Let:

```math
h_{ik}=\sigma(g_{ik}).
```

Define:

```math
m_{ik}\in\{0,1\}
```

to indicate whether interval \(k\) is observed for subject \(i\), and:

```math
y_{ik}\in\{0,1\}
```

to indicate an event in interval \(k\).

The loss:

```math
\mathcal L_i
=
-
\sum_{k=1}^{K}
m_{ik}
\left[
y_{ik}\log h_{ik}
+
(1-y_{ik})\log(1-h_{ik})
\right]
```

is a censored likelihood if \(m_{ik}\) is constructed from the observed
at-risk process.

## Event Case

If subject \(i\) fails in interval \(r\):

```math
y_{ik}
=
\mathbf 1[k=r],
```

and:

```math
m_{ik}
=
\mathbf 1[k\le r].
```

Then:

```math
\mathcal L_i
=
-
\sum_{k<r}\log(1-h_{ik})
-
\log h_{ir}.
```

This is:

```math
-\log
\left[
h_{ir}
\prod_{k<r}(1-h_{ik})
\right].
```

## Censored Case

If subject \(i\) is censored after interval \(r\):

```math
y_{ik}=0,
\qquad
m_{ik}=\mathbf 1[k\le r].
```

Then:

```math
\mathcal L_i
=
-
\sum_{k\le r}\log(1-h_{ik})
=
-\log S_i(\tau_r).
```

The model is trained only on survival up to the observed censoring horizon.

## Inside-Interval Censoring

If censoring occurs at \(X_i\in(\tau_{r-1},\tau_r)\), there are two common masks:

Conservative:

```math
m_{ik}=\mathbf 1[k<r].
```

Boundary approximation:

```math
m_{ik}=\mathbf 1[k\le r].
```

The second assumes event-free status through \(\tau_r\), which may be false.
The first discards partial interval information.

## Gradients

For one interval:

```math
\ell_{ik}
=
-
m_{ik}
\left[
y_{ik}\log\sigma(g_{ik})
+
(1-y_{ik})\log\sigma(-g_{ik})
\right].
```

Then:

```math
\frac{\partial\ell_{ik}}{\partial g_{ik}}
=
m_{ik}(\sigma(g_{ik})-y_{ik})
=
m_{ik}(h_{ik}-y_{ik}).
```

For a linear head:

```math
g_{ik}=w_k^\top z_i+b_k,
```

the gradients are:

```math
\frac{\partial\mathcal L}{\partial w_k}
=
\sum_i m_{ik}(h_{ik}-y_{ik})z_i,
```

```math
\frac{\partial\mathcal L}{\partial z_i}
=
\sum_km_{ik}(h_{ik}-y_{ik})w_k.
```

The slide representation receives a weighted sum of time-specific error
directions.

## WSI Attention Gradient

If:

```math
z_i=\sum_j a_{ij}h_{ij},
```

and attention is held fixed:

```math
\frac{\partial\mathcal L_i}{\partial h_{ij}}
=
a_{ij}
\sum_km_{ik}(h_{ik}-y_{ik})w_k.
```

Define time-aggregated survival error vector:

```math
e_i
=
\sum_km_{ik}(h_{ik}-y_{ik})w_k.
```

Then:

```math
\frac{\partial\mathcal L_i}{\partial h_{ij}}
=
a_{ij}e_i.
```

So all patches receive the same error direction, scaled by attention, unless
attention or the readout is time-dependent.

## Time-Dependent Attention Gradient

If:

```math
z_{ik}=\sum_j a_{ijk}h_{ij},
\qquad
g_{ik}=w_k^\top z_{ik}+b_k,
```

then:

```math
\frac{\partial\mathcal L_i}{\partial h_{ij}}
=
\sum_k
m_{ik}(h_{ik}-y_{ik})
\frac{\partial g_{ik}}{\partial h_{ij}}.
```

Holding \(a_{ijk}\) fixed:

```math
\frac{\partial\mathcal L_i}{\partial h_{ij}}
=
\sum_k
m_{ik}(h_{ik}-y_{ik})
a_{ijk}w_k.
```

Now patches can receive different risk gradients for different time horizons.

## Mini-Batch Shape

Unlike Cox, the discrete hazard likelihood decomposes over subjects:

```math
\mathcal L=\sum_i\mathcal L_i.
```

There is no risk-set denominator coupling patients. This makes stochastic
optimization easier for WSI.

But the time-bin gradients can be sparse:

```math
m_{ik}=0
\quad\Rightarrow\quad
\frac{\partial\ell_{ik}}{\partial g_{ik}}=0.
```

Late bins often receive little signal under heavy censoring.

## Dense Summary

```math
\begin{aligned}
\mathcal L_i
&=
-
\sum_km_{ik}
[y_{ik}\log h_{ik}+(1-y_{ik})\log(1-h_{ik})],\\
\frac{\partial\mathcal L_i}{\partial g_{ik}}
&=
m_{ik}(h_{ik}-y_{ik}),\\
\frac{\partial\mathcal L_i}{\partial z_i}
&=
\sum_km_{ik}(h_{ik}-y_{ik})w_k.
\end{aligned}
```

The mask is the mathematical object that prevents censored future intervals from
becoming false negatives.
