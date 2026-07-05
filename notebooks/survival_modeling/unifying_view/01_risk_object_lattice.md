# Risk Object Lattice

Survival models differ by what object they output.

## Scalar Risk

```math
\eta_i=f(z_i).
```

Defines ordering. Needs baseline recovery for calibrated survival under Cox.

## Discrete Hazards

```math
h_{ik}
=
\Pr(T_i\in I_k\mid T_i>\tau_{k-1},z_i).
```

Defines:

```math
S_i(\tau_k)=\prod_{\ell\le k}(1-h_{i\ell}).
```

## Continuous Hazard

```math
\lambda_i(t)=\lambda(t\mid z_i).
```

Defines:

```math
S_i(t)=\exp\left[-\int_0^t\lambda_i(u)\,du\right].
```

## Survival Curve

```math
S_i(t)=\Pr(T_i>t\mid z_i).
```

Must satisfy:

```math
S_i(0)=1,
\qquad
S_i(t)\ \text{nonincreasing},
\qquad
0\le S_i(t)\le1.
```

## Event-Time PMF

```math
p_{ik}=\Pr(T_i\in I_k\mid z_i).
```

Defines:

```math
F_i(\tau_k)=\sum_{\ell\le k}p_{i\ell}.
```

## Competing-Risk CIF

```math
F_{ic}(t)=\Pr(T_i\le t,J_i=c\mid z_i).
```

Must satisfy:

```math
\sum_cF_{ic}(t)\le1.
```

## Lattice

```text
eta
    -> ordering only

h_k
    -> S_k -> F_k

lambda(t)
    -> Lambda(t) -> S(t) -> F(t), f(t)

p_k
    -> F_k, S_k

p_kc
    -> F_c,k, S_k

F_c(t)
    -> competing-risk cumulative incidence
```

The model's output object determines what losses and metrics are valid.
