# Selective State Spaces And The Mamba Scan

Source:

```text
Gu and Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
https://arxiv.org/abs/2312.00752
```

## 1. Input-Dependent Selection

A time-invariant SSM uses one transition pair for all tokens:

```math
s_t
=
\overline A s_{t-1}
+
\overline B u_t.
```

Mamba makes parts of the update input-dependent:

```math
\Delta_t
=
\mathrm{softplus}
\left(
W_{\Delta}u_t+b_{\Delta}
\right),
```

```math
B_t
=
W_Bu_t,
\qquad
C_t
=
W_Cu_t.
```

For learned continuous transition A:

```math
\overline A_t
=
\exp(\Delta_t A),
```

```math
\overline B_t
=
(\Delta_t A)^{-1}
\left(
\exp(\Delta_t A)-I
\right)
\Delta_tB_t.
```

The selected recurrence is:

```math
s_t
=
\overline A_t s_{t-1}
+
\overline B_tu_t,
\qquad
y_t
=
C_ts_t+D u_t.
```

The current token can decide how much prior state to retain and how strongly it
enters or reads the memory.

## 2. Causality

The state is a function of the prefix:

```math
s_t
=
F_{\theta}(u_1,\ldots,u_t).
```

For a strictly causal scan:

```math
\frac{\partial y_t}{\partial u_k}
=
0
\qquad
\text{for }k>t.
```

A global receptive field means every previous token can influence later tokens
through compressed state. It does not mean all tokens interact symmetrically.

## 3. Selective Forgetting

In a scalar state mode:

```math
s_t
=
\alpha_t s_{t-1}
+
\beta_tu_t.
```

After two steps:

```math
s_2
=
\alpha_2\alpha_1s_0
+
\alpha_2\beta_1u_1
+
\beta_2u_2.
```

A later token can preserve or attenuate earlier evidence by changing its
retention factor. This differs from a fixed convolution kernel.

## 4. Scan Complexity

For state width N, feature width d, and length L, a straightforward recurrent
implementation scales as:

```math
\mathrm{cost}_{\mathrm{scan}}
=
O(LdN).
```

Hardware-aware parallel scans improve training throughput while preserving the
associative recurrence. This should not be compared with transformer O(L^2)
without stating width, state size, and implementation.

## 5. S4 Versus Mamba

| Property | S4/S4D | Mamba |
| --- | --- | --- |
| transition | time-invariant after discretization | input-dependent selection |
| training computation | structured convolution | hardware-aware selective scan |
| memory statistic | fixed linear filter basis | token-conditioned retain/write/read |
| order | one-way scan is order-sensitive | one-way scan is order-sensitive |
| WSI anchor | S4MIL | MambaMIL |

Mamba does not remove the need to choose a slide order. It adapts the state
update after that order has been supplied.
