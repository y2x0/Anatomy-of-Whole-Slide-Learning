# Linear State-Space Discretization And Convolution

Structured SSMs expose two equivalent computations: a recurrent scan and a
structured convolution.

## 1. Continuous System

For input u(t), state x(t), and output y(t):

```math
\frac{d x(t)}{d t}
=
A x(t)+B u(t),
```

```math
y(t)
=
C x(t)+D u(t).
```

The dimensions are:

```math
x(t)\in\mathbb{R}^{N},
\qquad
u(t)\in\mathbb{R}^{d_{\mathrm{in}}},
\qquad
y(t)\in\mathbb{R}^{d_{\mathrm{out}}}.
```

N is a memory-basis dimension, distinct from sequence length L and patch width
d.

## 2. Zero-Order-Hold Discretization

With step size Delta:

```math
\overline A
=
\exp(\Delta A),
```

```math
\overline B
=
(\Delta A)^{-1}
\left(\exp(\Delta A)-I\right)
\Delta B.
```

The discrete recurrence is:

```math
x_t
=
\overline A x_{t-1}
+
\overline B u_t,
\qquad
y_t
=
C x_t+D u_t.
```

D u_t is a direct skip from the current patch and carries no prefix memory.

## 3. Prefix Expansion

With x_0 equal to zero:

```math
x_t
=
\sum_{k=1}^{t}
\overline A^{t-k}\overline B u_k.
```

Therefore:

```math
y_t
=
\sum_{k=1}^{t}
C\overline A^{t-k}\overline B u_k
+
D u_t.
```

The influence kernel at lag ell is:

```math
K_{\ell}
=
C\overline A^{\ell}\overline B.
```

The scan is causal: a future token cannot alter an earlier output unless a
reverse-direction scan is added.

## 4. Convolutional Form

For length L, form:

```math
\overline K
=
\left(
C\overline B,
C\overline A\overline B,
\ldots,
C\overline A^{L-1}\overline B
\right).
```

Then:

```math
Y
=
\overline K * U
+
D U.
```

The recurrent and convolutional forms represent the same linear time-invariant
map. The convolution exposes parallel training; the recurrence exposes memory
and streaming inference.

## 5. Stability And Memory

If:

```math
\rho(\overline A)<1,
```

then old inputs decay in a stable linear mode. For eigenvalue lambda:

```math
\left\|
C\overline A^{\ell}\overline B
\right\|
\approx
O\left(|\lambda|^{\ell}\right).
```

Long memory requires modes near unit magnitude, while modes too close to one
can make optimization and numerical stability delicate. The SSM is a finite
basis projection, not a lossless archive of its prefix.

## 6. Multi-Channel WSI Input

For:

```math
U_i
=
\left[u_{i,t,r}\right]
\in
\mathbb{R}^{L_i\times d},
```

a channelwise representation is:

```math
Y_{i,:,r}
=
\mathcal{S}_{\theta,r}
\left(U_{i,:,r}\right),
\qquad
r=1,\ldots,d.
```

The S4MIL implementation follows an S4D block with a pointwise output
transform that mixes channels.

## 7. Complexity Claim

For fixed state size and width, a scan or structured convolution can be linear
in sequence length:

```math
\mathrm{cost}_{\mathrm{SSM}}(L)
=
O(L).
```

This is length scaling. State size, channels, kernel implementation, and
hardware still determine constants and memory cost.
