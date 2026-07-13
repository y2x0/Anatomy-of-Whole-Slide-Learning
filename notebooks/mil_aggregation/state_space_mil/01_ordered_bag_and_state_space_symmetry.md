# Ordered Bag And State-Space Symmetry

## 1. The Representation Changes First

For slide i, patch embeddings are initially a multiset:

```math
\mathcal{B}_i
=
\left\{
h_{ij}\in\mathbb{R}^{d}:j=1,\ldots,n_i
\right\}.
```

An SSM requires an ordered sequence. Let sigma_i be a permutation:

```math
\sigma_i
\in
\mathfrak{S}_{n_i}.
```

The actual SSM input is:

```math
U_i
=
\begin{bmatrix}
h_{i,\sigma_i(1)}^{\top}\\
\vdots\\
h_{i,\sigma_i(n_i)}^{\top}
\end{bmatrix}
\in
\mathbb{R}^{n_i\times d}.
```

The full map is:

```math
Y_i
=
\mathcal{C}_{\theta}^{\mathrm{SSM}}(U_i),
\qquad
z_i
=
\mathcal{R}_{\theta}(Y_i).
```

Even if R is invariant over output rows, the scan has already used sigma_i to
define which token is earlier and which state receives memory.

## 2. Order Sensitivity

For a scalar recurrence:

```math
s_t
=
a s_{t-1}+b u_t,
\qquad
y_t
=
c s_t,
\qquad
s_0=0.
```

After two tokens:

```math
y_2
=
c\left(ab u_1+b u_2\right).
```

After swapping them:

```math
y_2^{\mathrm{swap}}
=
c\left(ab u_2+b u_1\right).
```

Unless the recurrence is degenerate or the inputs coincide:

```math
y_2
\neq
y_2^{\mathrm{swap}}.
```

The order is a geometry choice. Raster, Hilbert, learned, and file orders
define different hypothesis classes.

## 3. Equivariance Is Not Invariance

A sequence operator generally does not satisfy:

```math
\mathcal{C}_{\theta}^{\mathrm{SSM}}(PU)
=
P\mathcal{C}_{\theta}^{\mathrm{SSM}}(U).
```

A new permutation changes the ordered object:

```math
U_i^{\sigma'}
=
P_{\sigma'\leftarrow\sigma}U_i^\sigma.
```

Evaluating several orders and averaging is a possible ensemble. A single scan
does not regain set invariance merely because final pooling is a mean or max.

## 4. Padding Is An Input Convention

For batch length L_max, define:

```math
m_{it}
=
\mathbf{1}\left\{t\le n_i\right\}.
```

A masked recurrence can be written:

```math
s_{it}
=
m_{it}
\left(A_t s_{i,t-1}+B_tu_{it}\right)
+
(1-m_{it})s_{i,t-1}.
```

Without a mask or an equivalent safe convention, padding can alter the state
and slide prediction. MambaMIL pads when length is not divisible by its segment
size and de-pads after restoration.

## 5. Shape Map

For U_i with shape n_i by d:

```math
U_i
\in
\mathbb{R}^{n_i\times d}
\longmapsto
Y_i
\in
\mathbb{R}^{n_i\times d_y}.
```

For channel or head r, the finite state is:

```math
s_{i,t,r}
\in
\mathbb{R}^{N_r}.
```

The state is a compressed prefix statistic:

```math
(u_{i1},\ldots,u_{it})
\longmapsto
s_{i,t,r}.
```

N_r is the memory width, not the number of patches.

## 6. C/R/G/S Placement

```text
C:
    recurrent, structured convolutional, or selective state-space scan

R:
    max, mean, attention, final-state, or task-specific sequence reduction

G:
    scan order, optional coordinates, segment order, and direction

S:
    slide label, survival target, patch auxiliary label, or multitask objective
```
