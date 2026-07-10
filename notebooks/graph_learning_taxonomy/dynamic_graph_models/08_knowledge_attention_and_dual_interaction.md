# Knowledge Attention And Dual Interaction

This note derives local knowledge-aware attention, contrasts it with GAT, and writes the additive and multiplicative fusion channels in nodewise and matrix form.

Source:

```text
Li et al., Dynamic Graph Representation with Knowledge-aware Attention for
Histopathology Whole Slide Image Analysis, CVPR 2024.
https://arxiv.org/abs/2403.07719
Official implementation: https://github.com/WonderLandxD/WiKG
```

## Local Attention Distribution

Normalize triplet compatibilities only over selected sources:

```math
\pi_{vu}
=
\frac{
\exp(a_{vu})
}{
\sum_{j\in\mathcal{N}_K(v)}
\exp(a_{vj})
},
\qquad
u\in\mathcal{N}_K(v).
```

For each target:

```math
\pi_{vu}>0,
```

```math
\sum_{u\in\mathcal{N}_K(v)}
\pi_{vu}
=
1.
```

The neighborhood message is:

```math
m_v
=
\sum_{u\in\mathcal{N}_K(v)}
\pi_{vu}t_u
\in
\mathbb{R}^{d}.
```

Conditional on the selected support, `m_v` lies in the convex hull of selected
tail vectors:

```math
m_v
\in
\mathrm{conv}
\left(
\{t_u:u\in\mathcal{N}_K(v)\}
\right).
```

This convex-hull restriction applies to the pre-fusion message, not to the
updated node state after learned affine maps and nonlinearities.

## Distinction From GAT

A single-head GAT score has the generic form:

```math
a_{vu}^{\mathrm{GAT}}
=
\mathrm{LeakyReLU}
\left(
[q_v\Vert t_u]c
\right)
```

on a pre-existing support.

WiKG instead uses:

```math
a_{vu}^{\mathrm{WiKG}}
=
t_u
\left[
\tanh(q_v+r_{vu})
\right]^{\top},
```

where `r_vu` is itself generated from the head-tail selection score. The
attention kernel and graph construction are coupled.

The distinction is:

```text
GAT:
    learn edge weights on supplied support

WiKG:
    learn directed support, construct a relation state, then learn a second
    distribution on that support
```

## Dual-Interaction Fusion

WiKG combines the target head and neighborhood message through additive and
multiplicative channels:

```math
h_v^{+}
=
\phi_1
\left(
\left(q_v+m_v\right)W_1+b_1
\right)
+
\phi_2
\left(
\left(q_v\odot m_v\right)W_2+b_2
\right).
```

Under the row-vector convention:

```math
q_v,m_v
\in
\mathbb{R}^{d},
```

```math
W_1,W_2
\in
\mathbb{R}^{d\times d},
```

```math
b_1,b_2
\in
\mathbb{R}^{d}.
```

The released implementation uses LeakyReLU for both `phi_1` and `phi_2`.

The two channels preserve different interactions.

Additive channel:

```math
q_v+m_v
```

contains first-order target and neighborhood information but does not identify
coordinatewise agreement by itself.

Multiplicative channel:

```math
q_v\odot m_v
```

contains coordinatewise second-order interaction in the learned basis.

For coordinate `r`:

```math
\left[
q_v\odot m_v
\right]_r
=
[q_v]_r[m_v]_r.
```

Because the product is basis-dependent, it should not automatically be read as
a biological interaction. It is a learned feature-coordinate interaction.

## Matrix Form

Let the selected local attention matrix be:

```math
\Pi[v,u]
=
\begin{cases}
\pi_{vu},
&
u\in\mathcal{N}_K(v),
\\
0,
&
u\notin\mathcal{N}_K(v).
\end{cases}
```

Then:

```math
M
=
\Pi T
\in
\mathbb{R}^{n\times d}.
```

The graph context update is:

```math
H^{+}
=
\phi_1
\left(
\left(Q+M\right)W_1
+
\mathbf{1}_n b_1^{\top}
\right)
+
\phi_2
\left(
\left(Q\odot M\right)W_2
+
\mathbf{1}_n b_2^{\top}
\right).
```

Dropout is applied to the updated node-state matrix in the released
implementation before graph readout.
