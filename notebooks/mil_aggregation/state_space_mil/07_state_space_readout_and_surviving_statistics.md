# State-Space Readout And Surviving Statistics

Let a state-space context operator produce:

```math
Y_i
=
\left[
y_{i1}^{\top};\ldots;y_{iL_i}^{\top}
\right]
\in
\mathbb{R}^{L_i\times d}.
```

The scan and the readout are separate. Different readouts expose different
statistics of the same compressed trajectory.

## 1. Coordinatewise Max

The S4MIL released model uses:

```math
[z_i^{\mathrm{max}}]_r
=
\max_{1\le t\le L_i}
[y_{it}]_r.
```

Max preserves extreme responses but discards their order and multiplicity. If
two sequences have the same coordinatewise maxima, the classifier receives the
same vector even when the extrema occur at different positions.

## 2. Mean

```math
z_i^{\mathrm{mean}}
=
\frac{1}{L_i}
\sum_{t=1}^{L_i}y_{it}.
```

This is a first moment of the state-space output trajectory. It is invariant to
permuting output rows after the scan, but not to permuting input rows before it:

```math
\mathrm{Mean}\left(\mathcal{C}_{\mathrm{SSM}}(PX_i)\right)
\neq
\mathrm{Mean}\left(\mathcal{C}_{\mathrm{SSM}}(X_i)\right).
```

## 3. Attention Readout

The released MambaMIL head scores contextualized tokens:

```math
s_{it}
=
f_{\theta}(y_{it}),
\qquad
\alpha_{it}
=
\frac{\exp(s_{it})}
{\sum_{u=1}^{L_i}\exp(s_{iu})}.
```

The slide statistic is:

```math
z_i^{\mathrm{attn}}
=
\sum_{t=1}^{L_i}
\alpha_{it}y_{it}.
```

This is a weighted first moment of order-conditioned states. The weight at t is
not an isolated patch importance score because y_it already contains a prefix
summary.

## 4. Final-State Readout

A strict final-state readout is:

```math
z_i^{\mathrm{final}}
=
s_{i,L_i}.
```

For a linear SSM:

```math
s_{i,L_i}
=
\sum_{k=1}^{L_i}
\overline A^{L_i-k}
\overline B u_{ik}.
```

This is a position-weighted trajectory summary. It is not a mean and does not
preserve each token equally.

## 5. Patch-Level Supervision

If every token has an auxiliary target:

```math
\mathcal{L}_{\mathrm{patch},i}
=
\sum_{t=1}^{L_i}
m_{it}
\ell_{\mathrm{patch}}
\left(q_{it},\widehat q_{it}\right).
```

With slide loss:

```math
\mathcal{L}_i
=
\mathcal{L}_{\mathrm{slide},i}
+
\lambda\mathcal{L}_{\mathrm{patch},i}.
```

The auxiliary term can prevent the state from encoding only a slide shortcut,
but it also changes the target statistic: tokens must be useful locally and
through the final slide readout.

## 6. Task Heads

Classification:

```math
\widehat p_i
=
\mathrm{softmax}
\left(
W_{\mathrm{cls}}z_i+b_{\mathrm{cls}}
\right).
```

Cox-style risk:

```math
\eta_i
=
w_{\mathrm{risk}}^{\top}z_i+b_{\mathrm{risk}}.
```

Discrete hazards:

```math
\widehat h_{ik}
=
\sigma(w_k^{\top}z_i+b_k),
\qquad
\widehat S_i(k)
=
\prod_{r=1}^{k}
(1-\widehat h_{ir}).
```

The same state can support different tasks, but each loss selects which
trajectory collisions matter.

## 7. C/R/G/S Placement

```text
C:
    ordered recurrent or convolutional state-space context

R:
    max for S4MIL, attention weighted mean for released MambaMIL, or another
    explicitly named sequence statistic

G:
    scan order, direction, segment transposition, and optional coordinates

S:
    slide label, survival bins, and optional patch labels
```
