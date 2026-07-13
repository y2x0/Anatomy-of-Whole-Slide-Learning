# State-Space MIL Failure Modes And C/R/G/S

## 1. Same Bag, Different Prediction

Take a two-token scalar bag:

```math
\mathcal{B}
=
\{u_1,u_2\},
\qquad
u_1\neq u_2.
```

For recurrence and final-state readout:

```math
s_t
=
a s_{t-1}+b u_t,
\qquad
z=s_2.
```

The two orders produce:

```math
z_{12}
=
ab u_1+b u_2,
\qquad
z_{21}
=
ab u_2+b u_1.
```

Their difference is:

```math
z_{12}-z_{21}
=
b(a-1)(u_1-u_2).
```

An SSM assigns different representations to the same unordered bag unless the
recurrence is degenerate. This is useful when the order is meaningful and a
failure when the order is arbitrary.

## 2. Order Geometry Failure

Let c_it be physical coordinates and sigma_i the scan order. Define:

```math
D(\sigma_i)
=
\frac{1}{|E_i^{\mathrm{spatial}}|}
\sum_{(u,v)\in E_i^{\mathrm{spatial}}}
\left|\sigma_i(u)-\sigma_i(v)\right|.
```

High distortion separates spatial neighbors by many state updates. SR-Mamba
adds a structured second scan, but its segment size R is still an inductive
choice rather than a proof of tissue geometry.

## 3. Finite-State Compression

The prefix map is:

```math
F_t:
\mathbb{R}^{td}
\longrightarrow
\mathbb{R}^{N}.
```

Distinct prefixes can collide:

```math
F_t(u_{1:t})
=
F_t(u_{1:t}^{\prime}).
```

S4's structured basis and Mamba's selective gating change which collisions are
likely; neither makes the state a lossless archive of all patches.

## 4. Causal Blindness To Unscanned Patches

For a one-way scan:

```math
\frac{\partial y_t}{\partial u_k}
=
0
\qquad
\text{when }k>t.
```

A late rare positive cannot alter earlier token states. Bidirectional Mamba and
SR-Mamba add paths but do not eliminate direction and order sensitivity.

## 5. Padding Leakage

If padded tokens are not inert:

```math
\mathcal{C}_{\mathrm{SSM}}(P_i,M_i)
\neq
\mathcal{C}_{\mathrm{SSM}}(P_i).
```

The mask and de-padding operation are part of the representation contract, not
a cosmetic batching detail.

## 6. Max Readout Collision

For output sequences Y and Y':

```math
\max_t Y_{t,r}
=
\max_t Y'_{t,r}
\qquad
\text{for every }r
```

implies identical max-pooled vectors even when maximizing locations and
trajectory order differ. Max is sensitive to extremes but blind to multiplicity
and position after context.

## 7. Attention Credit Is Contextual

For MambaMIL:

```math
z_i
=
\sum_t\alpha_{it}y_{it}.
```

The total input-patch effect contains both score and state paths:

```math
\frac{\partial z_i}{\partial u_{ik}}
=
\sum_{t\ge k}
\left[
\frac{\partial\alpha_{it}}{\partial u_{ik}}y_{it}
+
\alpha_{it}
\frac{\partial y_{it}}{\partial u_{ik}}
\right].
```

The attention weight alpha_it alone is not the total effect of patch k. Deletion
must recompute the scan and the readout.

## 8. Design Matrix

| Family | C: context | R: readout | G: induced geometry | Surviving statistic | Failure |
| --- | --- | --- | --- | --- | --- |
| S4MIL | structured S4D convolution | coordinatewise max | inherited patch/grid order | extreme state response per channel | order artifact and max collision |
| MambaMIL | selective causal Mamba | attention weighted mean in released code | original order plus optional SR order | weighted first moment of order-conditioned states | direction and finite-state compression |
| SR-Mamba | parallel original/reordered scans | downstream aggregation after branch fusion | segment-transposed path plus original path | fused multi-order trajectory | segment-size and padding dependence |

## 9. C/R/G/S Rule

`text
C:
    state is a compressed prefix statistic, not a set of independent patches

R:
    name the exact reduction after the scan; max, mean, attention, and final
    state preserve different information

G:
    the order is a modeling assumption and should be reported with the method

S:
    slide and patch supervision determine whether the state preserves global,
    local, or risk-relevant evidence
`

The concise description is:

```math
\boxed{
\text{slide statistic}
=
\mathcal{R}
\left(
\text{finite-state scan}
\left(
\text{ordered patch sequence}
\right)
\right)
}.
```
