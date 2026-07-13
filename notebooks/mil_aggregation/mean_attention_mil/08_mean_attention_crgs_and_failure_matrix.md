# Mean and Attention MIL C/R/G/S and Failure Matrix

| Method | Context `C` | Readout `R` | Surviving statistic | Main failure |
|---|---|---|---|---|
| mean MIL | none or inherited | uniform mean | first moment | rare-signal dilution |
| ABMIL | instance score only | softmax weighted mean | weighted first moment | attention collapse |
| CLAM | class-conditioned instance selection | class-specific attention | class-conditional first moment | pseudo-label shortcut |
| DSMIL | critical-instance relation | instance plus bag branch | extreme plus conditional mean | critical-instance switch |
| Additive MIL | optional instance map | sum of signed scores | additive evidence | missing interactions |
| DTFD-MIL | pseudo-bag tier | two-stage distillation | tiered evidence statistic | subset partition bias |

## Unified Map

```math
H_i
\xrightarrow{\mathcal C}
\widetilde H_i
\xrightarrow{\mathcal R}
z_i
\xrightarrow{\mathcal H}
\widehat y_i.
```

The first independent design question is not “which attention module?” It is
which bag statistic the label requires and which collisions the readout must
avoid.

## Operator Contracts

For transformed patch states u_ij, the principal readouts are

```math
z_i^{\mathrm{mean}}
=
\frac{1}{n_i}\sum_{j=1}^{n_i}u_{ij},
\qquad
z_i^{\mathrm{sum}}
=
\sum_{j=1}^{n_i}u_{ij}.
```

Attention replaces uniform mass with a learned probability measure:

```math
a_{ij}
=
\frac{\exp(s_{ij})}{\sum_{\ell=1}^{n_i}\exp(s_{i\ell})},
\qquad
z_i^{\mathrm{attn}}
=
\sum_{j=1}^{n_i}a_{ij}v_{ij}.
```

The normalized mean is replication-invariant, while the sum changes with
exposure. An attention readout can also be replication-invariant in value space
while changing its scores through the denominator. These are different
cardinality assumptions and should not be grouped under one generic “pooling.”

## Collision Tests

Let H and H' be two bags. A readout collision is

```math
\mathcal R(\mathcal C(H))
=
\mathcal R(\mathcal C(H')).
```

For mean pooling, duplicating every instance leaves z unchanged. For additive
pooling it multiplies the evidence under the same transformed states. For
attention, duplicating a state changes both the numerator and denominator, and
the resulting representation depends on whether the duplicated state receives
the same score. A method should therefore report the bag transformations under
which its statistic is intended to be invariant.

The failure matrix can be read as a design implication:

```math
\text{label assumption}
\longrightarrow
\text{surviving statistic}
\longrightarrow
\text{unavoidable collision class}.
```

Rare-positive MIL needs an upper-tail or witness statistic; distributed-burden
MIL needs a count-sensitive statistic; and multimodal or spatial MIL needs those
relations to enter C or G before the final readout.
