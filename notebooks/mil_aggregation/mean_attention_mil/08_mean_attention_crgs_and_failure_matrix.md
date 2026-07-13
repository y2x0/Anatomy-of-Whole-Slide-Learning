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

