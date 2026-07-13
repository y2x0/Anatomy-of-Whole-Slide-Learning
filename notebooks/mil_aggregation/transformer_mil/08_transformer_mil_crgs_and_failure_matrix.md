# Transformer MIL C/R/G/S and Failure Matrix

| Family | Context `C` | Readout `R` | Geometry `G` | Surviving statistic | Failure |
|---|---|---|---|---|---|
| TransMIL | self-attention | CLS/global token | PPEG grid injection | relational global token | wrong layout or support approximation |
| exact transformer MIL | dense self-attention | pooling or CLS | optional order/coordinates | pairwise contextual statistic | quadratic cost and shortcut context |
| sampled transformer | sparse/landmark attention | sampled readout | sampler-defined support | approximate relational statistic | rare patch loss |

## Explanation Rule

```text
final attention row -> routing into readout;
full gradient -> local computational sensitivity;
token deletion -> recomputed model effect;
topology/layout swap -> geometry dependence.
```

The transformer family is not “attention pooling with more layers.” Its context
operator changes the object that the readout sees.

