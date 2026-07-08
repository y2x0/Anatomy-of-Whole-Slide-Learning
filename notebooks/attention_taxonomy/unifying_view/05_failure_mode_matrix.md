# Failure Mode Matrix

Attention failure modes follow from the operator slot that fails.

## Matrix

| Slot | Failure | Mathematical Symptom |
|---|---|---|
| Score | shortcut compatibility | artifacts receive large scores |
| Score | weak separation | rare positives do not beat background |
| Normalization | dilution | mass spread over many irrelevant tokens |
| Normalization | collapse | one token receives nearly all mass |
| Support | missing evidence | true evidence has zero allowed weight |
| Support | wrong locality | necessary long-range interaction absent |
| Value | missing statistic | averaged values do not encode needed fact |
| Readout | bottleneck | many mechanisms compressed to one vector |
| Query | query collapse | multiple queries learn same measure |
| Geometry | wrong bias | positional prior favors wrong relation |
| Explanation | false localization | weight map confused with evidence |

## Score Failure

If:

```math
s_{\mathrm{artifact}}
>
s_{\mathrm{pathology}},
```

then attention follows artifact even with correct normalization.

## Normalization Failure

If:

```math
a_j
=
\frac{\exp(s_j)}{\sum_{\ell}\exp(s_\ell)},
```

then large bags can dilute moderate positive scores.

## Support Failure

If:

```math
M_{uv}=0,
```

then:

```math
a_{uv}=0
```

even when the score would have been high.

## Value Failure

If values are:

```math
r_j
=
W_V h_j,
```

and `h_j` lacks spatial scale or tissue-compartment information, then the
weighted sum cannot preserve that information.

## Readout Failure

A single readout:

```math
z
=
\sum_j a_jr_j
```

can erase multimodality. Multiple disease mechanisms may be forced into one
average.

## Dense Summary

Debug attention by asking:

```text
did the model score the right thing?
was the right thing allowed by support?
did normalization preserve enough mass?
did the value encode the needed statistic?
did the readout keep enough objects?
is the heat map being overclaimed?
```
