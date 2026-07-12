# WSI Attention Counterexamples And C/R/G/S

## Counterexample One: High Attention, Zero Effect

Let values be:

```math
v_1=v_2=v.
```

For any attention distribution:

```math
z
=
\alpha_1v
+
\alpha_2v
=
v.
```

The map can move all mass between patches without changing prediction.

## Counterexample Two: High Attention, Negative Evidence

For class head `w`:

```math
\alpha_1=0.9,
\qquad
w^{\top}v_1=-1,
```

```math
\alpha_2=0.1,
\qquad
w^{\top}v_2=20.
```

Direct contributions are:

```math
c_1=-0.9,
\qquad
c_2=2.
```

The lower-attention patch supports the class more strongly.

## Counterexample Three: Duplicate Tissue

If one diagnostic feature appears in `m` near-identical patches, attention can
split mass:

```math
\alpha_j
\approx
\frac{A}{m}
```

for each duplicate. No single patch appears important despite large total
evidence from that morphology.

## Counterexample Four: Context Mediator

Patch `i` can alter another patch's contextual representation:

```math
h_j'
=
\mathcal{C}_j(H).
```

Even if final readout attention on `i` is low:

```math
\alpha_i\approx0,
```

the mediated path can be large:

```math
\frac{\partial F}
{\partial h_j'}
\frac{\partial h_j'}
{\partial h_i}
\ne
0.
```

## C/R/G/S Matrix

| Method | Context `C` | Readout `R` | Geometry `G` | Explanation claim `S` |
|---|---|---|---|---|
| ABMIL attention | independent patch encoding | normalized weighted mean | simplex weights and value geometry | relative representation mass |
| CLAM-MB | independent patch encoding plus class branches | class-specific weighted means | branch-specific attention and clustering | class-conditional patch ranking |
| Transformer attention | repeated first moment | simplex weights and tokens | row-stochastic attention products | routing between contextual tokens |
| Additive MIL | patch evidence map | unnormalized sum | signed class-score addition | exact patch credit in score space |

## Diagnostic Checklist

Before calling attention an explanation, specify:

```text
the score or decision being explained
whether values and head alignment are included
whether context-mediated paths exist
whether removal recomputes normalization and context
whether alternative maps preserve the prediction
whether maps are stable across seeds and nuisance transforms
whether localization and faithfulness are evaluated separately
```

## Unified Failure Principle

```math
\text{attention score}
\longrightarrow
\text{normalized weight}
\longrightarrow
\text{value aggregation}
\longrightarrow
\text{head score}
\longrightarrow
\text{rendered heatmap}.
```

Each arrow can break the identification of weight with contribution. Additive
MIL removes several arrows by designing the class score itself as a sum of
signed patch evidence.
