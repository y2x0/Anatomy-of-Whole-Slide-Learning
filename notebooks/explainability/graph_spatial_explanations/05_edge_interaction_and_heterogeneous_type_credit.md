# Edge Interaction and Heterogeneous Type Credit

HEAT assigns node types and Pearson-style edge attributes before heterogeneous
message passing.

## 1. Edge Attribute

For source `u` and target `v`, an edge feature can be the feature correlation

```math
r_{uv}
=\mathrm{corr}(x_u,x_v).
```

It is a similarity statistic, not a causal interaction. A high correlation can
reflect shared stain or encoder nuisance.

## 2. Edge Removal Effect

For edge `e=(u,v)`, define

```math
\Delta_e
=F(G)-F(G\setminus\{e\}).
```

If message normalization is rerun, deleting one edge also changes the relative
weights of the remaining incoming edges. A fixed message mask and a rebuilt
normalization therefore estimate different edge effects.

## 3. Type-Conditional Credit

Let `V_a` be nodes of type `a`. A type-level deletion contrast is

```math
\Delta_a
=F(G)-F(G\setminus V_a).
```

This measures the model's dependence on a whole entity class, but it also
changes graph density and type-wise pooling. It is not the sum of node effects
unless the model response is additive under those interventions.

## 4. Edge Interaction Score

For a pair of stain or node types `a,b`, aggregate edge-level weights:

```math
I_{ab}
=\frac{1}{|E_{ab}|}
\sum_{(u,v)\in E_{ab}}\beta_{uv},
```

where `beta_uv` is the relevant GAT edge attention. This is a routing summary.
It does not prove that cross-type interactions increase the class logit.

## 5. Correct Target

To claim type interaction explains a class score, use a target-specific contrast
or signed contribution. A type-pair attention average alone explains only how
messages were normalized.

