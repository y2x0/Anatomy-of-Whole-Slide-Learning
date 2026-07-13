# Attention Pooling Failure Modes

Attention pooling is flexible, but its flexibility is still constrained by the
fact that it returns one weighted average.

## 1. Attention Collapse

If one score dominates:

```math
s_{ij^\star}
\gg
s_{ij}
\quad
j\ne j^\star,
```

then:

```math
a_{ij^\star}\approx 1,
\qquad
z_i\approx v_{ij^\star}.
```

The model becomes close to max pooling. This is useful for sparse-positive
signals but brittle when the selected patch is noisy or artifactual.

## 2. Diffuse Attention

If scores are nearly equal:

```math
s_{ij}\approx c,
```

then:

```math
a_{ij}\approx \frac{1}{n_i}.
```

The model collapses back toward mean pooling. Rare positive evidence is diluted
unless the value map amplifies it before averaging.

## 3. Non-Identifiability

Different attention-value pairs can yield the same slide embedding:

```math
\sum_j a_jv_j
=
\sum_j \widetilde a_j\widetilde v_j.
```

Therefore attention weights are not uniquely determined by the final prediction.
Two models can agree on slide labels while producing different heatmaps.

## 4. High Attention Is Not Positive Evidence

For class vector
```math
q
```
:

```math
o_i
=
q^\top z_i
=
\sum_j a_{ij}q^\top v_{ij}.
```

Patch
```math
j
```
contributes positively only if:

```math
q^\top v_{ij}>0.
```

A large
```math
a_{ij}
```
 with negative
```math
q^\top v_{ij}
```
is high-weight negative evidence.
This is one reason attention maps should not be read as instance labels without
additional assumptions.

## 5. Missing Interactions

ABMIL attention scores instances independently:

```math
s_{ij}=s_\theta(h_{ij}).
```

If the label depends on pairwise structure:

```math
y_i
=
f(h_{ia},h_{ib})
```

and neither
```math
h_{ia}
```
 nor
```math
h_{ib}
```
is individually informative, independent
attention may fail. A transformer, graph, or state-space context operator must
encode the interaction before pooling.

## 6. Critical-Instance Amplification

For DSMIL-style pooling, a false critical instance:

```math
j^\star
=
\arg\max_j s_j
```

can define the wrong query. Then attention retrieves patches similar to a false
positive:

```math
a_j
\propto
\exp(q_{j^\star}^\top q_j).
```

This failure is worse than selecting one wrong patch: it can recruit an entire
wrong morphology cluster.

## Diagnostic Questions

```text
1. What is the effective number of attended instances?
2. Are high-attention patches positive evidence or merely influential values?
3. Does the label require interactions that happen before pooling?
4. Do repeated trainings produce stable attention maps?
5. Does removing top-attention tissue change prediction as expected?
```

## Dense Summary

Attention pooling fails when the learned measure:

```math
\nu_{i,\theta}
=
\sum_j a_{ij}\delta_{v_{ij}}
```

places mass on the wrong statistic. The heatmap is a trace of the readout, not a
proof of biological causality.
