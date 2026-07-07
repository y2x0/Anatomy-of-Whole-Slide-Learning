# Failure Modes

No-geometry models fail when spatial arrangement is part of the target.

## Same Multiset, Different Layout

Construct:

```math
X_A
=
\{(h_j,c_j)\}_{j=1}^{n},
\qquad
X_B
=
\{(h_j,c'_j)\}_{j=1}^{n}.
```

The feature multiset is identical:

```math
\{h_j\}_{j=1}^{n}
=
\{h_j\}_{j=1}^{n}.
```

Any no-geometry model gives:

```math
f_{\varnothing}(X_A)
=
f_{\varnothing}(X_B).
```

If the true labels differ:

```math
y_A\ne y_B,
```

then the model class is misspecified.

## Interface Blindness

Many tissue patterns are relational:

```text
tumor near stroma
immune cells at invasive front
necrosis surrounded by viable tumor
gland architecture
regional heterogeneity
```

No-geometry models can detect the components but not their arrangement.

Formally, if the target depends on a pairwise statistic:

```math
T(X_i)
=
\sum_{j,k}
\mathbf{1}\{\|c_{ij}-c_{ik}\|\le r\}
\alpha(h_{ij},h_{ik}),
```

then a geometry-erasing projection loses the relevant variable:

```math
T(X_i)
\notin
\sigma(\{h_{ij}\}_{j=1}^{n_i}).
```

## False Interpretability

Attention heatmaps from no-geometry models are often interpreted spatially after
training. But the model did not reason with the map as geometry. It produced
weights:

```math
a_{ij}
=
a_\theta(h_{ij},H_i),
```

not:

```math
a_{ij}
=
a_\theta(h_{ij},c_{ij},G_i,H_i).
```

The displayed heatmap may localize high-scoring morphology, but it does not prove
that spatial relationships were used.

## Shortcut Safety And Shortcut Loss

Ignoring geometry can protect against coordinate shortcuts:

```math
P(Y\mid C)
\ne
P_{\mathrm{test}}(Y\mid C).
```

But it also removes legitimate spatial signal:

```math
P(Y\mid H,C)
\ne
P(Y\mid H).
```

The same design choice can improve robustness or destroy signal depending on
the data-generating process.

## Dense Summary

No-geometry fails when:

```math
I(Y;C\mid H)>0.
```

The model's invariance forces it to treat spatially different slides as the
same object.
