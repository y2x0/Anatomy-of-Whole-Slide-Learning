# Distribution Representation Failure Modes

Distribution representations are clean, but they can hide important structure.

## 1. Same Distribution, Different Geometry

If a slide is represented only by:

```math
\mu_i
=
\frac{1}{n_i}\sum_j\delta_{h_{ij}},
```

then any permutation of patches gives the same representation.

Two slides may have identical morphology distributions:

```math
\mu_i=\mu_k
```

but different spatial arrangements:

```text
clustered tumor
diffuse tumor
invasive front
separate foci
```

The distribution sees prevalence, not layout.

## 2. Moment Collision

If:

```math
T(\mu)=\int h\,d\mu(h),
```

then different distributions can collide:

```math
T(\mu_i)=T(\mu_k),
\qquad
\mu_i\ne\mu_k.
```

Adding more moments reduces collisions but increases estimation noise.

## 3. Prototype Collapse

If prototypes are poorly chosen, many morphologies share one assignment:

```math
q_m(h_a)\approx q_m(h_b)
```

even when
```math
h_a
```
 and
```math
h_b
```
are biologically different.

Then:

```math
p_{im}
=
\frac{1}{n_i}\sum_jq_m(h_{ij})
```

mixes distinct tissue patterns into one count.

## 4. Rare Morphology Suppression

Prototype prevalence is normalized:

```math
p_{im}
=
\frac{\#\text{patches assigned to }m}{n_i}.
```

If a rare but diagnostic morphology occupies few patches:

```math
p_{im}\ll1,
```

then downstream models may ignore it unless the task head is sensitive to sparse
components.

## 5. Dataset-Level Prototype Bias

If prototypes are fit on a source cohort:

```math
\{c_m\}_{m=1}^{M}
=
\mathrm{Fit}(\{\mu_i:i\in\mathcal{D}_{\text{source}}\}),
```

then a target cohort with new morphology may be forced into old bins.

This creates projection error:

```math
h
\mapsto
(q_1(h),\ldots,q_M(h))
```

even when no prototype is appropriate.

## 6. Metric Mismatch

Retrieval or classification may use:

```math
\|p_i-p_k\|_2
```

even though prototype geometry matters. If prototypes
```math
c_a
```
 and
```math
c_b
```
are close,
moving mass from
```math
a
```
 to
```math
b
```
should be less severe than moving it to a distant
prototype
```math
c_r
```
.

Euclidean histogram distance ignores this unless the representation encodes
ground geometry.

## Diagnostic Questions

1. Is the task about prevalence or arrangement?
2. Which statistic
```math
T(\mu)
```
survives?
3. Can two clinically different slides collide under
```math
T
```
?
4. Are prototypes interpretable and stable across cohorts?
5. Does the distance respect geometry between morphologies?

## Dense Summary

The distribution assumption is:

```text
the clinically relevant signal is in morphology prevalence and distribution shape
```

It fails when:

```text
spatial organization, rare events, or out-of-vocabulary morphology matter more than distributional summary
```
