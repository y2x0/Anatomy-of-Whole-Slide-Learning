# Mean Failure Modes

Mean pooling fails when the first moment is the wrong statistic.

## 1. Sparse Positive Dilution

Suppose a small positive set
```math
P_i
```
contains the diagnostic patches:

```math
P_i\subset\{1,\ldots,n_i\},
\qquad
|P_i|\ll n_i.
```

Mean pooling gives:

```math
z_i
=
\frac{1}{n_i}
\sum_{j\in P_i}u_{ij}
+
\frac{1}{n_i}
\sum_{j\notin P_i}u_{ij}.
```

The total weight of positive patches is:

```math
\frac{|P_i|}{n_i}.
```

If the positive morphology occupies one patch out of many, its contribution is
nearly invisible:

```math
\frac{|P_i|}{n_i}\approx0.
```

This is why mean pooling is weak for rare lesion detection.

## 2. Single-Patch Sensitivity Bound

If one patch changes by
```math
\Delta
```
:

```math
u_{ij}
\mapsto
u_{ij}+\Delta,
```

then the mean changes by:

```math
z_i
\mapsto
z_i+\frac{1}{n_i}\Delta.
```

So:

```math
\|\Delta z_i\|
=
\frac{1}{n_i}\|\Delta\|.
```

For very large bags, single-instance evidence is suppressed by design.

## 3. Multimodality Collision

Mean pooling can collapse distinct mixtures:

```math
\mu_i
=
\frac{1}{2}\delta_a
+
\frac{1}{2}\delta_b,
```

and:

```math
\mu_k
=
\delta_{(a+b)/2}.
```

Both have the same mean:

```math
\int u\,d\mu_i(u)
=
\int u\,d\mu_k(u)
=
\frac{a+b}{2}.
```

If the disease phenotype depends on coexistence of two morphologies rather than
their average, mean pooling erases the signal.

## 4. Nuisance Mass Dominance

Let a slide contain relevant tissue
```math
R_i
```
 and nuisance tissue
```math
N_i
```
:

```math
\{1,\ldots,n_i\}
=
R_i\cup N_i.
```

Mean pooling decomposes:

```math
z_i
=
\frac{|R_i|}{n_i}
\bar u_{R_i}
+
\frac{|N_i|}{n_i}
\bar u_{N_i}.
```

If nuisance tissue dominates area:

```math
\frac{|N_i|}{n_i}\gg\frac{|R_i|}{n_i},
```

the slide representation can become mostly a nuisance statistic.

Examples:

```text
background
normal tissue
stain intensity
tissue amount
scanner-specific texture
```

## 5. Geometry Loss

Two slides with the same patch multiset have the same mean:

```math
\{u_{i1},\ldots,u_{in}\}
=
\{u_{k1},\ldots,u_{kn}\}
\quad
\Longrightarrow
\quad
z_i=z_k.
```

Mean pooling cannot distinguish:

```text
clustered tumor from scattered tumor
invasive front from central tumor
organized glandular architecture from shuffled patches
```

unless geometry has already been encoded into the instance states.

## 6. Gradient Spreading

For a scalar loss
```math
\mathcal{L}
```
:

```math
z_i
=
\frac{1}{n_i}\sum_j u_{ij}.
```

The gradient to each instance is:

```math
\frac{\partial\mathcal{L}}{\partial u_{ij}}
=
\frac{1}{n_i}
\frac{\partial\mathcal{L}}{\partial z_i}.
```

Every instance receives the same global learning signal up to its downstream
Jacobian. When only a few patches cause the label, the supervision is spread
across many irrelevant patches.

## Diagnostic Questions

1. Is the target a prevalence, burden, rare event, or arrangement?
2. Does the mean use raw states or contextualized states?
3. Can two clinically distinct slides share the same first moment?
4. Is bag size meaningful or nuisance?
5. Does supervision need to localize sparse evidence?

## Dense Summary

The mean pooling assumption is:

```text
the first moment is enough
```

Failure occurs when:

```text
the label depends on extremes, multimodality, spatial arrangement, or rare evidence
```
