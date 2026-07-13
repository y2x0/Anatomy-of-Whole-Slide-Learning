# Failure Modes

Frozen-feature pipelines fail when the downstream target is not visible in the
frozen representation or when the downstream cohort changes the meaning of the
pretrained geometry.

## Representation Collapse

If:

```math
f_{\phi_0}(x)
=
f_{\phi_0}(x'),
```

but:

```math
Y(x)
\ne
Y(x'),
```

then no downstream deterministic head can separate
```math
x
```
 and
```math
x'
```
through frozen
features.

## Wrong Invariance

Pretraining may learn invariance:

```math
f_{\phi_0}(t(x))
\approx
f_{\phi_0}(x).
```

This is good only if:

```math
Y(t(x))
=
Y(x).
```

If the downstream task depends on the transformed factor, frozen features erase
signal.

## Cohort Shift

Let
```math
P_0(H,Y)
```
 be the pretraining/downstream source distribution and
```math
P_1(H,Y)
```
the target cohort. A frozen probe estimates:

```math
P_0(Y\mid H)
```

but deploys under:

```math
P_1(Y\mid H).
```

Even if
```math
H
```
is good, the conditional target can shift.

## Aggregation Mismatch

Patch features may be excellent while the slide readout is wrong:

```math
Y
=
\mathrm{upper\ tail}(H),
```

but the model uses:

```math
z
=
\frac{1}{n}\sum_j h_j.
```

This is not a foundation-model failure. It is an adaptation/readout failure.

## Dense Summary

Frozen features succeed when:

```math
P(Y\mid X)
=
P(Y\mid H)
```

and the downstream readout can express the needed slide statistic. They fail
when either the representation or the aggregation bottleneck discards the task
axis.
