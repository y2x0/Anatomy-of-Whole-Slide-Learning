# DSMIL Critical Instance and Bag Context

## 1. Instance Branch

DSMIL assigns an instance-level class score

```math
q_{ijc}=f_c(h_{ij}).
```

The critical instance for class `c` is

```math
j_{ic}^{\star}\in\arg\max_j q_{ijc}.
```

## 2. Bag Branch

The critical instance supplies a class query or reference that weights other
instances. Abstractly,

```math
b_{ijc}=\mathrm{sim}(h_{ij},h_{ij_{ic}^{\star}}),
\qquad
a_{ijc}=\mathrm{softmax}_j(b_{ijc}),
```

and

```math
z_{ic}=\sum_j a_{ijc}h_{ij}.
```

The final score combines instance and bag branches.

## 3. Surviving Statistic

DSMIL preserves both an extreme class-specific instance and a critical-instance-
conditioned weighted summary. It is not equivalent to ordinary attention over
an independently learned query.

## 4. Instability

If two instances nearly tie,

```math
q_{ijc}\approx q_{i\ell c},
```

the selected critical instance can switch discontinuously. The bag attention and
explanation can change even when the input changes infinitesimally.

