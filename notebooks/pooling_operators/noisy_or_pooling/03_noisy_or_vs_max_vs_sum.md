# Noisy-Or Versus Max Versus Sum

Noisy-or sits between max pooling and additive evidence pooling.

Let:

```math
p_{ij}\in[0,1].
```

Max probability pooling is:

```math
r_i^{\max}
=
\max_j p_{ij}.
```

Sum probability pooling is:

```math
r_i^{\mathrm{sum}}
=
\sum_j p_{ij}.
```

Noisy-or is:

```math
r_i^{\mathrm{or}}
=
1-\prod_j(1-p_{ij}).
```

## Inequalities

Noisy-or is at least the maximum:

```math
r_i^{\mathrm{or}}
\ge
r_i^{\max}.
```

By the union bound:

```math
r_i^{\mathrm{or}}
\le
\sum_j p_{ij}.
```

Thus:

```math
\max_j p_{ij}
\le
1-\prod_j(1-p_{ij})
\le
\sum_j p_{ij}.
```

## Interpretation

```text
max:
    one instance is enough, weak positives do not accumulate

sum:
    all probabilities add without saturation

noisy-or:
    weak probabilities accumulate but total probability saturates at 1
```

## Many Weak Positives

If:

```math
p_{ij}=p
```

for all $j$, then:

```math
r_i^{\mathrm{or}}
=
1-(1-p)^{n_i}.
```

For small $p$:

```math
1-(1-p)^{n_i}
\approx
n_i p.
```

For large $n_i p$, it saturates:

```math
r_i^{\mathrm{or}}
\to
1.
```

## C/R/G/S Placement

```text
G:
    none unless instance probabilities use context

C:
    probability model p_ij

R:
    max, sum, or noisy-or probability aggregation

S:
    binary bag label and assumed label-generation logic
```

## Dense Summary

Noisy-or is the right mathematical readout when:

```text
each instance can independently trigger positivity,
and the bag probability is the union probability.
```

It is the wrong readout when instances are strongly correlated or when the task
depends on prevalence rather than at-least-one evidence.

