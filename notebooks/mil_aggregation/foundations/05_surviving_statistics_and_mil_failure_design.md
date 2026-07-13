# Surviving Statistics and MIL Failure Design

## 1. Aggregation as Information Loss

Every readout maps a variable-size bag to a fixed statistic:

```math
z_i=\mathcal R(\widetilde H_i).
```

Two bags collide when

```math
\mathcal R(\widetilde H_i)
=\mathcal R(\widetilde H_{i'}).
```

The task head cannot distinguish a collision.

## 2. Typical Survivors

```text
mean: first moment;
max: extreme statistic;
attention: weighted first moment;
sum: burden or additive evidence;
prototype: similarity or component statistics;
distribution: moments, quantiles, or transport features;
transformer/graph: contextual states followed by a readout;
state-space: order-dependent compressed recurrence.
```

## 3. Failure-Mode Design

```math
\text{inductive bias}
\to
\text{surviving statistic}
\to
\text{failure mode}.
```

Examples:

```text
sparse-positive max -> artifact extreme;
mean pooling -> rare-signal dilution;
attention -> collapse or shortcut routing;
graph -> wrong topology or oversmoothing;
sequence -> ordering artifact or long-range loss;
hierarchy -> assignment bottleneck;
distribution -> sample-size variance and layout blindness.
```

## 4. New-Method Test

A proposed MIL method should specify which collision class it separates. If it
does not preserve a statistic absent from an existing operator, its novelty may
be a parameterization or optimization change rather than a new aggregation
archetype.

