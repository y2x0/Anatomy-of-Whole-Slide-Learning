# Feature Space Versus Physical Space

Learned topology can connect patches by physical distance, feature similarity,
or a mixture of both.

## Physical Kernel

A physical adjacency score can be:

```math
g_{\mathrm{phys}}(j,k)
=
-\frac{\|c_j-c_k\|_2^2}{2\sigma^2}.
```

This favors nearby patches.

## Feature Kernel

A morphology adjacency score can be:

```math
g_{\mathrm{feat}}(j,k)
=
\frac{\phi(h_j)^\top\phi(h_k)}
{\|\phi(h_j)\|_2\|\phi(h_k)\|_2}.
```

This favors visually similar patches, regardless of distance.

## Hybrid Topology

A hybrid score is:

```math
g_\theta(j,k)
=
g_{\mathrm{feat}}(j,k)
+
\lambda g_{\mathrm{phys}}(j,k).
```

When $\lambda$ is large, learned topology is local. When $\lambda$ is small,
the graph can connect distant morphology.

## What Relation Is Being Learned?

Feature-space edges answer:

```text
which patches look related?
```

Physical-space edges answer:

```text
which patches are tissue neighbors?
```

Task-learned edges may answer:

```text
which patches jointly predict the label?
```

These are not the same relation.

## C/R/G/S Placement

```text
G:
    physical, feature, or hybrid relation

C:
    relation-weighted message passing

R:
    pooling after relation-induced context

S:
    labels may push topology toward predictive shortcuts
```

## Dense Summary

Learned topology should state its metric:

```math
\text{physical distance}
\ne
\text{morphology similarity}
\ne
\text{label-predictive relation}.
```

Confusing these is the main conceptual error.
