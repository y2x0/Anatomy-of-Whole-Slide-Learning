# Identifiability, Regularization, And Failure Modes

Learned topology is flexible, but weakly identifiable.

## Non-Identifiability

Suppose:

```math
z_i
=
\mathcal{R}
\left(
\mathcal{C}_{A_\theta}(H_i)
\right).
```

Many different adjacency functions can yield the same slide prediction:

```math
A_{\theta}
\ne
A_{\theta'}
\quad
\text{but}
\quad
f_{\theta}(H_i)
=
f_{\theta'}(H_i).
```

Thus a learned graph is not automatically an interpretable tissue graph.

## Shortcut Topology

With slide-level labels, topology can connect patches that share artifacts:

```math
A_{\theta}(j,k)
\uparrow
\quad
\text{because}
\quad
h_j,h_k
\text{ share scanner or stain signal}.
```

The graph may be predictive without being biological.

## Locality Regularization

A physical regularizer can penalize long edges:

```math
\mathcal{L}_{\mathrm{dist}}
=
\sum_{j,k}
A_{jk}\|c_j-c_k\|_2^2.
```

This encourages local topology.

## Entropy Regularization

Topology entropy is:

```math
H(A_j)
=
-\sum_k A_{jk}\log A_{jk}.
```

High entropy spreads attention. Low entropy selects sparse neighbors. The
correct direction depends on the intended geometry.

## Prior Graph Regularization

Given a physical prior graph $A^{0}$:

```math
\mathcal{L}_{\mathrm{prior}}
=
\|A_\theta-A^{0}\|_F^2.
```

This keeps learned topology near physical adjacency while allowing task-specific
deviations.

## Failure Modes

```text
non-identifiable edges:
    many graphs explain the same prediction

shortcut topology:
    edges connect artifacts or cohort signals

overlocal regularization:
    distant but meaningful morphology cannot interact

overglobal topology:
    physical tissue structure disappears

unstable neighborhoods:
    small feature changes create different edges
```

## Dense Summary

Learned topology must be evaluated as both:

```text
predictive mechanism
```

and:

```text
claimed geometry
```

Those are different standards.
