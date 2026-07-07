# Set, Distribution, And Graph Equivalence

Set, distribution, and graph representations are not always different raw data.
Often they start from the same patch embeddings and differ by what structure the
model is allowed to use.

## Same Carrier, Different Structure

A tiled slide gives patch embeddings:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i}.
```

The empirical measure is:

```math
\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\delta_{h_{ij}}.
```

As carriers of unordered patch information, $H_i$ with multiplicity and $\mu_i$
are equivalent. Each patch appears once in the multiset and once as an atom in
the empirical measure.

The graph object adds relation data:

```math
\mathcal{G}_i
=
(V_i,E_i,H_i),
\qquad
V_i=\{1,\ldots,n_i\}.
```

The edge set $E_i$ is not recoverable from $\mu_i$ unless the embedding already
encodes the relevant geometry. This is the first real break:

```text
set/distribution:
    what patches exist

graph:
    which patches can interact as neighbors
```

## Hypothesis Classes

A set model learns a permutation-invariant function:

```math
f(H_i)
=
f(\{h_{i\pi(j)}\}_{j=1}^{n_i})
```

for any permutation $\pi$.

A distribution model learns a functional:

```math
f(H_i)
=
F(\mu_i)
=
\psi(T(\mu_i)).
```

If $T$ is:

```math
T(\mu_i)
=
\int \phi(h)\,d\mu_i(h),
```

then this is a Deep Sets-style statistic:

```math
T(\mu_i)
=
\frac{1}{n_i}\sum_j\phi(h_{ij}).
```

A graph model learns an isomorphism-invariant function of both states and edges:

```math
f(H_i,E_i)
=
f(PH_i,PE_iP^\top),
```

where $P$ permutes nodes and edges together.

Thus the symmetry gets weaker as more structure is admitted:

```text
distribution statistic:
    invariant to every rearrangement preserving mu

set attention:
    invariant after equivariant interactions among all patches

graph:
    invariant only to relabelings that preserve adjacency
```

## When They Collapse

Graph collapses to set when the context operator ignores $E_i$:

```math
\mathcal{C}(H_i,E_i)=\mathcal{C}(H_i).
```

Complete-graph attention without coordinates is also set-like:

```math
E_i=V_i\times V_i.
```

Every node can see every other node, and the only remaining structure is learned
from the patch features themselves.

Distribution collapses to mean MIL when:

```math
T(\mu_i)
=
\int h\,d\mu_i(h).
```

Distribution becomes richer than mean MIL when $T$ includes higher moments,
prototypes, mixture responsibilities, kernel features, or transport distances.
But it is still blind to layout unless geometry enters $h$, $T$, or the metric.

## Toy Counterexample: Same Measure, Different Layout

Let a slide contain two patch types:

```math
a:
    \text{tumor-like},
\qquad
b:
    \text{stroma-like}.
```

Slide 1 places all tumor patches in one connected cluster:

```text
aaaabbbb
aaaabbbb
bbbbbbbb
bbbbbbbb
```

Slide 2 has the same number of each patch type but scatters tumor patches:

```text
abababab
babababa
abababab
babababa
```

If patch embeddings are exactly $a$ or $b$, both slides have the same empirical
measure:

```math
\mu_1=\mu_2.
```

Any statistic $T(\mu)$ gives the same result for both slides. A graph or
hierarchy can distinguish them only if spatial adjacency or region membership is
included before readout.

## Toy Counterexample: Wrong Graph

Suppose the true biological signal is local gland formation, but edges are built
by embedding kNN. If stain intensity dominates the embedding, kNN may connect
patches with similar color across distant tissue compartments:

```math
E_i
=
\mathrm{kNN}(h_{ij})
```

instead of:

```math
E_i
=
\mathrm{kNN}(c_{ij}).
```

Message passing then smooths across biologically unrelated tissue. The graph is
more structured than a set, but the wrong structure can be worse than no
structure.

## Toy Counterexample: Wrong Sequence

A state-space or recurrent model sees an ordered list:

```math
(h_{i\sigma(1)},\ldots,h_{i\sigma(n_i)}).
```

If $\sigma$ follows a raster scan, nearby sequence positions may be nearby on the
slide. If $\sigma$ is random, the same model receives an arbitrary trajectory.
The bag is unchanged, but the sequence representation changes the inductive
bias.

## Positive-Instance MIL In This Language

Classic MIL assumes latent instance labels:

```math
y_{ij}\in\{0,1\}
```

and a bag label:

```math
y_i
=
\max_j y_{ij}.
```

This is neither a graph nor a distribution assumption. It is a label-generation
assumption:

```text
one positive patch is sufficient to make the slide positive
```

Mean, distribution, graph, and hierarchy models can all be wrong if this logical
assumption is the real task and their readout dilutes the rare positive patch.

## Dense Summary

```text
set and empirical distribution:
    same unordered carrier, different functional language

distribution statistic:
    what finite summary of mu survives

graph:
    carrier plus adjacency; can distinguish layouts that share mu

sequence:
    carrier plus order; can model long trajectories but depends on sigma

hierarchy:
    carrier plus parent map; can preserve scale but can compress too early
```

The question is not only "what is the slide?" It is:

```text
which equivalences are we forcing before the task has a chance to look?
```
