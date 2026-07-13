# Dynamic Adjacency

A learned topology defines adjacency from the slide itself:

```math
A_{ijk}
=
g_\theta(h_{ij},h_{ik},c_{ij},c_{ik}).
```

Often this is normalized:

```math
\alpha_{ijk}
=
\mathrm{softmax}_{k}
g_\theta(h_{ij},h_{ik},c_{ij},c_{ik}).
```

Then context is:

```math
\widetilde h_{ij}
=
\sum_{k=1}^{n_i}
\alpha_{ijk}v_\theta(h_{ik}).
```

This looks like attention, but the geometry interpretation is:

```math
A_{\theta,i}
=
[\alpha_{ijk}]_{j,k}.
```

The model learns a directed weighted graph.

## Sparse Learned Graph

A sparse dynamic graph selects top neighbors:

```math
\mathcal{N}_{\theta}(j)
=
\mathrm{topK}_{k}
g_\theta(h_j,h_k,c_j,c_k).
```

Then:

```math
\widetilde h_j
=
\sum_{k\in\mathcal{N}_{\theta}(j)}
\alpha_{jk}v_\theta(h_k).
```

The topology changes with features and parameters.

## Static Versus Dynamic

Static graph:

```math
A_i
=
A(C_i)
```

Dynamic graph:

```math
A_{\theta,i}
=
A_\theta(H_i,C_i).
```

Task-conditioned dynamic graph:

```math
A_{\theta,i}^{(c)}
=
A_\theta(H_i,C_i,c)
```

for class or task index
```math
c
```
.

## C/R/G/S Placement

```text
G:
    learned adjacency A_theta

C:
    message passing or attention over learned edges

R:
    graph or set pooling after dynamic context

S:
    task supervision indirectly shapes the topology
```

## Dense Summary

Learned topology turns geometry into a hypothesis:

```math
\text{which patches should interact?}
```

The answer is learned from weak labels unless extra spatial supervision or
regularization is added.
