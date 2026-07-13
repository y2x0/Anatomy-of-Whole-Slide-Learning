# Graph Hierarchy Attention And Multiscale Context

## 1. A three-stage composition

Let H_i^{(0)} be fine cell or patch states. A hierarchical graph hybrid can be
written

```math
H_i^{(1)}
=
\mathcal R_0
\left(
\mathcal C_0(H_i^{(0)},A_i^{(0)}),
P_i^{(0)}
\right),
```

```math
H_i^{(2)}
=
\mathcal C_1
\left(
H_i^{(1)},A_i^{(1)}
\right),
\qquad
z_i
=
\mathcal R_{\mathrm{att}}
\left(
H_i^{(2)}
\right).
```

The first context is fine-scale graph message passing, R_0 is a child-to-parent
assignment, the second context is coarse graph message passing, and the final
readout is attention.

This is the structural pattern behind a graph hierarchy with attention readout,
even if a particular paper uses sum, max, or layerwise concatenation instead.

## 2. HACT boundary

HACT supplies a cell graph, tissue graph, and assignment matrix B_i. The
fine-to-coarse transfer is

```math
U_i
=
B_i^{\mathsf T}
\mathcal C_{\mathrm{cell}}
\left(
H_i^{\mathrm{cell}},A_i^{\mathrm{cell}}
\right).
```

The tissue graph then produces

```math
\widetilde H_i^{\mathrm{tissue}}
=
\mathcal C_{\mathrm{tissue}}
\left(
\left[
H_i^{\mathrm{tissue}}\middle\Vert U_i
\right],
A_i^{\mathrm{tissue}}
\right).
```

A sum readout is

```math
z_i
=
\sum_b\widetilde h_{ib}^{\mathrm{tissue}}.
```

Replacing the sum with attention yields

```math
\alpha_{ib}
=
\frac{\exp q(\widetilde h_{ib}^{\mathrm{tissue}})}
{\sum_c\exp q(\widetilde h_{ic}^{\mathrm{tissue}})},
\qquad
z_i
=
\sum_b\alpha_{ib}
\widetilde h_{ib}^{\mathrm{tissue}}.
```

The latter is a valid hybrid design, but it is not the same surviving statistic
as the sum readout described in the HACT formulation.

## 3. Heterogeneous graph context

If nodes have types t in a set T, a type-conditioned graph operator may be

```math
\widetilde h_a
=
\phi_{t_a}(h_a)
+
\sum_{b\in\mathcal N(a)}
\psi_{t_a,t_b}
\left(
h_a,h_b,e_{ab}
\right).
```

The edge type and node type are part of G. A later attention score

```math
q_a
=
w^{\mathsf T}
\left[
\tanh(V\widetilde h_a)
\odot
\sigma(U\widetilde h_a)
\right]
```

weights heterogeneous contextual states. It does not erase the type-dependent
message passing that created them.

## 4. Dynamic graph plus multiscale readout

With feature-derived support, the graph itself depends on H:

```math
A(H)_{jk}
=
\mathbf 1
\left[
k\in\mathcal N_K(j;H)
\right].
```

The forward map becomes

```math
z
=
\mathcal R
\left(
\mathcal C(H,A(H))
\right).
```

If coarsening occurs before graph construction, the support is A(P^{\mathsf T}H).
If graph construction occurs before coarsening, it is A(H). These are generally
different graphs and therefore different hypothesis classes.

## 5. Surviving multiscale credit

With local attention alpha and coarse attention beta, the effective weight of
fine state a is

```math
\omega_a
=
\beta_{\pi(a)}
\alpha_{a,\pi(a)}
```

only when both context operators are absent or treated as fixed linear maps. With
graph or transformer context, the total derivative also contains neighborhood
paths:

```math
\frac{\partial z}{\partial h_a}
=
\sum_b
\frac{\partial z}{\partial \widetilde h_b}
\frac{\partial \widetilde h_b}{\partial h_a}.
```

The product of attention weights is therefore a routing heuristic, not a general
attribution theorem.

## 6. C/R/G/S placement

`text
C: fine graph message passing, coarse graph message passing, and optional
   type- or feature-conditioned attention.

R: assignment transfer, then sum, max, attention, or layerwise multiscale readout.

G: fine graph, coarse graph, parent map, node types, and dynamic supports.

S: slide labels, cell or tissue pseudo-types, and any auxiliary graph losses.

This hybrid retains a coarse statistic of relationally contextualized fine states.
`
