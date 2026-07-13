# Region Coarsening And Information Bottlenecks

## 1. Linear coarsening

Let H^{(\ell)} be child states at one level:

```math
H^{(\ell)}
\in
\mathbb R^{n_\ell\times d_\ell}.
```

A hard assignment matrix P maps children to parents. Sum coarsening is

```math
U_{\mathrm{sum}}
=
P^{\mathsf T}H^{(\ell)}
\in
\mathbb R^{n_{\ell+1}\times d_\ell}.
```

The b-th parent state is

```math
u_b
=
\sum_{a:\pi(a)=b}h_a.
```

Mean coarsening normalizes by parent cardinality:

```math
D=\mathrm{diag}(P^{\mathsf T}\mathbf 1),
\qquad
U_{\mathrm{mean}}
=
D^{-1}P^{\mathsf T}H^{(\ell)}.
```

The two operators encode different invariances. Sum retains cellularity or
patch count in the feature magnitude; mean is invariant to replication of
identical children inside every parent.

For a weighted assignment A with row sums one,

```math
U_A=A^{\mathsf T}H^{(\ell)}.
```

is a weighted additive coarsener. If columns of A do not sum to one, its scale
also depends on total routing mass.

## 2. Nullspace and indistinguishable configurations

The linear map H\mapsto P^{\mathsf T}H has a nontrivial nullspace whenever
n_\ell>n_{\ell+1}. Let Delta satisfy

```math
P^{\mathsf T}\Delta=0.
```

Then

```math
P^{\mathsf T}(H+\Delta)=P^{\mathsf T}H.
```

For each feature coordinate, the dimension of the nullspace is

```math
\dim\ker(P^{\mathsf T})
=
n_\ell-\mathrm{rank}(P).
```

With one nonempty disjoint parent for every column,
rank(P)=n_{\ell+1}, so the coarsener removes
n_\ell-n_{\ell+1} degrees of freedom per feature coordinate.

The discarded subspace includes within-parent contrasts. For a parent b with
children C_b, any perturbation satisfying

```math
\sum_{a\in C_b}\delta_a=0
```

is invisible to sum coarsening. A later parent-level transformer cannot reconstruct
which child carried the positive and negative deviation.

## 3. Attention coarsening

A parent-specific attention coarsener has

```math
\alpha_{ab}
=
\frac{
\exp q_{ab}
}{
\sum_{a':\pi(a')=b}\exp q_{a'b}
},
\qquad
u_b
=
\sum_{a:\pi(a)=b}\alpha_{ab}h_a.
```

The weights are normalized within each parent, so u_b is a convex combination
of child states. A gated score may be

```math
q_{ab}
=
w^{\mathsf T}
\left[
\tanh(Vh_a)
\odot
\sigma(Uh_a)
\right].
```

The weights depend on H, so attention has no fixed linear nullspace. It still
creates a learned selection bottleneck. If alpha concentrates on one child, the
parent state is effectively a top-instance statistic. If alpha is uniform, the
parent state approaches a mean.

A hierarchy of attention coarseners is composition, not one attention layer:

```math
H^{(L)}
=
\mathcal R_{L-1}\circ\cdots\circ\mathcal R_1
\left(H^{(0)}\right).
```

The Jacobian factorization is

```math
\frac{\partial H^{(L)}}{\partial H^{(0)}}
=
\prod_{\ell=0}^{L-1}
\frac{\partial \mathcal R_\ell}{\partial H^{(\ell)}}.
```

Small singular values at any level create a gradient and information bottleneck
for all descendants of that level.

## 4. Preservation of counts and locality

Let c_b=m_b be the child count. Sum pooling can carry c_b through feature
magnitude, while mean pooling cannot distinguish

```math
\{h,h\}
\quad\text{from}\quad
\{h\}
```

inside one parent. A count-augmented mean restores this particular statistic:

```math
\widetilde u_b
=
\left[
\frac{1}{m_b}\sum_{a\in C_b}h_a
\middle\Vert
\log(1+m_b)
\right].
```

This does not restore the lost arrangement of children. If spatial coordinates
are not passed to the parent, two different layouts with the same child
multiset and same count remain equivalent.

A region hierarchy preserves locality only relative to its partition. It does
not preserve exact geometry unless the coordinate graph, bounding boxes, or
relative positions are retained as inputs to later operators.

## 5. Complexity

Suppose n fine units are split into R regions of average size m=n/R. Full
within-region attention plus region-level attention has rough pairwise cost

```math
O(Rm^2+R^2).
```

This is less than flat O(n^2) only under a favorable regime such as R\ll n and
balanced regions. A pathological split can make the first term large, and
R close to n makes the second term approach flat attention. Hierarchy is an
algorithmic opportunity, not a complexity guarantee.

## 6. Design consequence

A coarsener should specify which of these invariants it intends:

`text
- replication invariance: mean or normalized attention;
- density sensitivity: sum or count-augmented mean;
- within-region arrangement: coordinates, local graph, or positional tokens;
- rare-child retention: top-k or sparse attention with an explicit recall cost;
- cross-region interaction: parent-level context after coarsening.
`

The phrase region pooling is incomplete until the statistic, normalization, and
information discarded by the region map are stated.
