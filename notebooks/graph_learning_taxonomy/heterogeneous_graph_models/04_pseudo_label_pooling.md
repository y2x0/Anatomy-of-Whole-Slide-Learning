# Pseudo-Label Pooling

Pseudo-label pooling is the readout operator in Chan et al.

After HEAT message passing:

```math
\widetilde H_i
=
\{\widetilde h_{iv}\}_{v\in V_i}.
```

Every node has a teacher-derived pseudo-label type:

```math
\tau_i(v)
\in
\mathcal{T}.
```

PL pooling groups nodes by these fixed pseudo-label names.

## Type-Specific Node Sets

For each node type:

```math
a\in\mathcal{T},
```

define:

```math
V_{ia}
=
\{v\in V_i:\tau_i(v)=a\}.
```

The feature matrix for nodes of type `a` is:

```math
\widetilde H_{ia}
=
\begin{bmatrix}
\widetilde h_{iv_1}\\
\vdots\\
\widetilde h_{iv_{n_{ia}}}
\end{bmatrix},
\qquad
v_j\in V_{ia}.
```

This object exists only when:

```math
V_{ia}
\neq
\varnothing.
```

## Empty-Type Convention

The paper's PL-Pool algorithm loops over the pseudo-label vocabulary:

```math
a\in\mathcal{T}.
```

But a slide can have no patches of a given type. Therefore a fixed-size
type-row matrix needs a missing-type convention.

Define an observed-type mask:

```math
m_{ia}
=
\mathbf{1}
\{V_{ia}\neq\varnothing\}.
```

For observed types:

```math
z_{ia}
=
\mathcal{R}_a
\left(
\{\widetilde h_{iv}:v\in V_{ia}\}
\right)
\in
\mathbb{R}^{d_{\mathrm{PL}}}.
```

For missing types, choose an implementation convention:

```math
z_{ia}
=
\eta_a
\quad
\text{when}
\quad
m_{ia}=0,
```

where `eta_a` can be a zero vector, learned missing-type vector, or masked row.
The paper does not make this edge case central, but the mathematics has to make
it explicit.

## PL-Pooled Type Matrix

After the missing-type convention is defined, stack the type rows:

```math
Z_i^{\mathrm{PL}}
=
\begin{bmatrix}
z_{ia_1}\\
\vdots\\
z_{ia_{|\mathcal{T}|}}
\end{bmatrix}
\in
\mathbb{R}^{|\mathcal{T}|\times d_{\mathrm{PL}}}.
```

The row index has a fixed teacher-vocabulary meaning:

```math
Z_i^{\mathrm{PL}}[a,:]
=
z_{ia}.
```

The paper's algorithm writes the observed-type row as:

```math
z_{ia}
=
\mathrm{readout}_a
\left(
\widetilde H_{ia}
\right).
```

## Graph-Level Readout

The PL-pool output is the type-row matrix:

```math
Z_i^{\mathrm{PL}}
\in
\mathbb{R}^{|\mathcal{T}|\times d_{\mathrm{PL}}}.
```

The final graph vector is determined by another readout:

```math
z_i
=
\mathcal{R}_{\mathrm{graph}}
\left(
Z_i^{\mathrm{PL}},
m_i
\right).
```

Mean readout over observed types is one possible convention:

```math
z_i
=
\frac{1}{|\mathcal{T}_{i}^{\mathrm{obs}}|}
\sum_{a\in\mathcal{T}_{i}^{\mathrm{obs}}}
z_{ia}.
```

where:

```math
\mathcal{T}_{i}^{\mathrm{obs}}
=
\{a\in\mathcal{T}:V_{ia}\neq\varnothing\}.
```

Then:

```math
\widehat y_i
=
\mathcal{H}_\theta(z_i).
```

## Why This Is Not Ordinary Clustering

Cluster-based graph pooling learns or infers clusters from node embeddings:

```math
c_v
=
g_\theta(\widetilde h_v).
```

Chan et al. instead use:

```math
c_v
=
\tau(v),
```

where `tau(v)` comes from a pretrained nucleus-type model.

The row labels therefore come from a fixed teacher vocabulary:

```text
neoplastic
inflammatory
connective
dead
non-neoplastic epithelial
no label
```

This is the paper's core pooling claim:

```text
pseudo-label clusters reduce the inconsistent-cluster problem of
embedding-similarity pooling
```

The claim is conditional on the teacher and majority-vote construction. The
rows are fixed by pseudo-label names; their biological correctness is only as
good as the upstream nucleus classifier and patch-level compression.

## Row-Semantics Proposition

Let:

```math
Z_i^{\mathrm{PL}}
\in
\mathbb{R}^{|\mathcal{T}|\times d_{\mathrm{PL}}}
```

be the PL-pooled matrix whose row indexed by type `a` is:

```math
Z_i^{\mathrm{PL}}[a,:]
=
z_{ia}.
```

For two slides `i` and `j`, the same row index refers to the same teacher
pseudo-label:

```math
Z_i^{\mathrm{PL}}[a,:]
\text{ and }
Z_j^{\mathrm{PL}}[a,:]
```

both summarize nodes assigned:

```math
\tau(v)=a.
```

Thus PL pooling removes the arbitrary row-permutation ambiguity of learned
cluster pooling, conditional on the fixed teacher vocabulary. A learned
clustering method can produce cluster `1` in slide `i` and cluster `1` in slide
`j` with different meanings. PL pooling fixes:

```math
\text{row }a
\equiv
\text{teacher pseudo-label }a.
```

This proposition does not say the pooled vector preserves the full within-type
distribution, and it does not prove that the teacher pseudo-label is always
biologically correct. It only says the rows of `Z_i^{\mathrm{PL}}` have
consistent teacher-vocabulary semantics across slides.

## What Survives Pooling

PL pooling preserves:

```text
one pooled vector per pseudo-label type
chosen type-wise statistic after HEAT context
teacher-vocabulary row identity
missing-type mask or convention
```

It discards:

```text
individual patch identities inside each type
within-type spatial arrangement
minority nuclei inside patches already lost before tau(v)
uncertainty in HoVer-Net pseudo-labels
```

Formally, after type pooling, the graph-level classifier sees:

```math
Z_i^{\mathrm{PL}}
=
\left[
z_{ia}
\right]_{a\in\mathcal{T}},
```

not:

```math
\{\widetilde h_{iv}:v\in V_i\}
```

directly.

## C/R/G/S Placement

```text
\mathcal{G}:
    tau(v) defines fixed teacher pseudo-label clusters

\mathcal{C}:
    HEAT has already contextualized nodes

\mathcal{R}:
    PL pooling maps nodes to type-wise pooled rows, then to a graph vector

\mathcal{S}:
    slide label trains the final classifier; pseudo-labels come from pretrained
    HoVer-Net, not from slide labels
```
