# Delete, Replace, Insert, And Recompute

## Deletion

For subset `S`:

```math
X_{-S}
=
X\setminus S.
```

Deletion changes bag cardinality:

```math
|X_{-S}|
=
|X|-|S|.
```

Any cardinality-sensitive model receives two simultaneous changes: tissue
content and set size.

## Zero Replacement

For embeddings `H` and mask `m_j`:

```math
\widetilde h_j
=
m_jh_j
+
\left(
1-m_j
\right)0.
```

The slot remains in normalization and context operations. Zero replacement is
not deletion unless the architecture explicitly masks the slot.

## Reference Replacement

```math
\widetilde h_j
=
m_jh_j
+
\left(
1-m_j
\right)b_j.
```

Choices include mean tissue, benign tissue, conditional imputation, or a mask
token. Each defines a different contrast.

## Insertion

For donor patch set `A`:

```math
X_{+A}
=
X\cup A.
```

Insertion can test whether a feature set is sufficient to increase a model
score, but donor context, multiplicity, and compatibility matter.

## Recompute Versus Freeze

Full recomputation evaluates:

```math
F
\left(
\mathcal{T}_S(X)
\right)
```

through every model layer. Frozen-context approximation changes only a later
readout while retaining contextual embeddings from the original bag.

For context encoder:

```math
H'
=
\mathcal{C}(H),
```

generally:

```math
\mathcal{C}
\left(
H_{-S}
\right)
\ne
\left[
\mathcal{C}(H)
\right]_{-S}.
```

## Attention Recomposition

Deleting one instance with original attention `alpha_i` renormalizes all
remaining weights:

```math
\alpha_j^{(-i)}
=
\frac{\alpha_j}{1-\alpha_i}.
```

The observed effect contains removed value and redistributed readout mass.

## Spatial Coherence

Deleting isolated patches creates a different intervention from deleting one
contiguous region:

```math
S_{\mathrm{scattered}}
\ne
S_{\mathrm{region}}
```

even at equal size. Histologic concepts often occupy structured regions, so
subset geometry belongs in the intervention definition.
