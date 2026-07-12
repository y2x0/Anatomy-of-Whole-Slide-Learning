# LRP Conservation And Propagation Rules

Primary anchor:

- Schramowski et al. "xMIL: Insightful Explanations for Multiple Instance
  Learning in Histopathology." 2024.
  https://arxiv.org/abs/2406.04280

## Relevance Messages

Starting from selected output relevance, layer-wise relevance propagation sends
relevance from layer `l+1` to layer `l`:

```math
r_i^{(l)}
=
\sum_j
\frac{
q_{ij}
}{
\sum_{i'}q_{i'j}
}
r_j^{(l+1)}.
```

The layer-specific quantity `q_ij` defines how neuron `i` contributes to neuron
`j`.

## Local Conservation

If denominators are nonzero:

```math
\sum_i
r_i^{(l)}
=
\sum_j
r_j^{(l+1)}.
```

Proof:

```math
\sum_i
r_i^{(l)}
=
\sum_j
\left[
\frac{
\sum_iq_{ij}
}{
\sum_{i'}q_{i'j}
}
\right]
r_j^{(l+1)}
=
\sum_j
r_j^{(l+1)}.
```

Repeated conservation decomposes selected output score at the input level.

## Epsilon Stabilization

For linear preactivation:

```math
z_j
=
\sum_i
x_iw_{ij}
+
b_j,
```

an epsilon-style rule uses a stabilized denominator:

```math
r_i
=
\sum_j
\frac{
x_iw_{ij}
}{
z_j
+
\varepsilon
\mathrm{sign}(z_j)
}
r_j.
```

Stabilization improves numerical behavior but can break exact conservation by
absorbing relevance into the stabilizer.

## Rule Dependence

LRP is a family of propagation rules, not one unique attribution functional.
Linear layers, ReLU layers, layer normalization, and attention require
different choices. Two rule sets can conserve the same output while allocating
credit differently.

## Signed Relevance

```math
r_i>0
```

supports the selected output score, while:

```math
r_i<0
```

opposes it. Conservation permits positive and negative terms to cancel.

## Implementation Invariance Boundary

Because propagation rules depend on network decomposition, functionally
equivalent implementations can receive different relevance allocations. LRP's
conservation does not imply implementation invariance.

## Output Target

If initial relevance is class logit:

```math
r^{(L)}
=
F_c(X),
```

the explanation decomposes that logit. Initializing probability, margin, or
survival risk defines a different conserved target.
