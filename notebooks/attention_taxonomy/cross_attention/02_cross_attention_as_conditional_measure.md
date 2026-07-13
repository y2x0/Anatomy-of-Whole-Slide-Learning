# Cross-Attention As Conditional Measure

Cross-attention defines a conditional measure:

```math
\nu_\theta(Y\mid x_a)
=
\sum_{b=1}^{n}a_{ab}\delta_{y_b}.
```

The output is:

```math
m_a
=
\int v_\theta(y)\,d\nu_\theta(y\mid x_a).
```

This is a query-conditioned summary of the source object.

## Conditional Readout

For a single class query:

```math
q_c
\in
\mathbb{R}^{d_k},
```

and patch keys:

```math
k_j
=
K h_j,
```

the class-specific readout is:

```math
z^{(c)}
=
\sum_{j=1}^{n}
a_j^{(c)}W_V h_j,
```

where:

```math
a_j^{(c)}
=
\mathrm{softmax}_j
\left(
\frac{q_c^\top k_j}{\sqrt{d_k}}
\right).
```

This generalizes class-specific attention: the class index becomes a query.

## Multiple Query Readouts

With `M` learned queries:

```math
Q
=
\{q_m\}_{m=1}^{M},
```

the output is:

```math
Z
=
\{z_m\}_{m=1}^{M},
\qquad
z_m
=
\sum_{j}a_{mj}W_V h_j.
```

Instead of one slide vector, the model preserves `M` query-conditioned moments.

This is the readout logic behind seed-based set pooling and many learned-token
readouts.

## Query Sensitivity

For a single query, write:

```math
z(q)
=
\sum_j a_j(q)v_j,
\qquad
a_j(q)
=
\mathrm{softmax}_j
\left(
\frac{q^\top k_j}{\sqrt{d_k}}
\right).
```

Differentiating through the normalization gives:

```math
\frac{\partial z}{\partial q}
=
\frac{1}{\sqrt{d_k}}
\sum_j
a_j(q)
v_j
\left(
k_j
-
\mathbb{E}_{a(q)}[k]
\right)^\top.
```

Equivalently, the query Jacobian is a value-key cross-covariance under the
attention measure. If value directions are uncorrelated with key directions,
changing the query can change the weights without materially changing the
readout. Query diversity should therefore be assessed in both attention space
and value space.

## Alignment Versus Explanation

The cross-attention row:

```math
a_{a:}
```

answers:

```text
which source tokens helped construct query output a?
```

It does not directly answer:

```text
which source tokens caused the prediction?
```

The final task head may use only some query outputs, or may transform them
through nonlinear layers. Alignment is a computational relation, not a causal
claim.

## Dense Summary

Cross-attention changes the slide representation from:

```text
one unconditional slide statistic
```

to:

```text
one statistic per query
```

The query source is the inductive bias: class, prompt, pathway, modality, or
memory.
