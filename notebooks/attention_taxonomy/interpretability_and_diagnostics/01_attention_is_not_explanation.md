# Attention Is Not Explanation

Attention weights describe a computation:

```math
z
=
\sum_j a_jv_j.
```

The weights say how the value vectors were averaged. They do not, by
themselves, say which patch caused the prediction.

## Contribution To A Linear Head

If the task logit is:

```math
o
=
w^\top z,
```

then:

```math
o
=
\sum_j a_j w^\top v_j.
```

The patch contribution to the logit is:

```math
c_j
=
a_j w^\top v_j.
```

This is a decomposition of the observed logit, not a deletion effect. If patch
`j` is removed and the remaining attention weights are renormalized, the
model-level deletion effect is:

```math
\Delta_j
=
g
\left(
\sum_{k=1}^{n}a_kv_k
\right)
-
g
\left(
\sum_{k\ne j}\widetilde a_{k\mid-j}v_k
\right),
```

where:

```math
\widetilde a_{k\mid-j}
=
\frac{\exp(s_k)}
{\sum_{r\ne j}\exp(s_r)}.
```

Even for a linear head, `Delta_j` includes the change in every remaining
weight. For context attention or nonlinear encoders, removing one patch can
also change downstream token states. A contribution decomposition and a
counterfactual deletion effect answer different questions.

High attention with negative evidence can reduce a class logit:

```math
a_j\text{ high},
\qquad
w^\top v_j<0.
```

Low attention with strong evidence can still matter if its value alignment is
large.

## Nonlinear Heads

If:

```math
o
=
g(z),
```

then local sensitivity with respect to the value vector, holding the attention
weights fixed, is:

```math
\frac{\partial o}{\partial v_j}
=
a_j
\nabla_z g(z).
```

The attention weight is only one factor in the derivative.

For original patch embeddings, this fixed-attention derivative is incomplete.
If both values and scores depend on `h_j`, then:

```math
\frac{\partial o}{\partial h_j}
=
\left(
\frac{\partial v_j}{\partial h_j}
\right)^\top
\frac{\partial o}{\partial v_j}
+
\sum_k
\left(
\frac{\partial s_k}{\partial h_j}
\right)^\top
\frac{\partial o}{\partial s_k}.
```

The first term is the value path. The second term is the attention-score path.
In self-attention, query and key paths add further dependencies across rows.

## Context Attention

For context attention:

```math
h_u'
=
\sum_v a_{uv}v_v,
```

the final prediction may be many layers away:

```math
\widehat y
=
\mathcal{H}
\left(
\mathcal{R}
\left(
\mathcal{C}^{(L)}(H)
\right)
\right).
```

An early-layer high attention edge may have little effect on the final output.

## Attention As Explanation Requires Conditions

Attention is closer to explanation when:

```text
there is one attention readout
the value vectors or their derivative paths are accounted for
the head is linear
the attention map is stable
counterfactual removal changes the prediction
```

Most WSI architectures violate at least one of these.

## Dense Summary

Attention weight is:

```text
representation construction mass
```

Evidence is closer to:

```math
a_j
\times
\text{value alignment}
\times
\text{head sensitivity}.
```

The distinction matters because attention maps can be visually persuasive even
when they do not identify pathology evidence.
