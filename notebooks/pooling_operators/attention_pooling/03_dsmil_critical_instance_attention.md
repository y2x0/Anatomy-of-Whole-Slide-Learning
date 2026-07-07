# DSMIL Critical-Instance Attention

DSMIL combines two ideas:

```text
1. an instance branch that finds high-scoring critical instances
2. a bag branch that attends to the bag using the critical instance as reference
```

This is different from ABMIL. ABMIL learns attention scores directly. DSMIL
first creates an instance classifier and then uses the most suspicious instance
to build a bag representation.

## Instance Branch

Let an instance classifier produce class scores:

```math
s_{ijc}
=
w_c^\top h_{ij}.
```

The max instance score for class $c$ is:

```math
m_{ic}
=
\max_j s_{ijc}.
```

The critical instance index is:

```math
j_i^\star(c)
=
\arg\max_j s_{ijc}.
```

This branch preserves an extreme statistic:

```math
m_{ic}
=
\max_j w_c^\top h_{ij}.
```

So DSMIL retains the sparse-positive MIL bias: one critical instance can drive a
class score.

## Bag Branch

Let query and value maps be:

```math
q_{ij}
=
Qh_{ij},
\qquad
v_{ij}
=
Vh_{ij}.
```

For class $c$, the critical query is:

```math
q_i^\star(c)
=
q_{i,j_i^\star(c)}.
```

DSMIL-style critical-instance attention can be written:

```math
a_{ij}^{(c)}
=
\frac{
\exp
\left(
\frac{
q_i^\star(c)^\top q_{ij}
}{\sqrt{d_q}}
\right)
}{
\sum_{\ell}
\exp
\left(
\frac{
q_i^\star(c)^\top q_{i\ell}
}{\sqrt{d_q}}
\right)
}.
```

The class-conditioned bag embedding is:

```math
z_i^{(c)}
=
\sum_j a_{ij}^{(c)}v_{ij}.
```

The bag logit is then computed from $z_i^{(c)}$.

## What Statistic Survives?

DSMIL preserves two coupled statistics:

```text
max branch:
    strongest instance score per class

bag branch:
    value average weighted by similarity to the critical instance
```

Mathematically:

```math
z_i^{(c)}
=
\mathbb{E}_{j\sim a_i^{(c)}}[v_{ij}],
\qquad
a_i^{(c)}
\propto
\exp
\left(
q_i^\star(c)^\top q_{ij}
\right).
```

The attention measure is not learned only from independent patch scores. It is
anchored to the selected critical instance.

## Critical Instance As A Query

ABMIL asks:

```text
which patches have high learned attention score?
```

DSMIL asks:

```text
which patches look similar to the most class-suspicious patch?
```

This changes the inductive bias. If disease appears as a coherent morphology,
critical-instance attention can collect related patches. If the max instance is
a false positive, the bag branch can amplify the wrong morphology.

## Coupling To Max Pooling

Because $j_i^\star(c)$ comes from an argmax, the bag branch is discontinuously
coupled to the instance branch. A small change in instance scores can switch:

```math
j_i^\star(c)
\to
\widetilde j_i^\star(c),
```

which changes the query and therefore all attention weights.

This makes DSMIL more structured than simple max pooling but still vulnerable to
critical-instance instability.

## Dense Summary

DSMIL is not merely attention pooling. It is:

```math
H_i
\xrightarrow{\text{instance classifier}}
j_i^\star(c)
\xrightarrow{\text{critical query}}
a_i^{(c)}
\xrightarrow{\text{weighted value readout}}
z_i^{(c)}.
```

The surviving statistic is a critical-instance-centered weighted first moment,
plus an explicit max score. That is why DSMIL sits between max MIL and attention
MIL.

