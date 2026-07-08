# Problem Setup

Attention is a learned weighting operator. It receives states, optional
geometry, and optional supervision, then returns a data-dependent linear
combination of values.

For a whole slide:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}\in\mathbb{R}^{d}.
```

The repository decomposition is:

```math
\widetilde H_i
=
\mathcal{C}(H_i;G_i,S_i),
\qquad
z_i
=
\mathcal{R}(\widetilde H_i;G_i,S_i),
\qquad
\widehat y_i
=
\mathcal{H}(z_i).
```

Attention can appear in either the context operator or the readout operator:

```text
context attention:
    H_i -> H_i'

readout attention:
    H_i -> z_i
```

The same formula can therefore mean different things.

## General Attention Operator

Let each target unit be indexed by `u` and each source unit by `v`.

```math
q_u
=
W_Q h_u,
\qquad
k_v
=
W_K h_v,
\qquad
r_v
=
W_V h_v.
```

A score is:

```math
s_{uv}
=
\psi_\theta(q_u,k_v,e_{uv}),
```

where `e_uv` may contain geometry, edge type, modality type, relative position,
or a mask.

The weights are:

```math
a_{uv}
=
\sigma_{v\in\mathcal{A}(u)}
\left(s_{uv}\right),
```

and the message is:

```math
m_u
=
\sum_{v\in\mathcal{A}(u)}a_{uv}r_v.
```

The neighborhood can be:

```text
all patches:
    transformer self-attention

one bag to one slide token:
    attention MIL or PMA readout

graph neighbors:
    graph attention

other modality:
    cross-attention and co-attention

memory bank:
    retrieval attention
```

## Attention As A Measure

For a fixed query `u`, the weights define a measure over source indices:

```math
\nu_{u,\theta}
=
\sum_{v\in\mathcal{A}(u)}
a_{uv}\delta_v.
```

The message is an expectation:

```math
m_u
=
\mathbb{E}_{v\sim\nu_{u,\theta}}[r_v].
```

This is the key simplification. Attention is not magic context. It is:

```text
score
normalization
weighted value expectation
```

## What Can Survive?

A single-head readout attention layer produces:

```math
z_i
=
\sum_{j=1}^{n_i}a_{ij}r_{ij}.
```

The surviving statistic is a weighted first moment in value space.

Multi-head context attention produces:

```math
h'_{iu}
=
\mathrm{concat}_{m=1}^{M}
\sum_{v}a^{(m)}_{iuv}r^{(m)}_{iv}.
```

The surviving statistic per node is a collection of head-specific weighted
moments.

## Attention Failure Modes

The main failure modes follow from the operator:

```text
attention collapse:
    weight mass concentrates on too few instances

attention dilution:
    informative rare patches receive small mass

wrong support:
    sparse or masked attention removes necessary patches

shortcut alignment:
    attention follows artifacts correlated with labels

normalization pathology:
    adding irrelevant patches changes all weights

explanation mismatch:
    high weight is not necessarily causal evidence
```

## Dense Summary

Attention answers:

```text
which source units should this query average?
```

The mathematical object is:

```math
\boxed{
\nu_{u,\theta}
=
\sum_v
\sigma_{u}
\left(
\psi_\theta(q_u,k_v,e_{uv})
\right)
\delta_v
}
```

Every attention method in this family is a different choice of score,
normalization, allowed source set, value map, and readout.
