# Class-Conditioned Readout

Single-branch attention learns one measure:

```math
\nu_i
=
\sum_j a_{ij}\delta_{h_{ij}}.
```

Class-specific attention learns one measure per class:

```math
\nu_i^{(c)}
=
\sum_j a_{ij}^{(c)}\delta_{h_{ij}}.
```

## Class-Specific Scores

For class $c$, define:

```math
s_{ij}^{(c)}
=
w_c^\top
\left(
\tanh(V_ch_{ij})
\odot
\mathrm{sigmoid}(U_ch_{ij})
\right).
```

Then:

```math
a_{ij}^{(c)}
=
\frac{\exp(s_{ij}^{(c)})}
{\sum_{\ell=1}^{n_i}\exp(s_{i\ell}^{(c)})}.
```

The class-specific slide embedding is:

```math
z_i^{(c)}
=
\sum_j a_{ij}^{(c)}h_{ij}.
```

The class logit can be:

```math
o_i^{(c)}
=
q_c^\top z_i^{(c)}.
```

## What Changes From ABMIL?

ABMIL uses:

```math
z_i
=
\sum_j a_{ij}h_{ij}
```

and then predicts all classes from the same slide vector. Class-specific
attention uses:

```math
z_i^{(1)},\ldots,z_i^{(C)}.
```

Thus each class can preserve a different weighted first moment.

## Multi-Branch Versus Shared-Branch

A shared attention branch computes:

```math
a_{ij}
\quad
\text{for all classes}.
```

This assumes the same tissue is relevant for every class.

A multi-branch class-specific attention module computes:

```math
a_{ij}^{(c)}
\quad
c=1,\ldots,C.
```

This allows:

```text
class 1:
    tumor gland morphology

class 2:
    necrosis

class 3:
    lymphocytic infiltrate
```

to be summarized by different tissue subsets.

## Statistic That Survives

For class $c$, the statistic is:

```math
z_i^{(c)}
=
\mathbb{E}_{j\sim a_i^{(c)}}[h_{ij}].
```

The model does not preserve the full collection of class-relevant patches. It
preserves one class-conditioned first moment.

## Decision Geometry

For a one-vs-rest class logit:

```math
o_i^{(c)}
=
q_c^\top
\sum_j a_{ij}^{(c)}h_{ij}
=
\sum_j a_{ij}^{(c)}q_c^\top h_{ij}.
```

The class head and the class attention head interact. A patch matters for class
$c$ only through both:

```text
attention:
    does the class readout look there?

value alignment:
    does the class head treat that patch as evidence?
```

## Dense Summary

Class-specific attention changes:

```text
one slide measure
```

into:

```text
one slide measure per class
```

The mathematical benefit is class-conditioned evidence selection. The cost is
more heads, more pseudo-explanation ambiguity, and more opportunities for
attention branches to specialize to shortcuts.

