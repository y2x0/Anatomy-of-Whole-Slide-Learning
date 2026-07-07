# C/R/G/S Pooling Decomposition

Pooling is the $\mathcal{R}$ part of the full whole-slide map:

```math
H_i
\xrightarrow{\mathcal{C}}
\widetilde H_i
\xrightarrow{\mathcal{R}}
z_i
\xrightarrow{\mathcal{H}}
\widehat y_i.
```

The point of the pooling notes is to make $\mathcal{R}$ mathematically explicit.

## C: Context Operator

The context operator creates the states that pooling sees:

```math
\widetilde H_i
=
\mathcal{C}(H_i;G_i,S_i)
=
\{u_{ij}\}_{j=1}^{n_i}.
```

Examples:

```text
identity:
    u_ij = h_ij

graph:
    u_ij has neighbor information

transformer:
    u_ij has all-pairs attention information

state-space:
    u_ij has sequence-scan information

prototype assignment:
    u_ij may include q_ijm or responsibilities
```

The same readout can mean different things depending on $\mathcal{C}$.

## R: Readout Operator

The readout maps many states to one statistic:

```math
z_i
=
\mathcal{R}(\{u_{ij}\}_{j=1}^{n_i}).
```

Mean:

```math
z_i
=
\frac{1}{n_i}\sum_j u_{ij}.
```

Max:

```math
z_i
=
\max_j g_\theta(u_{ij}).
```

Attention:

```math
z_i
=
\sum_j a_{ij}v_\theta(u_{ij}).
```

Prototype:

```math
z_i
=
\left(
\frac{1}{n_i}\sum_j q_{ij1},
\ldots,
\frac{1}{n_i}\sum_j q_{ijM}
\right).
```

Each readout defines an equivalence relation: two slides collide when they have
the same $z_i$.

## G: Geometry

Geometry may be absent:

```math
G_i=\varnothing.
```

Then pooling is permutation invariant over instances.

Geometry may also be encoded before readout:

```math
u_{ij}
=
\mathcal{C}(h_{ij};G_i).
```

or inside the readout itself:

```math
q_{ijm}
\propto
\exp(-d_G(h_{ij},c_m)).
```

For prototype pooling, $G$ can be prototype geometry:

```math
C_{mn}
=
\|c_m-c_n\|^2.
```

For graph pooling, $G$ is adjacency. For hierarchy pooling, $G$ is parent-child
structure. For sequence pooling, $G$ is order.

## S: Supervision

Supervision shapes the pooling operator even when it does not appear at
inference.

Attention:

```math
S_i=y_i
\quad\Rightarrow\quad
s_\theta
\ \text{is trained by slide labels}.
```

CLAM:

```math
S_i
=
(y_i,T_i^{+},T_i^{-})
```

where $T_i^{+}$ and $T_i^{-}$ are pseudo-instance sets induced by attention.

PANTHER:

```math
S_i
=
\text{unsupervised prototype objective plus downstream labels}.
```

Survival pooling:

```math
S_i
=
(X_i,\delta_i)
```

and the same $z_i$ may be trained under Cox, hazard, ranking, or competing-risk
losses.

## Surviving Statistic

The surviving statistic is the information in $z_i$:

```text
mean:
    first moment

max:
    strongest instance evidence

attention:
    learned weighted first moment

class-specific attention:
    class-conditioned weighted first moments

prototype:
    morphology prevalence and residuals
```

The failure mode follows from the statistic that does not survive.

## Dense Summary

```text
C says what each instance knows.
R says what survives.
G says what structure can be used.
S says what pressure shaped the statistic.
```

Pooling is not just an implementation detail. It is the mathematical contract
between the slide representation and the task head.

