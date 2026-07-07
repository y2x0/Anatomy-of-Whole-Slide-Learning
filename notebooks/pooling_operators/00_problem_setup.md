# Problem Setup

Pooling is the compression step in whole-slide learning.

Start with a slide representation:

```math
\mathcal{X}_i.
```

A context operator may produce contextualized instance states:

```math
\widetilde H_i
=
\mathcal{C}_\theta(\mathcal{X}_i)
=
\{u_{ij}\}_{j=1}^{n_i},
\qquad
u_{ij}\in\mathbb{R}^{d}.
```

Pooling maps many states to one slide statistic:

```math
z_i
=
\mathcal{R}_\theta(\widetilde H_i).
```

The task head then predicts:

```math
\widehat y_i
=
\mathcal{H}_\theta(z_i).
```

The central question:

```text
Which facts about the instance collection survive in z_i?
```

## Pooling As A Functional

For a set of instance states, define the empirical measure:

```math
\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\delta_{u_{ij}}.
```

A permutation-invariant pooling operator is a functional:

```math
z_i
=
T_\theta(\mu_i).
```

Examples:

```math
T(\mu_i)
=
\int u\,d\mu_i(u)
```

for mean pooling, and:

```math
T(\mu_i)
=
\sup_{u\in\mathrm{supp}(\mu_i)}g_\theta(u)
```

for max-style pooling.

## Equivalence Induced By Pooling

Every pooling operator defines an equivalence relation:

```math
\widetilde H_i
\sim_{\mathcal{R}}
\widetilde H_k
\quad
\Longleftrightarrow
\quad
\mathcal{R}(\widetilde H_i)
=
\mathcal{R}(\widetilde H_k).
```

If two slides are equivalent under $\mathcal{R}$, the downstream head cannot
distinguish them:

```math
\mathcal{H}(\mathcal{R}(\widetilde H_i))
=
\mathcal{H}(\mathcal{R}(\widetilde H_k)).
```

This is why pooling is not a technical footnote. It decides which slides become
mathematically identical to the model.

## Pooling Versus Context

No context:

```math
u_{ij}=h_{ij}.
```

Pooling directly summarizes raw patch embeddings:

```math
z_i
=
\mathcal{R}(\{h_{ij}\}_{j=1}^{n_i}).
```

With context:

```math
u_{ij}
=
\mathcal{C}_\theta(H_i)_j.
```

Pooling summarizes contextualized states:

```math
z_i
=
\mathcal{R}(\{u_{ij}\}_{j=1}^{n_i}).
```

The same readout formula can preserve different information depending on what
the context operator embedded into $u_{ij}$.

## Three Kinds Of Surviving Statistic

Moment-like statistic:

```math
z_i
=
\frac{1}{n_i}
\sum_j\phi_\theta(u_{ij}).
```

This preserves an average transformed feature.

Extreme statistic:

```math
z_i
=
\max_j g_\theta(u_{ij}).
```

This preserves the strongest instance-level evidence.

Distribution statistic:

```math
z_i
=
\left(
\frac{1}{n_i}\sum_j q_1(u_{ij}),
\ldots,
\frac{1}{n_i}\sum_j q_M(u_{ij})
\right).
```

This preserves a morphology histogram or prototype prevalence vector.

Most pooling operators are combinations or relaxations of these.

## Normalization Matters

Mean pooling:

```math
z_i
=
\frac{1}{n_i}\sum_j u_{ij}
```

is size-normalized.

Sum pooling:

```math
z_i
=
\sum_j u_{ij}
```

preserves count or tissue burden.

These are not interchangeable. If disease evidence is proportional to total
amount of tissue, sum-like pooling may be natural. If disease evidence is a
prevalence or composition, mean-like pooling may be natural.

## Dense Summary

```math
\boxed{
\widetilde H_i
\xrightarrow{\mathcal{R}}
z_i
}
```

Pooling is a statistical bottleneck. To understand a whole-slide method, ask:

```text
1. What are the instance states before pooling?
2. Is pooling invariant, order-aware, graph-aware, or query-based?
3. What statistic survives?
4. Which different slides collide under that statistic?
5. Which failure mode follows from that collision?
```
