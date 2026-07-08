# Attention Versus Adjacency Learning

Graph attention and graph construction are different operations.

## Fixed Adjacency, Learned Weights

If the adjacency is fixed:

```math
M_{uv}
=
\mathbf{1}\{v\in\mathcal{N}(u)\},
```

then attention learns:

```math
a_{uv}
>
0
\quad
\text{only where }M_{uv}=1.
```

The model can reweight edges but cannot create new ones.

## Learned Adjacency

A learned topology defines:

```math
M_{uv,\theta}
=
\mathbf{1}
\left\{
g_\theta(h_u,h_v,c_u,c_v)>0
\right\}.
```

or a soft adjacency:

```math
\alpha_{uv}
=
\mathrm{sigmoid}
\left(
g_\theta(h_u,h_v,c_u,c_v)
\right).
```

Then attention may be:

```math
a_{uv}
\propto
\alpha_{uv}\exp(e_{uv}).
```

Now the model learns both:

```text
which edges exist
how much each existing edge matters
```

## Decomposition

A graph attention message can be factored:

```math
h_u'
=
\sum_v
\underbrace{M_{uv}}_{\text{support}}
\underbrace{\omega_{uv,\theta}}_{\text{edge weight}}
\underbrace{W h_v}_{\text{value}}.
```

The support and weight should not be conflated.

## WSI Consequence

If pathology interaction depends on spatial adjacency, a feature-space learned
graph can connect morphologically similar but physically distant regions.

If pathology interaction depends on phenotype similarity, a physical kNN graph
can miss long-range repeated patterns.

The right graph depends on the mechanism:

```text
local invasion:
    physical adjacency matters

tumor subtype distribution:
    feature similarity may matter

immune-tumor interface:
    local cross-phenotype edges matter
```

## Dense Summary

Graph attention answers:

```text
how should existing neighbors be weighted?
```

Adjacency learning answers:

```text
which neighbors should exist?
```

The first is an attention problem. The second is a geometry problem.
