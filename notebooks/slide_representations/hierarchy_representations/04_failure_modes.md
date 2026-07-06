# Hierarchy Representation Failure Modes

Hierarchy representations are powerful because they encode scale. They fail
when the chosen scale structure does not match the biology or task.

## 1. Boundary Error

A hard parent map:

```math
\pi(v)=u
```

forces fine unit $v$ to belong to one parent. If a biological structure crosses
the boundary between two regions, the hierarchy splits one object into separate
summaries.

The lost relation is:

```math
v\sim w
\quad
\text{but}
\quad
\pi(v)\ne\pi(w).
```

Lateral edges or overlapping regions can reduce this but increase complexity.

## 2. Premature Compression

If coarsening happens before the model has extracted the relevant fine-scale
signal, then evidence is averaged away:

```math
h_u^{(\ell+1)}
=
\frac{1}{|\mathrm{Ch}(u)|}
\sum_{v\in\mathrm{Ch}(u)}h_v^{(\ell)}.
```

If only one child is positive:

```math
h_{v^\star}^{(\ell)}
\quad
\text{with}
\quad
|\mathrm{Ch}(u)|\gg1,
```

then its contribution has weight:

```math
\frac{1}{|\mathrm{Ch}(u)|}.
```

This is sparse-positive dilution inside the hierarchy.

## 3. Scale Mismatch

Suppose the task-relevant pattern has physical scale $r^\star$. A hierarchy
offers scales:

```math
\Delta_0,\Delta_1,\ldots,\Delta_L.
```

If:

```math
r^\star\ll\Delta_\ell
```

then level $\ell$ is too coarse. If:

```math
r^\star\gg\Delta_\ell
```

then level $\ell$ is too local.

Scale mismatch means the model either fragments the signal or smears it.

## 4. Parent Shortcut

A model may learn a high-level region shortcut:

```math
h_u^{(\ell+1)}
\to
\widehat y_i
```

without using the fine children in a meaningful way.

Examples:

```text
tissue amount
scanner artifact
stain intensity at region scale
background or necrosis prevalence
```

The hierarchy can make these shortcuts easier because coarse tokens often carry
large nuisance factors.

## 5. Cross-Scale Attribution Ambiguity

If prediction uses:

```math
z_i
=
\psi(z_i^{(0)},z_i^{(1)},\ldots,z_i^{(L)}),
```

then a high score may arise from fine morphology, coarse architecture, or an
interaction between scales.

Attribution to a level is not automatic:

```math
\frac{\partial \widehat y_i}{\partial h_v^{(0)}}
```

and:

```math
\frac{\partial \widehat y_i}{\partial h_u^{(2)}}
```

answer different questions.

## Diagnostic Questions

1. What are the levels?
2. Are parent-child assignments hard or soft?
3. Does coarsening occur before or after local context?
4. Which physical scale matches the phenotype?
5. Does readout use only the top token or multiple scales?

## Dense Summary

The hierarchy assumption is:

```text
biology is compositional across chosen scales
```

Failure occurs when:

```text
the chosen regions, boundaries, or compression levels do not match the disease signal
```
