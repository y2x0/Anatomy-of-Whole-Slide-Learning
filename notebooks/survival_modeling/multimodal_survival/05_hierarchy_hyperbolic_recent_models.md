# Hierarchy, Hyperbolic Geometry, And Recent Models

Recent multimodal survival models increasingly treat pathology and genomics as
hierarchical objects.

Examples:

```text
cells -> patches -> regions -> slide
genes -> pathways -> programs -> phenotype
patches -> prototypes -> tissue states -> prognosis
```

## Hierarchical Fusion

Let pathology hierarchy be:

```math
H_i^{(0)}=\text{patch tokens},
\qquad
H_i^{(1)}=\text{region tokens},
\qquad
H_i^{(2)}=\text{slide token}.
```

Let omics hierarchy be:

```math
G_i^{(0)}=\text{genes},
\qquad
G_i^{(1)}=\text{pathways},
\qquad
G_i^{(2)}=\text{programs}.
```

Fusion can occur levelwise:

```math
Z_i^{(\ell)}
=
\Phi_{\ell}(H_i^{(\ell)},G_i^{(\ell)}).
```

Survival readout:

```math
z_i=\mathcal{R}(\{Z_i^{(\ell)}\}_{\ell}),
\qquad
\eta_i=f(z_i).
```

## Hyperbolic Representation

Hyperbolic space is useful for tree-like hierarchy because volume grows
exponentially with radius.

In the Poincare ball:

```math
\mathbb{B}^{d}
=
\{x\in\mathbb{R}^{d}:\|x\|<1\}.
```

The distance is:

```math
d_{\mathbb{B}}(x,y)
=
\operatorname{arcosh}
\left(
1+
2
\frac{\|x-y\|^2}
{(1-\|x\|^2)(1-\|y\|^2)}
\right).
```

A survival model may embed modalities:

```math
u_i^p\in\mathbb{B}^{d},
\qquad
u_i^g\in\mathbb{B}^{d}.
```

Fusion can use hyperbolic operations or map to tangent space:

```math
v_i^p=\log_{0}(u_i^p),
\qquad
v_i^g=\log_{0}(u_i^g).
```

Then:

```math
z_i=\Phi(v_i^p,v_i^g).
```

## H2-Surv Role

H2-Surv is mathematically important because it changes the geometry of the
representation, not just the fusion layer.

The claim is:

```text
survival-relevant pathology/omics structure is hierarchical and better modeled
in hyperbolic space than Euclidean space.
```

This belongs in the representation layer:

```math
\mathcal{X}_{\text{slide}}
\to
\mathbb{B}^{d}
\to
\text{survival head}.
```

## Temporal Order Contrastive Learning

A survival-aware contrastive objective can encode that longer-surviving patients
should be ordered differently from shorter-surviving patients.

For comparable pair $i,j$:

```math
X_i<X_j,\quad \delta_i=1.
```

A directional contrastive loss can enforce:

```math
r_i>r_j
```

or structure embeddings so risk direction aligns with time ordering.

## Dense Summary

```text
hierarchical multimodal survival:
    aligns pathology and omics at multiple biological scales

hyperbolic survival:
    changes representation geometry for hierarchy

survival contrastive learning:
    uses event ordering to shape representation before risk head
```

These methods should be evaluated by whether the geometry changes the survival
statistic, not just whether the architecture is novel.
