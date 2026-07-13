# Bag Object and Latent Instance Labels

## 1. Bag Representation

For slide or patient `i`, let

```math
B_i=\{x_{ij},c_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}=f_\phi(x_{ij}).
```

`x_ij` is a patch, `c_ij` optional coordinate or metadata, and `h_ij` the
instance representation. The bag label is observed while instance labels are
usually latent.

## 2. Latent Instance State

Introduce latent instance state `z_ij` and bag label `y_i`:

```math
z_{ij}\in\{0,1\},
\qquad
y_i\in\{0,1\}.
```

The learning problem observes `(B_i,y_i)` but not `z_ij`. A model chooses a
functional

```math
\widehat y_i
=F_\theta(B_i)
=\mathcal H_\theta
\left(\mathcal R_\theta(\mathcal C_\theta(H_i))\right).
```

## 3. Instance-to-Bag Semantics

Different MIL assumptions specify different maps:

```math
\begin{aligned}
\text{standard positive MIL:}&\quad
y_i=1\Longleftrightarrow\exists j:z_{ij}=1,\\
\text{collective MIL:}&\quad
y_i=\psi(z_{i1},\ldots,z_{in_i}),\\
\text{burden MIL:}&\quad
y_i\text{ depends on }\sum_jz_{ij},\\
\text{relational MIL:}&\quad
y_i\text{ depends on }\{(z_{ij},z_{i\ell})\}_{j,\ell}.
\end{aligned}
```

The architecture should match the intended latent semantics rather than
silently assuming existential positivity.

## 4. WSI Consequence

Slide labels such as diagnosis, molecular subtype, or survival outcome can be
caused by a sparse region, diffuse burden, spatial arrangement, or a mixture of
these. MIL is therefore an assumption about the label-generating functional,
not merely a computational trick for large images.

