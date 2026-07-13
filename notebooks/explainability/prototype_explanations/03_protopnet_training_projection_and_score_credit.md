# ProtoPNet Training, Projection, and Score Credit

## 1. Class-Conditional Geometry

Let `P_c` be the prototypes assigned to class `c`. ProtoPNet augments
cross-entropy with clustering and separation terms of the form

```math
\mathcal L
=\mathcal L_{\mathrm{CE}}
+\lambda_1\mathcal L_{\mathrm{clst}}
+\lambda_2\mathcal L_{\mathrm{sep}},
```

```math
\mathcal L_{\mathrm{clst}}
=\frac1N\sum_{i=1}^N
\min_{p_k\in P_{y_i}}
\min_{\widetilde z\in\mathcal P(f(x_i))}
\|\widetilde z-p_k\|_2^2,
```

```math
\mathcal L_{\mathrm{sep}}
=-\frac1N\sum_{i=1}^N
\min_{p_k\notin P_{y_i}}
\min_{\widetilde z\in\mathcal P(f(x_i))}
\|\widetilde z-p_k\|_2^2.
```

The first term is image-to-prototype coverage: each image need only approach
one same-class prototype. It does not force every prototype to be used. The
second separates each image from its nearest wrong-class prototype; it does not
guarantee global margin between all patch-prototype pairs.

## 2. Projection

The prototype push replaces a learned vector by a same-class training latent:

```math
p_k\leftarrow
\arg\min_{\widetilde z\in\mathcal Z_{c(k)}}
\|\widetilde z-p_k\|_2^2.
```

After projection, `p_k` has a concrete source patch. This improves
referential interpretability, but it changes the function. Prediction stability
therefore depends on the displacement

```math
\delta_k=\|p_k^{\mathrm{after}}-p_k^{\mathrm{before}}\|_2
```

being small relative to the preprojection classification margin. Projection is
not a proof that the prototype is morphologically pure.

## 3. Exact Score Credit

With a linear last layer,

```math
F_c(x)=b_c+\sum_{k=1}^K w_{ck}g_k(f(x)).
```

the exact additive prototype contribution is

```math
\Gamma_{kc}=w_{ck}g_k(f(x)).
```

Similarity `g_k` alone is unsigned and class-agnostic. A highly activated
prototype can decrease class `c`'s logit when its output weight is negative, or
be irrelevant when that weight is zero. For logit differences,

```math
F_c-F_{c'}=(b_c-b_{c'})+
\sum_k(w_{ck}-w_{c'k})g_k,
```

so the explanation must name the target score or contrast.
