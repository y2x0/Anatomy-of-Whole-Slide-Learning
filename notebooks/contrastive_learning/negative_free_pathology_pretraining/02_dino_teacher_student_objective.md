# DINO Teacher-Student Objective

Primary anchor:

- Caron et al. "Emerging Properties in Self-Supervised Vision Transformers."
  ICCV 2021. https://arxiv.org/abs/2104.14294

## Student And Teacher Distributions

For network logits `g_{\theta_s}(x)` and student temperature `\tau_s`:

```math
P_s^{(k)}(x)
=
\frac{
\exp
\left(
g_{\theta_s}^{(k)}(x)/\tau_s
\right)
}{
\sum_{a=1}^{K}
\exp
\left(
g_{\theta_s}^{(a)}(x)/\tau_s
\right)
}.
```

With teacher center `c` and temperature `\tau_t`:

```math
P_t^{(k)}(x)
=
\frac{
\exp
\left(
\left[
g_{\theta_t}^{(k)}(x)-c_k
\right]/\tau_t
\right)
}{
\sum_{a=1}^{K}
\exp
\left(
\left[
g_{\theta_t}^{(a)}(x)-c_a
\right]/\tau_t
\right)
}.
```

The output coordinates are learned dimensions, not semantic class labels.

## Multi-Crop Relation

Let the view set contain two global crops and several local crops:

```math
\mathcal{V}(x)
=
\left\{
x_1^g,x_2^g,
x_1^{\ell},\ldots,x_m^{\ell}
\right\}.
```

The teacher sees only global crops:

```math
\mathcal{V}_t(x)
=
\left\{
x_1^g,x_2^g
\right\},
```

while the student sees every crop. The objective is:

```math
\mathcal{L}_{\mathrm{DINO}}(x)
=
\sum_{v_t\in\mathcal{V}_t(x)}
\sum_{
v_s\in\mathcal{V}(x):
v_s\ne v_t
}
H
\left(
P_t(v_t),
P_s(v_s)
\right).
```

This is global-to-global and global-to-local distillation. It is not symmetric
local-to-global teaching.

## Exact Logit Gradient

For one teacher-student view pair:

```math
\mathcal{L}
=
-\sum_k
P_t^{(k)}
\log P_s^{(k)}.
```

Because the teacher is stop-gradient:

```math
\frac{\partial\mathcal{L}}
{\partial g_{\theta_s}^{(k)}}
=
\frac{1}{\tau_s}
\left(
P_s^{(k)}-P_t^{(k)}
\right),
```

```math
\frac{\partial\mathcal{L}}
{\partial g_{\theta_t}^{(k)}}
=
0.
```

Teacher parameters move only through EMA.

## Cross-Entropy Decomposition

```math
H(P_t,P_s)
=
H(P_t)
+
\mathrm{KL}
\left(
P_t\|\|P_s
\right).
```

For fixed teacher outputs, optimizing the student minimizes KL divergence. The
teacher entropy itself is controlled indirectly by centering, sharpening, data
views, and teacher dynamics.

## Pathology Inductive Bias

When a local crop contains only a gland or cell cluster while the global crop
contains surrounding tissue, the objective asks:

```math
P_s
\left(
x^{\ell}
\right)
\approx
P_t
\left(
x^g
\right).
```

This transfers context-defined targets into local representations. It can
teach useful phenotype consistency, but can also suppress local distinctions
that vary within one global region.
