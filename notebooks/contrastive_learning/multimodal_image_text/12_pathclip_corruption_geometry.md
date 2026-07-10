# PathCLIP Corruption Geometry

Primary anchor:

- Wang et al. "Benchmarking PathCLIP for Pathology Image Analysis."
  arXiv 2024. https://arxiv.org/abs/2401.02651

The mapped PathCLIP paper is an evaluation of pathology vision-language models
under corruption. It should not be presented as a distinct contrastive
objective derivation.

## Corruption Operators

Let `c_{a,s}` denote corruption family `a` at severity `s`. The benchmark
considers brightness, contrast, Gaussian blur, resolution, saturation, hue,
and markup across four severity levels:

```math
x^{(a,s)}
=
c_{a,s}(x).
```

For normalized image encoder `u` and text prototype `v_c`, the class logit is:

```math
\ell_c^{(a,s)}(x)
=
\gamma
u
\left(
c_{a,s}(x)
\right)^{\top}v_c.
```

## Angular Representation Drift

Define corruption drift:

```math
d_{a,s}(x)
=
1-
u(x)^{\top}
u
\left(
c_{a,s}(x)
\right).
```

Class-margin change between classes `y` and `k` is:

```math
\Delta m_{yk}^{(a,s)}(x)
=
\gamma
\left[
u
\left(
c_{a,s}(x)
\right)
-
u(x)
\right]^{\top}
\left(
v_y-v_k
\right).
```

Large representation drift does not necessarily cause an error; only the
component along a decision-normal direction changes the class margin.

## First-Order Sensitivity

For a small pixel perturbation `\delta`:

```math
u(x+\delta)
\approx
u(x)
+
J_u(x)\delta.
```

Then:

```math
\Delta m_{yk}(x)
\approx
\gamma
\delta^{\top}
J_u(x)^{\top}
\left(
v_y-v_k
\right).
```

Robustness depends jointly on the image Jacobian and the prompt-defined class
direction.

## Retrieval Stability

For query image `x` and database images `z_j`, retrieval scores are:

```math
r_j(x)
=
u(x)^{\top}u(z_j).
```

Let `j_1` and `j_2` be the top two clean candidates, with gap:

```math
g(x)
=
r_{j_1}(x)-r_{j_2}(x).
```

The top retrieval remains stable if score perturbations satisfy:

```math
\left|
\Delta r_{j_1}(x)
\right|
+
\left|
\Delta r_{j_2}(x)
\right|
<
g(x).
```

Classification and retrieval can therefore fail at different corruption
levels because they depend on different angular gaps.

## Resolution And Blur Are Information Loss

For degradation operator `D`, distinct native images can collapse:

```math
x_a\ne x_b,
\qquad
D(x_a)=D(x_b).
```

No deterministic downstream encoder can distinguish them:

```math
u(D(x_a))
=
u(D(x_b)).
```

This is stronger than ordinary nuisance sensitivity. It is non-injectivity in
the observation map.

## Markup As A Shortcut Intervention

Let image decompose into tissue and annotation channels:

```math
X
=
\left(
X_{\mathrm{tissue}},
X_{\mathrm{markup}}
\right).
```

Markup corruption changes the second channel while leaving the first largely
fixed. Performance change estimates dependence on a non-morphologic feature,
although it does not by itself identify the causal training mechanism.

## Interpretation Boundary

The benchmark measures robustness of the full map:

```math
\text{pixels}
\longrightarrow
\text{embedding}
\longrightarrow
\text{prototype or retrieval ranking}.
```

It does not isolate whether failure originates in pretraining data,
architecture, prompt choice, or classifier candidate set.
