# Concept Objects and Explanation Targets

## 1. Four Mathematical Objects

Let layer `l` map an input to an activation vector:

```math
\phi_l(x)\in\mathbb R^{d_l}.
```

A concept may be represented as:

```math
\begin{aligned}
\text{separating direction:}&\quad v_C^{(l)}\in\mathbb R^{d_l},\\
\text{cluster:}&\quad \mathcal K_C\subset\mathbb R^{d_l},\\
\text{sparse coordinate:}&\quad h_C(x)\ge0,\\
\text{text-derived direction:}&\quad t_C\in\mathbb R^d.
\end{aligned}
```

The first is discriminative, the second distributional, the third
reconstructive, and the fourth cross-modal. Their semantic interpretation does
not follow from a common theorem.

## 2. Five Explanation Targets

For class score `F_c`, concept `C`, and input `x`, distinguish:

```math
\begin{aligned}
\text{presence:}&\quad P_C(x),\\
\text{sensitivity:}&\quad
D_C F_c(x),\\
\text{population sign frequency:}&\quad
\mathbb P[D_C F_c(X)>0\mid Y=c],\\
\text{additive credit:}&\quad \Gamma_C(x,c),\\
\text{intervention effect:}&\quad
F_c(x)-F_c(x^{\mathrm{do}(C\leftarrow C_0)}).
\end{aligned}
```

TCAV targets the second and third. ACE automates candidate construction and
then uses TCAV. ProtoMIL can yield additive logit credit. GECKO yields a
language-indexed slide representation whose coordinates can be inspected, but
its learned nonlinear concept weighting is not automatically score credit.

## 3. Concept Validity Has Three Layers

A concept explanation requires three separate alignments:

```math
\text{samples}
\xrightarrow{\text{representation}}
\text{latent object}
\xrightarrow{\text{semantic naming}}
\text{human concept}
\xrightarrow{\text{model functional}}
\text{prediction claim}.
```

Failure at each arrow has a different meaning:

- representation failure: examples are not geometrically coherent;
- semantic failure: a coherent cluster has no stable pathology meaning;
- functional failure: a real concept is not relevant to the target score.

## 4. Local Versus Global

A derivative at one activation is local:

```math
D_C F_c(x)
=\nabla_{\phi_l}F_c(x)^{\top}v_C^{(l)}.
```

A class-level average or sign frequency aggregates those local values. It is
global only relative to the chosen sample distribution, layer, concept set, and
reference set.

