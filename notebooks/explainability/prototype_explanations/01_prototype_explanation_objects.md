# Prototype Explanation Objects

## 1. One Name, Four Objects

For a slide bag

```math
B_i=\{h_{ij}\}_{j=1}^{n_i},
\qquad h_{ij}\in\mathbb R^d,
```

a prototype may be:

```math
\begin{aligned}
\text{latent exemplar:}&\quad p_k\in\mathbb R^d,\\
\text{mixture component:}&\quad (\pi_{ik},\mu_{ik},\Sigma_{ik}),\\
\text{attention anchor:}&\quad p_k\text{ inducing }s_{ijk},\\
\text{sparse concept direction:}&\quad f_k\in\mathbb R^d.
\end{aligned}
```

These induce different claims. A latent exemplar supports *resemblance*. A
mixture component supports *soft assignment and prevalence*. An attention
anchor supports *routing*. A sparse direction supports *coordinate-level
decomposition*.

## 2. The Explanation Chain

Let

```math
s_{ijk}=S(h_{ij},p_k),
\qquad
u_{ik}=A_k(\{s_{ijk},h_{ij}\}_{j=1}^{n_i}),
\qquad
F_{ic}=H_c(u_{i1},\ldots,u_{iK}).
```

There are at least four non-equivalent questions:

```math
\begin{aligned}
&\text{Which patch resembles }p_k? &&s_{ijk},\\
&\text{Is prototype }k\text{ present?} &&u_{ik},\\
&\text{How much score credit does }k\text{ receive?}
&&\Gamma_{ikc},\\
&\text{What happens if }k\text{ is removed?}
&&F_{ic}-F_{ic}^{(-k)}.
\end{aligned}
```

Equality among these quantities requires architectural assumptions. It does not
follow from calling the model prototype-based.

## 3. Identifiability

If a representation is transformed by an invertible matrix

```math
h' = Ah,
\qquad p'_k=Ap_k,
```

Euclidean prototype rankings are generally not preserved because

```math
\|h'-p'_k\|_2^2=(h-p_k)^\top A^\top A(h-p_k).
```

Thus prototypes are explanations relative to a learned metric geometry. They
are not intrinsic tissue categories unless that geometry and its semantic
alignment are independently validated.

## 4. Minimal Valid Claim

A prototype display can safely claim only:

```text
Under this encoder, metric, aggregation rule, and reference set, these patches
are highly associated with this prototype-derived model quantity.
```

Stronger claims about morphology, necessity, or causality require additional
tests.

