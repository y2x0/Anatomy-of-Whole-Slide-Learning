# TCAV Directional Derivatives

## 1. Network Factorization

Factor the class logit through layer `l`:

```math
F_c(x)=h_{l,c}(\phi_l(x)),
\qquad
h_{l,c}:\mathbb R^{d_l}\to\mathbb R.
```

For unit CAV `v_C`, TCAV defines conceptual sensitivity as

```math
S_{C,c,l}(x)
=\lim_{\epsilon\to0}
\frac{
h_{l,c}(\phi_l(x)+\epsilon v_C^{(l)})
-h_{l,c}(\phi_l(x))
}{\epsilon}.
```

When differentiable,

```math
S_{C,c,l}(x)
=\nabla h_{l,c}(\phi_l(x))^{\top}v_C^{(l)}.
```

This is a local directional derivative in activation space.

## 2. What the Sign Means

```math
S_{C,c,l}(x)>0
```

means an infinitesimal displacement along the chosen CAV increases the class
logit at the current activation. It does not imply:

```math
\text{concept present}
\quad\text{or}\quad
\text{concept necessary}
\quad\text{or}\quad
\text{finite concept insertion raises the score}.
```

Presence and sensitivity can disagree. A concept may be absent but the model
would respond strongly if moved toward it; it may be present where the score is
locally saturated.

## 3. Finite Changes and Curvature

For displacement size `delta`, Taylor expansion gives

```math
h_{l,c}(z+\delta v_C)-h_{l,c}(z)
=\delta S_{C,c,l}(x)
+\frac{\delta^2}{2}
v_C^{\top}\nabla^2h_{l,c}(z+\xi\delta v_C)v_C
```

for some `xi` between zero and one. TCAV omits the curvature term. Large
concept edits, ReLU boundary crossings, and off-manifold directions can reverse
the local conclusion.

## 4. Correlated Concepts

If two unit CAVs satisfy

```math
v_C^{\top}v_D\approx1,
```

then their sensitivities will be similar for any local gradient. Separate high
TCAV scores cannot identify which correlated concept the model uses. A
conditional direction can remove the span of known confounders:

```math
v_{C\mid D}
=\frac{(I-P_D)v_C}{\|(I-P_D)v_C\|_2},
```

provided the residual is nonzero. This changes the estimand to sensitivity
toward the component of `C` orthogonal to `D`.

