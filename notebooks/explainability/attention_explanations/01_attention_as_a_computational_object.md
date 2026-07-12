# Attention As A Computational Object

Primary anchors:

- Jain, Wallace. "Attention is not Explanation." NAACL 2019.
  https://arxiv.org/abs/1902.10186
- Wiegreffe, Pinter. "Attention is not not Explanation." EMNLP 2019.
  https://arxiv.org/abs/1908.04626

## Readout Attention

For patch representations:

```math
H
=
\left[
h_1^{\top};
\ldots;
h_n^{\top}
\right]
\in
\mathbb{R}^{n\times d},
```

attention scores and weights are:

```math
e_j
=
s_{\theta}(h_j,H),
```

```math
\alpha_j
=
\frac{
\exp(e_j)
}{
\sum_{k=1}^{n}
\exp(e_k)
}.
```

The pooled representation is:

```math
z
=
\sum_{j=1}^{n}
\alpha_jv_j,
\qquad
v_j
=
V_{\theta}(h_j,H).
```

The weight is a normalized coefficient in representation construction.

## Four Different Objects

Attention mass:

```math
\alpha_j.
```

Value content:

```math
v_j.
```

Class score alignment:

```math
w_c^{\top}v_j.
```

Finite removal effect:

```math
F_c(H)
-
F_c(H_{-j}).
```

These coincide only under restrictive conditions.

## Relative, Not Absolute, Weight

Adding one candidate changes every weight:

```math
\alpha_j'
=
\frac{
\exp(e_j)
}{
\exp(e_{n+1})
+
\sum_{k=1}^{n}
\exp(e_k)
}.
```

Thus the weight assigned to unchanged patch `j` depends on which other patches
are present.

## Shift Invariance

For constant `c`:

```math
\mathrm{softmax}(e+c\mathbf{1})
=
\mathrm{softmax}(e).
```

Only score differences are identifiable from normalized attention.

## Temperature

```math
\alpha_j(\tau)
=
\frac{
\exp(e_j/\tau)
}{
\sum_k\exp(e_k/\tau)
}.
```

As temperature approaches zero, attention approaches selection among maxima.
As temperature grows, it approaches uniform weighting. The underlying value
vectors can be unchanged while the displayed heatmap sharpness changes.

## Explanation Claim

Attention directly explains:

```math
\text{how normalized value mass enters this readout}.
```

It does not directly explain:

```math
\text{which patch caused the final clinical prediction}.
```

That stronger claim requires accounting for values, the head, context paths,
and counterfactual behavior.
