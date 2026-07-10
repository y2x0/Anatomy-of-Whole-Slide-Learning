# PLIP False Negatives And Retrieval Selection Bias

Primary anchor:

- Huang et al. "A Visual-Language Foundation Model for Pathology Image
  Analysis Using Medical Twitter." Nature Medicine 2023.
  https://www.nature.com/articles/s41591-023-02504-3

## One Post, Multiple Images

Suppose one post-level text `t` accompanies `m` images:

```math
\left{
x_1,ldots,x_m
\right}
\longleftrightarrow
t.
```

If converted into diagonal pairs, duplicate text embeddings satisfy:

```math
v_1=cdots=v_m=v.
```

For any associated image `x_i`, the row-softmax probability assigned to its
designated copy is bounded:

```math
p^{I\rightarrow T}_{ii}
\le
\frac{1}{m}.
```

Hence the row loss obeys:

```math
-\log p^{I\rightarrow T}_{ii}
\ge
\log m.
```

This irreducible loss is produced by indexing, not semantic error.

## Same-Diagnosis False Negatives

Let `Y_i` denote diagnosis. The expected number of same-diagnosis off-diagonal
texts for image `i` is:

```math
\mathbb{E}
\left[
F_i
\mid
Y_i=y
\right]
=
\left(B-1\right)
p_{\mathrm{batch}}(Y=y).
```

As batch size grows, candidate diversity improves but same-class false
negatives also grow linearly. Their gradient mass is:

```math
M_i^{\mathrm{FN}}
=
\sum_{
j\ne i:
Y_j=Y_i
}
p^{I\rightarrow T}_{ij}.
```

Hard semantically related captions can dominate this mass because softmax
weights increase exponentially with similarity.

## Retrieval-Selected PathLAION

Let a baseline encoder `f_0` define distance:

```math
d_0(x,q)
=
1-
\frac{f_0(x)^{\top}f_0(q)}
{\|f_0(x)\|_2\|f_0(q)\|_2}.
```

The selected subset is approximately:

```math
\mathcal{D}_{\mathrm{sel}}
=
\left{
(x,t):
\min_{q\in\mathcal{Q}}d_0(x,q)
\le
c
\right}.
```

Training a new encoder on this set estimates compatibility conditional on
baseline proximity:

```math
p_{\mathrm{train}}(x,t)
=
p
\left(
x,t
\mid
\min_q d_0(x,q)\le c
\right).
```

This creates a feedback path:

```math
f_0
\longrightarrow
\text{selected corpus}
\longrightarrow
f_1.
```

The second model can improve within that support while remaining blind to
pathology appearances that `f_0` failed to retrieve.

## Importance Weighting Thought Experiment

If target and selected densities were known, a target risk could be written:

```math
\mathbb{E}_{p_{\mathrm{target}}}
\left[
\ell(X,T)
\right]
=
\mathbb{E}_{p_{\mathrm{sel}}}
\left[
w(X,T)
\ell(X,T)
\right],
```

with:

```math
w(x,t)
=
\frac{p_{\mathrm{target}}(x,t)}
{p_{\mathrm{sel}}(x,t)}.
```

This identity requires support overlap:

```math
p_{\mathrm{target}}(x,t)>0
\Longrightarrow
p_{\mathrm{sel}}(x,t)>0.
```

Retrieval can violate this condition by assigning zero inclusion probability
to visually distant subpopulations.

## Failure Principle

More weak pairs reduce estimator variance only relative to the selected pair
law. They do not automatically reduce selection bias or relation noise.
