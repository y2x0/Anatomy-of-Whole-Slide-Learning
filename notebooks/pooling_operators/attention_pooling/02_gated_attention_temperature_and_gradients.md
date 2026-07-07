# Gated Attention, Temperature, And Gradients

The gated attention variant in ABMIL replaces a single $\tanh$ score with a
feature-wise interaction:

```math
s_{ij}
=
w^\top
\left(
\tanh(Vh_{ij})
\odot
\mathrm{sigmoid}(Uh_{ij})
\right).
```

The pooling formula is unchanged:

```math
a_{ij}
=
\mathrm{softmax}_{j}s_{ij},
\qquad
z_i
=
\sum_j a_{ij}h_{ij}.
```

## Why The Gate Matters

The ungated score is:

```math
s(h)=w^\top\tanh(Vh).
```

The gated score is:

```math
s(h)
=
\sum_{r=1}^{L}
w_r
\tanh(v_r^\top h)
\mathrm{sigmoid}(u_r^\top h).
```

Each hidden coordinate contributes only when two learned tests agree:

```text
tanh branch:
    signed morphology response

sigmoid branch:
    soft selector or gate
```

The gate can suppress a feature dimension even when the $\tanh$ branch is large.
This makes the score more expressive than a single saturating projection.

## Softmax Temperature

Introduce inverse temperature $\beta$:

```math
a_{ij}^{(\beta)}
=
\frac{\exp(\beta s_{ij})}
{\sum_\ell\exp(\beta s_{i\ell})}.
```

The entropy of attention is:

```math
H(a_i)
=
-\sum_j a_{ij}\log a_{ij}.
```

Increasing $\beta$ usually lowers entropy:

```math
\beta\uparrow
\quad\Rightarrow\quad
H(a_i)\downarrow.
```

This changes the statistical question:

```text
low beta:
    broad prevalence-like summary

high beta:
    sparse evidence selection
```

## Score Gradients

Let:

```math
z_i
=
\sum_j a_{ij}v_{ij},
\qquad
v_{ij}=v_\theta(h_{ij}).
```

For the softmax weights:

```math
\frac{\partial a_{ij}}{\partial s_{i\ell}}
=
a_{ij}
(\mathbf{1}\{j=\ell\}-a_{i\ell}).
```

Therefore:

```math
\frac{\partial z_i}{\partial s_{i\ell}}
=
a_{i\ell}(v_{i\ell}-z_i).
```

This is the most important local geometry of attention pooling. A patch changes
the slide embedding through its difference from the current pooled embedding.

If the class logit is:

```math
o_i
=
q^\top z_i,
```

then:

```math
\frac{\partial o_i}{\partial s_{i\ell}}
=
a_{i\ell}q^\top(v_{i\ell}-z_i).
```

A patch receives higher attention when moving the pooled embedding toward that
patch increases the class logit.

## Attention Is Competitive

The same derivative shows why attention is competitive. Increasing $s_{i\ell}$
does not only increase $a_{i\ell}$. It decreases other weights because:

```math
\sum_j a_{ij}=1.
```

Thus a high-attention patch can suppress diffuse evidence elsewhere. This is
desirable for sparse-positive tasks and dangerous for tasks where risk is
distributed across tissue.

## Attention Weight Versus Evidence

The attention weight $a_{ij}$ is not itself a class probability. The final logit
depends on both the weight and the value:

```math
o_i
=
q^\top
\sum_j a_{ij}v_{ij}
=
\sum_j a_{ij}q^\top v_{ij}.
```

A high-attention patch can contribute negative evidence if:

```math
q^\top v_{ij}<0.
```

Attention maps are therefore maps of readout weight, not guaranteed maps of
positive pathology.

## Dense Summary

Gated attention changes the scoring map:

```math
h
\mapsto
s_\theta(h).
```

Softmax changes scores into a probability measure:

```math
s
\mapsto
a.
```

The readout remains:

```math
z_i
=
\mathbb{E}_{j\sim a_i}[v_{ij}].
```

The mathematical burden is on $s_\theta$ and $v_\theta$: they must make a single
weighted first moment sufficient for the slide-level task.

