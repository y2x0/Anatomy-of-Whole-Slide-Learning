# WSI Cox Statistic

This note asks what slide statistic the Cox objective actually optimizes.

Let a WSI be represented by patch embeddings:

```math
H_i=\{h_{ij}\}_{j=1}^{n_i},
\qquad h_{ij}\in\mathbb{R}^d.
```

A slide encoder produces:

```math
z_i=\mathcal{R}(\mathcal{C}(H_i;G_i)).
```

A Cox head gives:

```math
\eta_i=f_\theta(z_i).
```

The loss only depends on the slide through $\eta_i$. Therefore the survival
statistic is whatever part of $H_i$ changes risk-set ordering after
$\mathcal{C},\mathcal{R},f_\theta$.

## Linear Attention MIL Cox

Suppose:

```math
z_i=\sum_{j=1}^{n_i}a_{ij}h_{ij},
\qquad
\sum_j a_{ij}=1,
\qquad
a_{ij}\ge0,
```

and:

```math
\eta_i=w^\top z_i.
```

Then:

```math
\eta_i
=
\sum_{j=1}^{n_i}a_{ij}w^\top h_{ij}.
```

Define patch risk evidence:

```math
e_{ij}=w^\top h_{ij}.
```

Then:

```math
\eta_i=\mathbb{E}_{j\sim a_i}[e_{ij}].
```

So the surviving statistic is the attention-weighted first moment of patch risk
evidence.

## Attention Gradient

For one event $i$, the Cox loss is:

```math
\mathcal{L}_i
=
-\eta_i+\log\sum_{k\in R_i}\exp(\eta_k).
```

For slide $m\in R_i$:

```math
\frac{\partial\mathcal{L}_i}{\partial z_m}
=
(p_{im}-\mathbf{1}[m=i])w.
```

For an attention-pooled slide:

```math
z_m=\sum_j a_{mj}h_{mj}.
```

If attention weights are treated as fixed for the moment:

```math
\frac{\partial\mathcal{L}_i}{\partial h_{mj}}
=
(p_{im}-\mathbf{1}[m=i])a_{mj}w.
```

Thus event slides receive negative pressure along $w$; competing risk-set
slides receive positive pressure along $w$, weighted by their Cox softmax
probability.

If $a_{mj}$ depends on $h_{mj}$, an additional term appears:

```math
\frac{\partial\eta_m}{\partial h_{mj}}
=
a_{mj}w
+
\sum_\ell
(w^\top h_{m\ell})
\frac{\partial a_{m\ell}}{\partial h_{mj}}.
```

The second term is the attention-routing gradient. It tells the model not only
which patch features increase risk, but which patches should receive weight.

## Mean Cox

If:

```math
z_i=\frac{1}{n_i}\sum_jh_{ij},
```

then:

```math
\eta_i
=
\frac{1}{n_i}\sum_jw^\top h_{ij}.
```

The Cox score is the average patch risk evidence. Sparse positive morphology is
diluted by benign tissue.

## Max Cox

If:

```math
z_i=\max_j h_{ij}
```

coordinatewise, then:

```math
\eta_i=w^\top \max_j h_{ij}
```

is not generally the same as:

```math
\max_j w^\top h_{ij}.
```

Coordinatewise max pooling can assemble a risk vector from different patches,
which may create a non-existent synthetic high-risk patch.

Instance-max Cox would instead define:

```math
\eta_i=\max_j w^\top h_{ij}.
```

That statistic is sparse but brittle.

## Prototype Cox

Let prototypes be $c_1,\ldots,c_M$. Define soft assignments:

```math
q_{ijm}
=
\frac{\exp[-\|h_{ij}-c_m\|^2/\tau]}
{\sum_{\ell=1}^M\exp[-\|h_{ij}-c_\ell\|^2/\tau]}.
```

Prototype prevalence:

```math
p_{im}
=
\frac{1}{n_i}\sum_{j=1}^{n_i}q_{ijm}.
```

Then Cox risk:

```math
\eta_i=w^\top p_i
=
\sum_{m=1}^M w_mp_{im}.
```

The surviving statistic is a distribution over morphologic prototypes. This is
better suited to diffuse risk patterns than attention over a few patches.

## Graph Cox

Let:

```math
h_{ij}^{(L)}
=
\operatorname{GNN}_{\theta}(h_{ij}^{(0)},G_i).
```

Then:

```math
z_i=\operatorname{READOUT}(\{h_{ij}^{(L)}\}),
\qquad
\eta_i=f(z_i).
```

The Cox loss sees graph context only after readout. Pairwise interactions
survive only if message passing has encoded them into node states or the readout
keeps interaction statistics.

If the readout is mean:

```math
\eta_i
=
w^\top
\frac{1}{n_i}\sum_jh_{ij}^{(L)}.
```

The final statistic is still a first moment, but of contextualized nodes.

## Transformer Cox

For transformer MIL:

```math
\widetilde H_i
=
\operatorname{Transformer}(H_i+P_i),
```

with readout:

```math
z_i=\widetilde h_{\mathrm{cls}}
\quad\text{or}\quad
z_i=\frac{1}{n_i}\sum_j\widetilde h_{ij}.
```

The Cox head again produces:

```math
\eta_i=f(z_i).
```

Attention can encode interactions before readout, but the survival loss still
compresses them into one score.

## The Identifiability Problem

If two slide encoders produce scores related by a monotone transformation:

```math
\eta_i' = a\eta_i+b,
\qquad a>0,
```

they can preserve concordance. But the Cox partial likelihood is not invariant
to scaling $a$, because softmax denominators change. Larger $a$ sharpens
risk-set probabilities:

```math
\frac{\exp(a\eta_i)}
{\sum_{j\in R_i}\exp(a\eta_j)}.
```

Thus Cox training learns both ordering and separation scale, but not absolute
hazard without baseline recovery.

## Dense Summary

For a WSI Cox model:

```math
H_i
\xrightarrow{\mathcal{C}}
\widetilde H_i
\xrightarrow{\mathcal{R}}
z_i
\xrightarrow{f}
\eta_i
\xrightarrow{\mathrm{risk\ sets}}
\mathcal{L}_{\mathrm{Cox}}.
```

The survival statistic is:

```math
\eta_i=f(\mathcal{R}(\mathcal{C}(H_i;G_i))).
```

Every design question becomes:

```text
which morphologic statistic is allowed to affect this one scalar ordering?
```
