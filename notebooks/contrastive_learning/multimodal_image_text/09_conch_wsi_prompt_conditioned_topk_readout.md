# CONCH WSI Prompt-Conditioned Top-K Readout

Primary anchor:

- Lu et al. "A Visual-Language Foundation Model for Computational Pathology."
  Nature Medicine 2024. https://arxiv.org/abs/2307.12914

CONCH evaluates zero-shot WSI classification through the MI-Zero-style
pipeline. The slide prediction is not a direct output of image-text
pretraining; it is a prompt-conditioned patch scoring rule followed by top-k
aggregation.

## Tile And Prompt Embeddings

For slide `i` with tiles `x_{ij}`:

```math
u_{ij}
=
\frac{f_{\theta}(x_{ij})}
{\|f_{\theta}(x_{ij})\|_2},
\qquad
j=1,\ldots,n_i.
```

For class `c` with prompt ensemble prototype `v_c`:

```math
v_c
=
\frac{
\sum_{r=1}^{R_c}
v_{c,r}
}{
\left\|
\sum_{r=1}^{R_c}
v_{c,r}
\right\|_2
}.
```

Class-specific tile evidence is:

```math
s_{ijc}
=
u_{ij}^{\top}v_c.
```

## Class-Specific Top-K Sets

Let `I_{ic}^{(K)}` be indices of the `K` largest values among tile scores for
class `c`:

```math
I_{ic}^{(K)}
=
\mathop{\arg\max}_{
I\subseteq\{1,\ldots,n_i\},
|I|=K
}
\sum_{j\in I}s_{ijc}.
```

The slide-class score is:

```math
S_{ic}^{(K)}
=
\frac{1}{K}
\sum_{j\in I_{ic}^{(K)}}
s_{ijc}.
```

Prediction is:

```math
\widehat y_i
=
\arg\max_c
S_{ic}^{(K)}.
```

The paper selects `K` from a finite validation set:

```math
K
\in
\left\{
1,5,10,50,100
\right\}.
```

Writing `A_{\mathrm{val}}(K)` for the validation metric, the selected readout
is more precisely:

```math
\widehat K
=
\arg\max_{K\in\{1,5,10,50,100\}}
A_{\mathrm{val}}(K).
```

Thus `zero-shot` describes the absence of task-specific parameter fitting
for the class prototypes, but not the absence of labeled validation data for
the aggregation hyperparameter. The reported test score is consequently
conditional on the finite candidate set and on the validation-selection
procedure.

## The Surviving Statistic

Sort class-specific scores:

```math
s_{i(1)c}
\ge
s_{i(2)c}
\ge
\cdots
\ge
s_{i(n_i)c}.
```

Then:

```math
S_{ic}^{(K)}
=
\frac{1}{K}
\sum_{r=1}^{K}
s_{i(r)c}.
```

Only the upper empirical tail of the prompt-conditioned score distribution
survives. Patch prevalence below rank `K` is discarded.

## Prompt Perturbations Change Support

Let the class prototype change from `v_c` to `v_c+\Delta v_c`. Tile-score
perturbation is:

```math
\Delta s_{ijc}
=
u_{ij}^{\top}\Delta v_c.
```

Even a small perturbation can change the selected support whenever two scores
are close. Define the boundary gap:

```math
g_{ic}^{(K)}
=
s_{i(K)c}
-
s_{i(K+1)c}.
```

If:

```math
\max_j
\left|
u_{ij}^{\top}\Delta v_c
\right|
<
\frac{g_{ic}^{(K)}}{2},
```

the top-k support is stable. When the gap is small, prompt wording can switch
which tissue regions enter the slide score.

## Piecewise Gradient

Away from score ties, the gradient with respect to a selected tile embedding
is:

```math
\frac{\partial S_{ic}^{(K)}}
{\partial u_{ij}}
=
\begin{cases}
v_c/K,
&
j\in I_{ic}^{(K)},
\\
0,
&
j\notin I_{ic}^{(K)}.
\end{cases}
```

The readout is sparse and piecewise smooth. At support-switch boundaries it is
nondifferentiable.

## C/R/G/S Placement

```math
\mathcal{C}
=
\text{independent tile encoding},
\qquad
\mathcal{R}
=
\text{class-specific top-k mean},
```

```math
\mathcal{G}
=
\text{text-defined angular evidence},
\qquad
\mathcal{S}
=
\text{prompt set and validation-selected }K.
```

The WSI method has no inter-tile context operator. Its slide inductive bias
comes from sparse order-statistic aggregation.
