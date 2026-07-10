# CTransPath Local-Global Backbone

CTransPath changes the patch encoder before changing the positive relation. Its
"global" context is global relative to tokens inside a cropped pathology image,
not global relative to all patches in a WSI.

Primary anchor:

- Wang et al. "Transformer-based Unsupervised Contrastive Learning for
  Histopathological Image Classification." Medical Image Analysis 2022.
  https://pubmed.ncbi.nlm.nih.gov/35952419/

## Input And CNN Stem

Let the input pathology patch be:

```math
x
\in
\mathbb{R}^{H\times W\times 3}.
```

The paper replaces the ordinary Swin patch partition with three convolutional
layers having kernels:

```math
3\times 3,
\qquad
3\times 3,
\qquad
1\times 1.
```

The resulting local feature map is:

```math
F^{(0)}
=
\mathrm{CNN}_{\theta_{\mathrm{stem}}}(x)
\in
\mathbb{R}^{
\frac{H}{4}
\times
\frac{W}{4}
\times
C_0
}.
```

The stem is a learned nonlinear tokenization map. It already mixes local
pixels before self-attention.

## Window Tensorization

For a window size `M\times M`, the number of non-overlapping windows at
the first stage is:

```math
N_{\mathrm{win}}
=
\frac{H}{4M}
\frac{W}{4M}.
```

For one window, reshape:

```math
I
\in
\mathbb{R}^{M^2\times C}.
```

For one attention head:

```math
Q
=
IW_Q,
\qquad
K
=
IW_K,
\qquad
V
=
IW_V,
```

where:

```math
Q,K,V
\in
\mathbb{R}^{M^2\times d_h}.
```

The paper writes window attention as:

```math
\mathrm{W\mbox{-}SA}(I)
=
\mathrm{Softmax}
\left(
\frac{
QK^{\top}
}{
\sqrt{d_h}
}
+
B
\right)
V,
```

where:

```math
B
\in
\mathbb{R}^{M^2\times M^2}
```

is the relative-position bias for token pairs inside the window.

## Multi-Head Window Attention

For `A` heads:

```math
\mathrm{W\mbox{-}MSA}(I)
=
\mathrm{Concat}
\left(
\mathrm{head}_1,
\ldots,
\mathrm{head}_A
\right)
W_O.
```

Each token directly interacts with at most `M^2` tokens in that layer.
The support is local and block structured.

## Shifted Windows

Window attention alone disconnects tokens in different windows. CTransPath
inherits Swin's alternating shift:

```math
\Delta
=
\left(
\left\lfloor
\frac{M}{2}
\right\rfloor,
\left\lfloor
\frac{M}{2}
\right\rfloor
\right).
```

The shifted partition changes which token pairs share a window. If
`A_{\mathrm{W}}` and `A_{\mathrm{SW}}` are the binary attention
supports, two consecutive blocks propagate along:

```math
A_{\mathrm{two\mbox{-}block}}
\subseteq
A_{\mathrm{W}}
\cup
A_{\mathrm{SW}}
\cup
A_{\mathrm{SW}}A_{\mathrm{W}}.
```

Cross-window context emerges by composition, not by dense all-token attention
in one layer.

## Exact Residual Block Map

Using layer normalization `\mathrm{LN}` and an MLP, the paper gives:

```math
\widehat Z^{\ell}
=
\mathrm{W\mbox{-}MSA}
\left(
\mathrm{LN}
\left(
Z^{\ell-1}
\right)
\right)
+
Z^{\ell-1},
```

```math
Z^{\ell}
=
\mathrm{MLP}
\left(
\mathrm{LN}
\left(
\widehat Z^{\ell}
\right)
\right)
+
\widehat Z^{\ell},
```

```math
\widehat Z^{\ell+1}
=
\mathrm{SW\mbox{-}MSA}
\left(
\mathrm{LN}
\left(
Z^{\ell}
\right)
\right)
+
Z^{\ell},
```

```math
Z^{\ell+1}
=
\mathrm{MLP}
\left(
\mathrm{LN}
\left(
\widehat Z^{\ell+1}
\right)
\right)
+
\widehat Z^{\ell+1}.
```

The residual path preserves an identity channel while attention and the MLP
add contextual corrections.

## Four-Stage Hierarchy

CTransPath uses a four-stage Swin hierarchy. Write:

```math
F^{(s+1)}
=
\mathcal{P}^{(s)}
\left(
\mathcal{T}^{(s)}
\left(
F^{(s)}
\right)
\right),
\qquad
s\in\{0,1,2\},
```

where `\mathcal{T}^{(s)}` is a stack of window and shifted-window blocks
and `\mathcal{P}^{(s)}` is the stage transition that reduces spatial
resolution while increasing channel capacity.

The terminal patch representation is:

```math
h
=
\mathcal{R}_{\mathrm{patch}}
\left(
F^{(3)}
\right).
```

This `h` is the representation passed to the contrastive projection
head during pretraining and retained as a feature for downstream tasks.

## Complexity

Let:

```math
T
=
\frac{H}{4}
\frac{W}{4}
```

be the first-stage token count.

Dense self-attention has pairwise score cost:

```math
\Theta
\left(
T^2d_h
\right).
```

Window attention has:

```math
\frac{T}{M^2}
\text{ windows}
```

with `M^4` pair scores per window, giving:

```math
\Theta
\left(
TM^2d_h
\right).
```

For fixed `M`, the pairwise attention cost is linear in token count.
Long-range interaction depth grows with the number of shifted-window
compositions and stage transitions.

## Two Different Context Scales

The CNN stem provides local inductive bias:

```math
x
\xrightarrow{
\mathrm{CNN}
}
\text{local texture and cellular features}.
```

The Swin hierarchy provides within-patch contextual interaction:

```math
\text{stem tokens}
\xrightarrow{
\mathrm{W\mbox{-}MSA/SW\mbox{-}MSA}
}
\text{multi-scale patch representation}.
```

Neither operation mixes two WSI patches:

```math
x_{ij}
\quad
\text{and}
\quad
x_{ik},
\qquad
j\ne k.
```

Slide context still requires a downstream MIL, graph, sequence, hierarchy, or
retrieval operator.

## C/R/G/S Placement

For the backbone alone:

```text
\mathcal{G}:
    a regular image-token lattice with alternating window supports

\mathcal{C}:
    CNN stem plus hierarchical W-MSA and SW-MSA blocks

\mathcal{R}:
    terminal patch representation before the SRCL projector

\mathcal{S}:
    none inside the backbone; SRCL supplies the pretraining target
```

## Failure Principle

CTransPath can encode broader context inside a crop while remaining blind to
where that crop sits in the slide. Calling both objects "global context"
collapses two different geometries:

```math
\text{global within patch}
\ne
\text{global within WSI}.
```
