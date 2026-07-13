# MambaMIL Vanilla Scan And MIL Head

Source:

`text
Yang, Wang, and Chen, MambaMIL: Enhancing Long Sequence Modeling with
Sequence Reordering in Computational Pathology, MICCAI 2024.
https://arxiv.org/abs/2403.06800
`

## 1. Input Projection

The paper maps a patch sequence to:

`math
X_i
=
\begin{bmatrix}
x_{i1}^{\top}\\
\vdots\\
x_{iL_i}^{\top}
\end{bmatrix}
\in
\mathbb{R}^{L_i\times D}.
`

The released implementation projects to width 512:

`math
H_i^{(0)}
=
\phi\left(X_iW_1+b_1\right)
\in
\mathbb{R}^{L_i\times 512}.
`

The row order is retained by the vanilla Mamba branch.

## 2. Paper-Level MambaMIL Block

For one branch:

`math
X_i^{\prime}
=
\mathrm{Norm}\left(H_i^{(\ell)}\right).
`

The paper writes the Mamba path as:

`math
Y_i
=
\mathrm{SSM}
\left(
\mathrm{SiLU}
\left(
\mathrm{Conv1D}
\left(
\mathrm{Linear}\left(X_i^{\prime}\right)
\right)
\right)
\right).
`

The gate is:

`math
Z_i
=
\mathrm{SiLU}
\left(
\mathrm{Linear}\left(X_i^{\prime}\right)
\right),
\qquad
X_i^{\prime\prime}
=
Z_i\odot Y_i.
`

Stacked blocks use residual addition:

`math
H_i^{(\ell+1)}
=
H_i^{(\ell)}
+
X_i^{\prime\prime}.
`

The SSM is the context operator; convolution, normalization, gate, and residual
determine how its state enters the token field.

## 3. Released-Code Attention Readout

After the stacked blocks and final normalization, the released model scores each
token:

`math
s_{it}
=
w_2^{\top}
\tanh\left(W_1h_{it}+b_1\right)
+
b_2.
`

Normalize across tokens:

`math
\alpha_{it}
=
\frac{\exp(s_{it})}
{\sum_{u=1}^{L_i}\exp(s_{iu})}.
`

The slide vector is:

`math
z_i
=
\sum_{t=1}^{L_i}
\alpha_{it}h_{it}
\in
\mathbb{R}^{512}.
`

The classifier is:

`math
\widehat p_i
=
\mathrm{softmax}
\left(
z_iW_{\mathrm{cls}}+b_{\mathrm{cls}}
\right).
`

For the survival configuration, output logits become hazard probabilities:

`math
\widehat h_{ik}
=
\sigma(\ell_{ik}),
\qquad
\widehat S_i(k)
=
\prod_{r=1}^{k}
\left(1-\widehat h_{ir}\right).
`

This is a task-head choice after Mamba; the SSM does not imply a particular
survival likelihood.

## 4. Context Variants

The released model exposes:

`text
Mamba:
    one forward selective scan

BiMamba:
    forward and reverse sequence processing

SRMamba:
    original-order branch plus segment-reordered branch
`

Bidirectionality reduces direction asymmetry but does not make arbitrary patch
order irrelevant. SR-Mamba changes the order in a structured way and restores
the reordered outputs before fusion.

## 5. C/R/G/S Placement

`text
C:
    stacked selective Mamba scans with convolution, gating, and residuals

R:
    released-code scalar attention weighted mean of contextualized tokens

G:
    input sequence order; SR-Mamba adds segment-transposed order

S:
    classification cross-entropy or discrete hazard/survival head
`
