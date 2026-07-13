# SR-Mamba Sequence Reordering Operator

Source:

`text
Yang, Wang, and Chen, MambaMIL: Enhancing Long Sequence Modeling in
Computational Pathology, MICCAI 2024.
https://arxiv.org/abs/2403.06800
`

## 1. Segmenting A Long Sequence

Let the input be:

`math
X_i
\in
\mathbb{R}^{L_i\times d}.
`

Choose segment size R:

`math
N_i
=
\left\lceil\frac{L_i}{R}\right\rceil,
\qquad
L_i^{\mathrm{pad}}
=
RN_i.
`

Pad with zero rows:

`math
X_i^{\mathrm{pad}}
\in
\mathbb{R}^{L_i^{\mathrm{pad}}\times d}.
`

Define the segment array:

`math
\mathsf{X}_i[r,n,:]
=
X_i^{\mathrm{pad}}[nR+r,:],
\qquad
r=0,\ldots,R-1,
\quad
n=0,\ldots,N_i-1.
`

Thus:

`math
\mathsf{X}_i
\in
\mathbb{R}^{R\times N_i\times d}.
`

## 2. The Reordered Sequence

The original sequence visits each segment in order:

`math
X_i^{\mathrm{orig}}[nR+r,:]
=
\mathsf{X}_i[r,n,:].
`

The reordered sequence visits the same within-segment position across all
segments first:

`math
X_i^{\mathrm{reord}}[rN_i+n,:]
=
\mathsf{X}_i[r,n,:].
`

Equivalently, transpose the first two axes of the padded segment array and
flatten. Features do not change, but scan locality does.

## 3. Original-Order Branch

Normalize:

`math
X_i^{\prime}
=
\mathrm{Norm}(X_i).
`

Then:

`math
Y_i
=
\mathrm{SSM}
\left(
\mathrm{SiLU}
\left(
\mathrm{Conv1D}
\left(
\mathrm{Linear}(X_i^{\prime})
\right)
\right)
\right).
`

Use the gate:

`math
Z_i
=
\mathrm{SiLU}
\left(
\mathrm{Linear}(X_i^{\prime})
\right),
\qquad
X_i^{\prime\prime}
=
Z_i\odot Y_i.
`

## 4. Reordered Branch And Restoration

The second branch scans:

`math
X_{i,r}^{\prime}
=
\mathrm{Norm}\left(X_i^{\mathrm{reord}}\right),
`

`math
Y_{i,r}
=
\mathrm{SSM}
\left(
\mathrm{SiLU}
\left(
\mathrm{Conv1D}
\left(
\mathrm{Linear}(X_{i,r}^{\prime})
\right)
\right)
\right).
`

Let psi restore reordered outputs to original row positions:

`math
Y_{i,r}^{\mathrm{restored}}
=
\psi(Y_{i,r}).
`

Gate after restoration:

`math
X_{i,r}^{\prime\prime}
=
Z_i\odot Y_{i,r}^{\mathrm{restored}}.
`

Restoration is essential: adding a reordered output before undoing its
permutation would pair a token's state with the wrong patch.

## 5. SR-Mamba Fusion

The paper's output equation is:

`math
X_i^{\mathrm{out}}
=
\mathrm{Linear}
\left(
X_i^{\prime\prime}
+
X_{i,r}^{\prime\prime}
\right)
+
X_i.
`

The branches carry different trajectories:

`text
original scan:
    local sequence neighborhoods in the input order

segment-transposed scan:
    cross-segment relationships at the same within-segment offset
`

The residual keeps a direct path from the pre-block token sequence.

## 6. What Reordering Does Not Do

SR-Mamba does not average over all permutations:

`math
\mathcal{C}_{\mathrm{SR}}(X)
\neq
\frac{1}{|\mathfrak{S}_{L}|}
\sum_{\sigma\in\mathfrak{S}_{L}}
\mathcal{C}_{\mathrm{SSM}}(P_{\sigma}X).
`

It evaluates two structured orders. The hypothesis class remains order-dependent,
but it exposes a second scan geometry without quadratic all-pairs interaction.
