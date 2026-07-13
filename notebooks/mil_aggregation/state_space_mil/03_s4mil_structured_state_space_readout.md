# S4MIL Structured State-Space Readout

Source:

`text
Fillioux et al., Structured State Space Models for Multiple Instance Learning
in Digital Pathology, MICCAI 2023.
https://arxiv.org/abs/2306.15789
`

## 1. Patch Sequence

The pipeline begins with:

`math
U_i
=
\left[
u_{i1}^{\top};\ldots;u_{iL_i}^{\top}
\right]
\in
\mathbb{R}^{L_i\times 1024}.
`

A learned projection maps patch features to model width:

`math
H_i^{(0)}
=
\phi
\left(
U_iW_{\mathrm{in}}
+
\mathbf{1}_{L_i}b_{\mathrm{in}}^{\top}
\right)
\in
\mathbb{R}^{L_i\times d}.
`

The sequence order is inherited from patch extraction/grid convention. It is not
a permutation-invariant set operator.

## 2. S4D Context

For channel or learned state-space head r:

`math
Y_{i,:,r}^{\mathrm{SSM}}
=
\overline K_r*H_{i,:,r}^{(0)}
+
D_rH_{i,:,r}^{(0)}.
`

With channel mixing:

`math
Y_i
=
\mathrm{Mix}
\left(
\mathrm{GELU}
\left(
Y_i^{\mathrm{SSM}}
\right)
\right)
\in
\mathbb{R}^{L_i\times d}.
`

The structured convolution is the parallel form of a linear SSM; normalization,
nonlinearity, and mixing make the full block nonlinear.

## 3. Released S4MIL Readout

The released S4MIL model applies coordinatewise max pooling:

`math
[z_i]_r
=
\max_{1\le t\le L_i}
[Y_i]_{t,r},
\qquad
r=1,\ldots,d.
`

Thus:

`math
z_i
=
\mathrm{MaxPool}_{t}(Y_i)
\in
\mathbb{R}^{d}.
`

The classifier is:

`math
\ell_i
=
z_iW_{\mathrm{cls}}+b_{\mathrm{cls}},
\qquad
\widehat p_i
=
\mathrm{softmax}(\ell_i).
`

Max preserves coordinatewise extreme responses. Different coordinates can
select different time points, so the result need not be one coherent patch state.

## 4. Multitask S4MIL

If token outputs receive auxiliary labels:

`math
\ell_{it}^{\mathrm{patch}}
=
[Y_i]_{t,:}W_{\mathrm{patch}}+b_{\mathrm{patch}}.
`

For slide label y_i, patch labels q_it where available, and mask m_it:

`math
\mathcal{L}_i
=
\mathcal{L}_{\mathrm{slide}}
\left(y_i,\widehat p_i\right)
+
\lambda
\sum_{t=1}^{L_i}
m_{it}
\mathcal{L}_{\mathrm{patch}}
\left(q_{it},\widehat q_{it}\right).
`

The paper's CAMELYON16 multitask experiment maps token outputs back to slide
coordinates for localization. The auxiliary term shapes the state representation
before max readout.

## 5. C/R/G/S Placement

`text
C:
    structured S4D state-space convolution over ordered patch features

R:
    coordinatewise max pooling in the released model

G:
    patch extraction order and any grid/coordinate convention

S:
    slide label with optional patch-level auxiliary supervision
`
