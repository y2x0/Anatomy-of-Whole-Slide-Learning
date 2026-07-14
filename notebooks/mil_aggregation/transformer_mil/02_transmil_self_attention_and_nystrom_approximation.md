# TransMIL: Correlated MIL, Nystrom Attention, And PPEG

Primary source: [Shao et al., TransMIL](https://arxiv.org/abs/2106.00908).
Implementation reference: [TransMIL code](https://github.com/szc19990412/TransMIL).

TransMIL is not ordinary attention pooling. Its source-level forward map
projects patch features, pads them into a square, prepends a class token,
applies a first Nyström attention layer, injects PPEG spatial context, applies
a second Nyström layer, and reads the final class token.

```math
H_i
\xrightarrow{\ \mathrm{Linear}+\mathrm{ReLU}\ }
\widetilde H_i
\xrightarrow{\ \mathrm{square\ padding}+\mathrm{CLS}\ }
H_i^{(0)}
\xrightarrow{\ \mathrm{NysAttn}_1\ }
H_i^{(1)}
\xrightarrow{\ \mathrm{PPEG}\ }
H_i^{(P)}
\xrightarrow{\ \mathrm{NysAttn}_2\ }
H_i^{(2)}
\xrightarrow{\ \mathrm{CLS}\ }
z_i
\xrightarrow{\ W_{\mathrm{out}}\ }
\widehat y_i.
```

## 1. Correlated Bag Object

For WSI i, let:

```math
\mathcal B_i
=
\left\{
h_{ij}
:
j=1,\ldots,n_i
\right\},
\qquad
h_{ij}\in\mathbb R^{1024}.
```

The source experiments use 256 by 256 tissue patches at 20x and discard
background. The number of patches is variable; the source paper reports a mean
of about 14,627 for TCGA slides and about 8,800 for CAMELYON16 after
preprocessing.

An iid-style MIL abstraction suppresses pairwise structure:

```math
P(Y_i\mid h_{i1},\ldots,h_{in_i})
\approx
\mathcal R
\left(
\left\{
h_{ij}
\right\}_{j=1}^{n_i}
\right).
```

TransMIL introduces a context map before the final readout:

```math
P(Y_i\mid h_{i1},\ldots,h_{in_i})
=
\mathcal H
\left(
\mathcal R
\left(
\mathcal C_i
\left(
\left\{
h_{ij}
\right\}_{j=1}^{n_i}
\right)
\right)
\right).
```

The intended hypothesis is that tumor-stroma, tumor-immune, and repeated tissue
patterns are informative through their relations, not only through their
marginal patch frequencies.

## 2. Input Projection

The official model maps 1024-dimensional patch features to 512-dimensional
tokens:

```math
\widetilde h_{ij}
=
\rho
\left(
W_{\mathrm{in}}h_{ij}
+
b_{\mathrm{in}}
\right)
\in
\mathbb R^{512},
```

with:

```math
W_{\mathrm{in}}\in\mathbb R^{512\times1024},
\qquad
\rho=\mathrm{ReLU}.
```

In matrix form:

```math
\widetilde H_i
=
\rho
\left(
H_iW_{\mathrm{in}}^{\mathsf T}
+
\mathbf 1b_{\mathrm{in}}^{\mathsf T}
\right)
\in
\mathbb R^{n_i\times512}.
```

The projection is a geometry bottleneck. If:

```math
W_{\mathrm{in}}h
=
W_{\mathrm{in}}h',
```

then the Transformer receives the same pre-activation for h and h'. A
task-relevant distinction lost here cannot be recovered by PPEG or attention.

## 3. Square Padding And Class Token

PPEG requires a square grid. Define:

```math
N_i
=
\left\lceil\sqrt{n_i}\right\rceil,
\qquad
\delta_i
=
N_i^2-n_i.
```

The official implementation pads by repeating the first delta_i tokens:

```math
\widetilde H_i^{\mathrm{pad}}
=
\left[
\widetilde H_i;
\widetilde H_{i,1:\delta_i}
\right]
\in
\mathbb R^{N_i^2\times512}.
```

This is not zero padding. A real patch can appear twice in the Transformer
sequence. With learned class token c:

```math
c\in\mathbb R^{512},
\qquad
H_i^{(0)}
=
\left[
c;
\widetilde H_i^{\mathrm{pad}}
\right]
\in
\mathbb R^{(N_i^2+1)\times512}.
```

The padding rule is part of the estimator. If the repeated token receives
attention at both positions, its effective multiplicity increases:

```math
\mathrm{credit}(h_{i1})
\approx
\mathrm{credit}(\mathrm{position}\ 1)
+
\mathrm{credit}(\mathrm{position}\ N_i^2).
```

## 4. Exact Self-Attention

For X in R^(L by d):

```math
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
```

One head forms:

```math
A
=
\mathrm{softmax}_{\mathrm{row}}
\left(
\frac{QK^{\mathsf T}}{\sqrt{d_h}}
\right)
\in\mathbb R^{L\times L},
\qquad
O=AV.
```

The class-token row is:

```math
o_{\mathrm{CLS}}
=
\sum_{\ell=0}^{L-1}
A_{0\ell}v_\ell.
```

The off-diagonal entries allow one patch to affect another before the final
readout. This is the difference between relational context and a context-free
weighted first moment:

```math
z_i^{\mathrm{ABMIL}}
=
\sum_j\alpha_{ij}h_{ij},
\qquad
z_i^{\mathrm{TransMIL}}
=
\mathrm{ReadCLS}
\left(
\mathcal C_i(H_i)
\right).
```

The official implementation uses two pre-layer-normalized attention layers with
residual addition and attention dropout 0.1.

## 5. Nyström Attention

Let L=N_i squared plus one be the padded sequence length and let m landmarks be
selected from the query and key sequences:

```math
\widetilde Q\in\mathbb R^{m\times d_h},
\qquad
\widetilde K\in\mathbb R^{m\times d_h}.
```

The three softmax factors are:

```math
A_{L,m}
=
\mathrm{softmax}_{\mathrm{row}}
\left(
\frac{Q\widetilde K^{\mathsf T}}{\sqrt{d_h}}
\right),
```

```math
A_{m,m}
=
\mathrm{softmax}_{\mathrm{row}}
\left(
\frac{\widetilde Q\widetilde K^{\mathsf T}}{\sqrt{d_h}}
\right),
```

```math
A_{m,L}
=
\mathrm{softmax}_{\mathrm{row}}
\left(
\frac{\widetilde QK^{\mathsf T}}{\sqrt{d_h}}
\right).
```

TransMIL uses the Nyström approximation:

```math
\widehat A
=
A_{L,m}
A_{m,m}^{+}
A_{m,L},
```

where plus is the Moore-Penrose pseudoinverse. The contextual output is:

```math
\widehat O
=
\widehat A V.
```

The softmax factors matter. The source method is not adequately represented by
an unnormalized raw-kernel factorization alone.

The official implementation uses:

```math
d=512,
\qquad
h=8,
\qquad
d_h=64,
\qquad
m=d/2=256,
\qquad
\text{six pseudoinverse iterations}.
```

Thus the landmark count is fixed by the model width in the released code, not
by each slide's number of patches.

## 6. Complexity And Approximation Error

Exact attention costs:

```math
\mathcal C_{\mathrm{exact}}
\propto
L^2d.
```

The Nyström factors cost:

```math
\mathcal C_{\mathrm{Nys}}
\propto
Lmd+m^2d,
```

for a fixed number of pseudoinverse iterations. When m and d are fixed:

```math
\mathcal C_{\mathrm{Nys}}
=
\mathcal O(L)
\quad
\text{as a function of }L.
```

The exact statement is not universally O(L): if m grows with L, the scaling
changes. The memory changes from an L by L matrix to factors of sizes L by m,
m by m, and m by L.

The contextual perturbation is:

```math
\widehat O-O
=
\left(
\widehat A-A
\right)V,
```

so:

```math
\left\|
\widehat O-O
\right\|_F
\le
\left\|
\widehat A-A
\right\|_2
\left\|
V
\right\|_F.
```

Small landmark count does not imply small task error:

```math
m\ll L
\not\Longrightarrow
\left\|
\widehat A-A
\right\|_2
\text{ small}.
```

Landmark coverage and the spectrum of the attention operator determine the
approximation quality.

## 7. PPEG Geometry

After the first contextual layer:

```math
H_i^{(1)}
=
\left[
c_i^{(1)};
F_i^{(1)}
\right],
\qquad
F_i^{(1)}
\in
\mathbb R^{N_i^2\times512}.
```

Reshape patch tokens:

```math
G_i^{(1)}
\in
\mathbb R^{512\times N_i\times N_i}.
```

The released PPEG module uses depthwise 7 by 7, 5 by 5, and 3 by 3
convolutions:

```math
P_i
=
G_i^{(1)}
+
\mathrm{DWConv}_{7}(G_i^{(1)})
+
\mathrm{DWConv}_{5}(G_i^{(1)})
+
\mathrm{DWConv}_{3}(G_i^{(1)}).
```

Each channel has its own spatial kernels. Flatten and restore the class token:

```math
H_i^{(P)}
=
\left[
c_i^{(1)};
\mathrm{Flatten}(P_i)
\right]
\in
\mathbb R^{(N_i^2+1)\times512}.
```

PPEG is conditional because it acts on the current token features. It is
spatial because sequence order determines grid placement:

```math
\mathrm{PPEG}
\left(
\mathrm{reshape}(\Pi F)
\right)
\neq
\Pi\,
\mathrm{PPEG}
\left(
\mathrm{reshape}(F)
\right)
```

for a general token permutation Pi.

PPEG is therefore not a neutral positional label. It adds local convolutional
context and can make neighboring sequence positions interact.

## 8. Complete Source-Level Forward Map

The first layer is:

```math
H_i^{(1)}
=
H_i^{(0)}
+
\mathrm{NysAttn}_1
\left(
\mathrm{LN}(H_i^{(0)})
\right).
```

Then:

```math
H_i^{(P)}
=
\mathrm{PPEG}
\left(
H_i^{(1)};N_i,N_i
\right).
```

The second layer is:

```math
H_i^{(2)}
=
H_i^{(P)}
+
\mathrm{NysAttn}_2
\left(
\mathrm{LN}(H_i^{(P)})
\right).
```

The final representation is:

```math
z_i
=
\mathrm{LN}
\left(
H_i^{(2)}[0,:]
\right)
\in
\mathbb R^{512}.
```

For C classes:

```math
\ell_i
=
W_{\mathrm{out}}z_i+b_{\mathrm{out}}
\in
\mathbb R^{C},
\qquad
\widehat p_i
=
\mathrm{softmax}(\ell_i).
```

There is no final mean or max pooling in this implementation. The class token
is the readout query.

## 9. Correlation And Credit

Ignoring nonlinearities for one linearized pass, the class-token path contains:

```math
H_i^{(2)}
\approx
\widehat A_i^{(2)}
\left[
\mathrm{PPEG}
\left(
\widehat A_i^{(1)}H_i^{(0)}
\right)
\right].
```

Thus a patch can affect the final class token through:

```math
\text{input projection}
\longrightarrow
\widehat A_i^{(1)}
\longrightarrow
\mathrm{PPEG}
\longrightarrow
\widehat A_i^{(2)}
\longrightarrow
\mathrm{CLS}.
```

The final attention row is only one factor:

```math
\mathrm{attention}_{0j}^{(2)}
\neq
\frac{\partial\ell_i}{\partial h_{ij}}
\neq
\ell_i(H_i)
-
\ell_i(H_i\setminus h_{ij}).
```

Deletion must recompute padding, both approximated attention layers, PPEG, and
the classifier.

## 10. Permutation And Layout

Without PPEG, attention without positional input is permutation equivariant:

```math
\mathrm{NysAttn}(\Pi H)
=
\Pi\,
\mathrm{NysAttn}(H).
```

The class-token readout is then invariant to patch permutation under the
idealized implementation:

```math
\mathrm{ReadCLS}
\left(
\mathrm{NysAttn}(\Pi H)
\right)
=
\mathrm{ReadCLS}
\left(
\mathrm{NysAttn}(H)
\right).
```

PPEG breaks that invariance:

```math
\mathrm{TransMIL}(\Pi H)
\neq
\mathrm{TransMIL}(H)
```

for general Pi. A nonzero response proves order dependence, not correct
physical geometry. If preprocessing order is not coordinate-preserving, PPEG
creates a false neighborhood.

## 11. Failure Modes

### 11.1 Repeated Padding Bias

When delta_i is positive:

```math
\delta_i>0
\Longrightarrow
\text{the first }\delta_i\text{ real tokens are duplicated}.
```

This changes effective multiplicity and can bias early sequence positions.

### 11.2 False Spatial Geometry

PPEG assumes sequence order can be placed on a square grid:

```math
\text{sequence adjacency}
\approx
\text{physical WSI adjacency}.
```

If the order is arbitrary, the convolutional neighborhoods are artificial.

### 11.3 Nyström Support Failure

Rare positive directions can be poorly represented by the selected landmarks:

```math
\widehat A_{0j}
\approx
0
\quad\text{while}\quad
A_{0j}>0.
```

Many weak long-range edges can jointly matter even when each individual edge is
small.

### 11.4 Class-Token Collision

Different contextualized bags can map to the same slide vector:

```math
H_i^{(2)}
\neq
H_{i'}^{(2)}
\quad\text{but}\quad
z_i=z_{i'}.
```

Global attention does not make the final 512-dimensional representation
lossless.

### 11.5 Approximation-Dependent Decision

Even small operator error can change a classifier:

```math
\left\|
\widehat A-A
\right\|_2
\text{ small}
\not\Longrightarrow
\arg\max_c\widehat\ell_{ic}
=
\arg\max_c\ell_{ic}.
```

The relevant perturbation passes through the values, PPEG, second layer,
normalization, and classifier.

### 11.6 Attention Explanation Error

Attention is routing, not signed task credit. A positive attention weight can
coexist with negative deletion effect because changing one token changes
normalization and all contextual states.

## 12. Source Ablation And Sanity Checks

The source paper reports an ablation in which PPEG outperforms no positional
encoding and sinusoidal alternatives on the reported tasks. On CAMELYON16, the
reported AUC values include:

```math
\mathrm{AUC}_{\mathrm{no\ position}}
=
0.8416,
\qquad
\mathrm{AUC}_{\mathrm{PPEG}}
=
0.9309.
```

This supports the usefulness of the conditional spatial operator in that
benchmark. It does not prove that every sequence order represents physical
coordinates.

### Layout Swap

Hold the patch multiset fixed and change only the order:

```math
\Delta_{\mathrm{layout}}
=
\left\|
\mathrm{TransMIL}(H_i)
-
\mathrm{TransMIL}(\Pi H_i)
\right\|_2.
```

Compare arbitrary permutations against physically meaningful coordinate swaps.

### Padding Swap

Compare repeated, zero, masked, and learned padding:

```math
\Delta_{\mathrm{pad}}
=
\left\|
\mathrm{TransMIL}_{\mathrm{repeat}}(H_i)
-
\mathrm{TransMIL}_{\mathrm{alternative}}(H_i)
\right\|_2.
```

This isolates the estimator's dependence on the padding statistic.

### Landmark Perturbation

Vary landmark selection or seed:

```math
\Delta_{\mathrm{landmark}}
=
\mathbb E_r
\left[
\left\|
\widehat A^{(r)}H_i
-
\overline{\widehat A}H_i
\right\|_F
\right].
```

This measures approximation-induced representation variance.

### Recomputed Deletion

For target logit l_i:

```math
\Delta_{ij}
=
\ell_i(H_i)
-
\ell_i
\left(
\mathrm{recompute}
\left(
H_i\setminus h_{ij}
\right)
\right).
```

Removing a token requires rebuilding the square padding and grid. Masking its
already computed attention row is not equivalent.

## 13. C/R/G/S Placement

| Component | TransMIL |
|---|---|
| C context operator | two Nyström-approximated multi-head self-attention layers with residual paths |
| R readout operator | final normalized class token and linear classifier |
| G geometry | square padding, sequence-to-grid mapping, and depthwise 7/5/3 PPEG fusion |
| S surviving statistic | pairwise contextual information compressed into a 512-dimensional class token |

The padding rule belongs to geometry and preprocessing. It changes the token
multiplicities presented to the context operator.

The distinction from ABMIL is:

```math
\mathrm{ABMIL}
\longrightarrow
\text{weighted first moment of independent patch states},
```

whereas:

```math
\mathrm{TransMIL}
\longrightarrow
\text{class-token statistic after approximate pairwise context and grid fusion}.
```

## 14. Bottom Line

The source-level TransMIL map is:

```math
\mathbb R^{n_i\times1024}
\xrightarrow{\ W_{\mathrm{in}}+\mathrm{ReLU}\ }
\mathbb R^{n_i\times512}
\xrightarrow{\ \mathrm{repeat\ square\ padding}\ }
\mathbb R^{(N_i^2+1)\times512}
\xrightarrow{\ \mathrm{NysAttn}_1\ }
\xrightarrow{\ \mathrm{PPEG}_{7,5,3}\ }
\xrightarrow{\ \mathrm{NysAttn}_2\ }
\mathbb R^{(N_i^2+1)\times512}
\xrightarrow{\ \mathrm{CLS}\ }
\mathbb R^{512}
\xrightarrow{\ W_{\mathrm{out}}\ }
\mathbb R^{C}.
```

Its mathematical shape is:

```math
\text{correlated instance context}
+
\text{Nyström support approximation}
+
\text{conditional grid geometry}
+
\text{class-token readout}.
```

The key surviving statistic is a learned relational summary. The key failure
question is whether the relation came from physical WSI geometry or from
preprocessing order and repeated padding.
