# Prov-GigaPath: Tile-to-Slide Representation

References: [Prov-GigaPath paper](https://www.nature.com/articles/s41586-024-07441-w),
[official implementation](https://github.com/prov-gigapath/prov-gigapath).

Prov-GigaPath is a two-stage model. DINOv2 learns a tile encoder first; a
LongNet-based masked autoencoder then learns context over the tile sequence of
an entire slide. The slide object is therefore an ordered, coordinate-indexed
token sequence, not a bag passed directly to mean MIL.

## 1. Tile-Level Pretraining

The pretraining corpus contains `1,384,860,229` segmented `256` by `256`
tiles from `171,189` slides. For tile `x_j`, the DINOv2 tile encoder gives:

```math
h_j
=
f_{\phi}(x_j)
\in
\mathbb{R}^{1536}.
```

The tile stage treats each tile as an instance. Its view, teacher, centering,
and masked-token relations belong to DINOv2; they do not yet encode the
relative arrangement of different tiles from the same slide.

## 2. Slide Serialization

Let tile `j` have pixel coordinate:

```math
c_j
=
\left(
c_{j,0},c_{j,1}
\right)
\in
\mathbb{R}^{2}.
```

The paper serializes tiles in row-major order. The public implementation maps a
coordinate to a position in a fixed `1000` by `1000` grid with grid size
`d_grid=256`:

```math
r_j
=
\left\lfloor
\frac{c_j}{256}
\right\rfloor,
```

```math
p_j
=
1
+
1000r_{j,0}
+
r_{j,1}.
```

The leading `1` reserves position zero for the class token. A fixed 2D
sinusoidal table supplies the positional vector:

```math
e_j^{\mathrm{pos}}
=
E_{\mathrm{2D\mbox{-}sin}}
\left(
p_j
\right).
```

This makes coordinate convention part of the learned input contract. A
translation, flip, or different tessellation changes the position indices even
when the tile images are unchanged.

## 3. Slide Token Map

The implementation first projects tile embeddings into the LongNet width:

```math
u_j
=
W h_j+b+e_j^{\mathrm{pos}},
\qquad
W\in\mathbb{R}^{768\times1536},
\qquad
u_j\in\mathbb{R}^{768}.
```

Prepending a learned class token gives:

```math
U^{(0)}
=
\left[
u_{\mathrm{CLS}};
u_1;
\ldots;
u_n
\right].
```

The released `gigapath_slide_enc12l768d` configuration uses a 12-layer,
768-dimensional LongNet encoder. The output class token is the default pooled
slide representation:

```math
z_{\mathrm{slide}}
=
\mathrm{LN}
\left(
U_{\mathrm{CLS}}^{(L)}
\right).
```

The implementation can instead return normalized mean pooling over contextual
tile tokens, but that is a separate readout choice from the default class-token
path.

## 4. Dilated Attention Support

Dense self-attention over `n` tiles has pairwise cost:

```math
\Theta
\left(
n^{2}d
\right).
```

LongNet replaces the dense support with dilated segment supports. At dilation
level `\ell`, let `\delta_\ell` be the stride and `w_\ell` the number of
attended positions on either side. An idealized support is:

```math
\mathcal S_{\ell}(i)
=
\left\{
i+q\delta_\ell:
\left|q\right|\le w_\ell
\right\}
\cap
\left\{0,\ldots,n\right\}.
```

The total support is:

```math
\mathcal S(i)
=
\bigcup_{\ell=1}^{L}
\mathcal S_{\ell}(i).
```

The attention matrix is sparse:

```math
A_{ij}
=
0
\qquad
\text{when}
\qquad
j\notin\mathcal S(i).
```

With bounded local widths and logarithmically increasing dilation, the score
cost is linear up to the number of dilation levels:

```math
\mathrm{cost}
=
\mathcal O
\left(
n d
\sum_{\ell=1}^{L}w_\ell
\right).
```

This is the computational reason tens of thousands of tiles can be processed
without the `n^2` cost of a vanilla transformer. It is not equivalent to every
tile attending to every other tile with equal direct support.

## 5. Slide-Level Masked Autoencoding

Let `m_j` indicate a masked tile token:

```math
m_j
\in
\left\{0,1\right\}.
```

The slide encoder receives a corrupted sequence:

```math
\widetilde u_j
=
\left(
1-m_j
\right)u_j
+
m_j e_{\mathrm{MASK}}.
```

The LongNet encoder contextualizes the corrupted sequence and a slide-level
decoder predicts the original tile embeddings:

```math
\widehat h_j
=
D_{\psi}
\left(
U_j^{(L)}
\right).
```

The masked reconstruction objective is:

```math
\mathcal L_{\mathrm{slide\mbox{-}MAE}}
=
\frac{1}{\left|\mathcal M\right|}
\sum_{j\in\mathcal M}
\left|
\widehat h_j-h_j
\right|_2^2,
\qquad
\mathcal M
=
\left\{j:m_j=1\right\}.
```

The target is a learned tile representation, not raw slide pixels. The slide
encoder is rewarded for predicting tile-level visual content from the
remaining spatial context.

## 6. Slide Augmentation Contract

During slide pretraining, the input sequence is augmented by cropping the
coordinate grid, randomly translating the crop while keeping tiles inside the
grid, and horizontally flipping coordinates with probability `0.5`. The
reported crop ratio is `0.875`. Abstractly, a coordinate transform `T` acts as:

```math
\widetilde c_j
=
T(c_j),
\qquad
\widetilde U^{(0)}
=
\mathrm{Serialize}
\left(
\left\{
\left(h_j,\widetilde c_j\right)
\right\}_{j=1}^{n}
\right).
```

The model is therefore encouraged to use context stable under these coordinate
transformations. It is not guaranteed to be invariant to arbitrary spatial
translation, rotation, missing tissue, or retiling.

## 7. Downstream Softmax Readout

The paper uses the contextualized tile outputs in a downstream softmax
attention layer. For contextual tile states `q_j`:

```math
a_j
=
w_a^{\top}
\tanh
\left(
W_a q_j+b_a
\right),
```

```math
\alpha_j
=
\frac{\exp(a_j)}{\sum_{r=1}^{n}\exp(a_r)},
\qquad
z_{\mathrm{attn}}
=
\sum_{j=1}^{n}\alpha_j q_j.
```

The downstream head can then map `z_attn` to a task label. Consequently, the
full prediction map is:

```math
\left\{x_j,c_j\right\}_{j=1}^{n}
\longrightarrow
\left\{h_j\right\}_{j=1}^{n}
\longrightarrow
\left\{q_j\right\}_{j=1}^{n}
\longrightarrow
z_{\mathrm{attn}}
\longrightarrow
\widehat y.
```

The pretraining readout and downstream softmax readout are distinct. A strong
LongNet slide encoder does not imply that the final task uses the class token.

## 8. What Information Survives

Prov-GigaPath can preserve:

```text
local tile morphology from DINOv2;
coordinate-indexed spatial relations;
long-range sequence context through dilated attention;
tile content predictable from surrounding slide context.
```

It can lose information through tissue segmentation, coordinate quantization,
masking, sparse attention, and the final attention readout. In particular, two
slides can have similar tile marginals but different arrangements:

```math
\left\{h_j^{(A)}\right\}_{j=1}^{n}
=
\left\{h_j^{(B)}\right\}_{j=1}^{n}
```

while:

```math
\left\{c_j^{(A)}\right\}_{j=1}^{n}
\neq
\left\{c_j^{(B)}\right\}_{j=1}^{n}.
```

The slide encoder can distinguish them only to the extent that coordinate
positions and sparse attention expose that arrangement.

## C/R/G/S Placement

```text
C:
    DINOv2 tile encoder, linear tile projection, and LongNet dilated attention

R:
    class-token slide representation or downstream softmax attention over
    contextualized tile states

G:
    row-major sequence, quantized 1000-by-1000 coordinates, fixed 2D positions,
    and sparse dilated support

S:
    tile-level DINOv2 targets followed by masked reconstruction of slide tile
    embeddings
```

## Failure Matrix

| Failure | Mathematical source | Consequence |
|---|---|---|
| coordinate quantization | many coordinates share one grid index | distinct layouts can collide |
| serialization bias | 2D slide becomes a 1D sequence | row/column relations are anisotropic |
| sparse support | `j` outside `S(i)` has zero direct attention | some long-range pairs interact only through paths |
| mask ambiguity | multiple tiles fit the same context | reconstruction averages rare morphology |
| tile filtering | background and low-occupancy tiles are removed | diagnostic evidence in discarded regions is absent |
| readout mismatch | class token versus softmax attention differs | pretraining representation is not the task statistic |
| cohort shortcut | Providence acquisition and tissue mixture | transfer can fail under scanner or institution shift |
