# PPEG Positional Encoding and WSI Geometry

## 1. Scope and source contract

This note isolates one question in TransMIL:

> What spatial hypothesis is introduced when a bag of patch tokens is reshaped
> into a square grid and processed by PPEG?

The answer is narrower than the slogan "the transformer uses position." The
official TransMIL implementation does not attach a learned absolute vector to
each original WSI coordinate. It performs four concrete operations:

1. It pads the number of patch tokens to a square by repeating existing tokens.
2. It removes the class token before the spatial reshape.
3. It applies three depthwise convolutions with kernels 7, 5, and 3, together
   with an identity residual.
4. It flattens the resulting grid and prepends the class token again.

This distinction matters. PPEG is a learned local operator on an imposed raster
and not a complete coordinate model. It can make nearby tokens interact, but it
does not by itself know the physical distance between two patches, the tissue
mask, the slide boundary, or the magnification at which the coordinates were
measured.

The paper-level anchor is Shao et al., TransMIL: Transformer based Correlated
Multiple Instance Learning for Whole Slide Image Classification:

https://arxiv.org/abs/2106.00908

The equations below follow the public implementation's forward map. The
implementation details are important because a mathematically elegant zero-pad
model would describe a different network from the released model.

## 2. Bag notation and the two coordinate spaces

Let slide (i) provide a variable-size bag of patch features. The feature
encoder has already converted each selected patch into a vector.

```math
X_i
=
\begin{bmatrix}
x_{i1}^{\mathsf T}\\
\vdots\\
x_{in_i}^{\mathsf T}
\end{bmatrix}
\in
\mathbb R^{n_i\times d_0}.
```

Each patch also has a physical or extraction coordinate. Write it as a two
dimensional point in slide coordinates.

```math
c_{ij}=(u_{ij},v_{ij})\in\mathbb R^2,
\qquad
C_i=(c_{i1},\ldots,c_{in_i}).
```

There are two distinct maps that are often conflated:

```math
\text{physical coordinates } C_i
\xrightarrow{\;\mathcal R_i\;}
\text{raster positions } G_i
\xrightarrow{\;\mathcal P_i\;}
\text{ordered token matrix } X_i.
```

The first map is a data and preprocessing decision. It bins or sorts physical
patches into a grid. The second map is the tensor order presented to the model.
PPEG only sees the grid tensor after these choices have been made. If two
patches are adjacent in tensor order but far apart in the WSI, the convolution
will treat them as neighbors anyway.

For a true regular patch lattice with stride (s), a valid raster map can be
written as

```math
g_{ij}
=
\left(
\frac{u_{ij}-u_0}{s},
\frac{v_{ij}-v_0}{s}
\right)
\in\mathbb Z^2.
```

For a bag produced by tissue filtering, however, the occupied coordinates are
usually a strict subset of a rectangular lattice. The model must then decide
whether empty lattice cells are deleted, represented by a mask, or silently
replaced by another token. TransMIL's square padding handles the number of
tokens, not arbitrary missing physical cells.

## 3. The exact TransMIL token map

### 3.1 Feature projection

In the released implementation, the input feature width is 1024 and the
transformer width is 512. The first map is a learned affine projection followed
by a ReLU.

```math
H_i^{\mathrm{in}}
=
\mathrm{ReLU}\!\left(X_iW_{\mathrm{in}}^{\mathsf T}
+\mathbf 1_{n_i}b_{\mathrm{in}}^{\mathsf T}\right)
\in\mathbb R^{n_i\times d},
\qquad
d_0=1024,
\quad d=512.
```

This projection is not positional. It changes the feature geometry before any
spatial operation. A position effect observed after PPEG can therefore be
entangled with the learned channel projection.

### 3.2 Square completion

The implementation chooses the smallest square that can hold the bag.

```math
N_i
=
\left\lceil\sqrt{n_i}\right\rceil,
\qquad
L_i=N_i^2,
\qquad
\delta_i=L_i-n_i.
```

If (delta_i=0), no completion is needed. If (delta_i>0), the released
code repeats the first (delta_i) real patch tokens. Let the completion map
be

```math
\kappa_i(r)
=
\begin{cases}
r, & 1\le r\le n_i,\\
r-n_i, & n_i<r\le L_i,
\end{cases}
```

where the second case is used only for the small remainder in the official
implementation. In the common case (delta_i<n_i), this gives

```math
\widetilde h_{ir}
=
h_{i,\kappa_i(r)}
\qquad
1\le r\le L_i.
```

Equivalently, define a binary completion matrix (D_i) with one selected input
column in every output row.

```math
D_i\in\{0,1\}^{L_i\times n_i},
\qquad
(D_i)_{rj}=\mathbf 1\{j=\kappa_i(r)\},
\qquad
\widetilde H_i=D_iH_i^{\mathrm{in}}.
```

Every row of (D_i) contains one 1, but some columns can contain more than one
1. Thus completion preserves neither the empirical measure nor the unique
patch count. It changes the bag before PPEG by assigning extra multiplicity to
the first tokens.

This is not the same as zero-padding:

```math
D_iH_i^{\mathrm{in}}
\ne
\begin{bmatrix}H_i^{\mathrm{in}}\\0_{\delta_i\times d}\end{bmatrix}
\quad\text{in general}.
```

The distinction is consequential because attention and convolution are both
nonlinear downstream. A repeated tissue token can receive attention, be mixed
into neighboring cells, and be counted again by the class-token readout.

### 3.3 Class-token separation

Let q denote the learned class token. The transformer sequence has length
L_i+1, but PPEG acts only on the L_i patch positions.

```math
Z_i^0
=
\begin{bmatrix}
q^{\mathsf T}\\
\widetilde H_i
\end{bmatrix}
\in\mathbb R^{(L_i+1)\times d}.
```

For the PPEG submodule, write

```math
U_i^0
=
Z_i^0[2:(L_i+1),:]
\in\mathbb R^{L_i\times d}.
```

The class token is therefore not placed in a grid cell. It does not receive a
convolutional neighborhood. Its access to local spatial information occurs by
transformer attention before or after the PPEG operation, depending on the
TransLayer in which PPEG is inserted.

## 4. Grid reshape as an operator

To make the geometry explicit, define a row-major reshape operator

```math
\mathcal R_{N_i}:
\mathbb R^{L_i\times d}
\longrightarrow
\mathbb R^{N_i\times N_i\times d}.
```

For row (a), column (b), and channel (r),

```math
\bigl[\mathcal R_{N_i}(U)\bigr]_{a,b,r}
=
U_{(a-1)N_i+b,r}.
```

The inverse flattening operator is

```math
\bigl[\mathcal F_{N_i}(V)\bigr]_{(a-1)N_i+b,r}
=
V_{a,b,r}.
```

The composition is exact on a full square:

```math
\mathcal F_{N_i}\!\left(\mathcal R_{N_i}(U)\right)=U.
```

The nontrivial geometric assumption is not the reshape itself. It is the claim
that row-major neighbors in this tensor correspond to meaningful WSI
neighbors. If the token list was created by an arbitrary sampler, then
(mathcal R_{N_i}) imposes a synthetic topology.

A physical-coordinate-aware rasterizer would instead choose a map
(ho_i) satisfying, as far as possible,

```math
\|g_{ij}-g_{ik}\|_2
\approx
\frac{1}{s}\|c_{ij}-c_{ik}\|_2
```

for occupied patches (j,k). The released TransMIL model does not learn or
verify this relation. It assumes that the supplied feature order already has a
useful spatial interpretation.

## 5. The PPEG operator

### 5.1 Channelwise local convolution

After reshape, transpose channels into the convolution layout. Let

```math
V_i
=
\mathcal T\!\left(\mathcal R_{N_i}(U_i^0)\right)
\in\mathbb R^{d\times N_i\times N_i},
```

where the transpose map exchanges the last feature axis with the first axis.

For a depthwise kernel with spatial support K x K, each channel has its own
kernel and bias. With zero boundary padding,

```math
\bigl[\mathrm{DWConv}_{K}(V)\bigr]_{r,a,b}
=
b^{(K)}_r
+
\sum_{\substack{u,v\in\mathbb Z\\|u|,|v|\le (K-1)/2}}
W^{(K)}_{r,u,v}
V_{r,a-u,b-v},
```

where values outside the square grid are zero.

Depthwise means that the convolution does not mix feature channels:

```math
\mathrm{DWConv}_{K}
=
\mathrm{diag}\!\left(
\mathrm{Conv}_{K,1},\ldots,\mathrm{Conv}_{K,d}
\right).
```

Cross-channel mixing is supplied by the transformer projections and the input
projection, not by PPEG itself.

### 5.2 The released three-scale residual

The PPEG map used by TransMIL is

```math
\mathrm{PPEG}(V)
=
V
+\mathrm{DWConv}_{7}(V)
+\mathrm{DWConv}_{5}(V)
+\mathrm{DWConv}_{3}(V).
```

Flattening back into token layout gives

```math
U_i^{\mathrm{ppeg}}
=
\mathcal F_{N_i}\!\left(
\mathcal T^{-1}\!\left(\mathrm{PPEG}(V_i)\right)
\right)
\in\mathbb R^{L_i\times d}.
```

The full sequence after PPEG is

```math
Z_i^{\mathrm{ppeg}}
=
\begin{bmatrix}
q_i^{\mathrm{ctx}}\\
U_i^{\mathrm{ppeg}}
\end{bmatrix},
```

where q_i^{ctx} is the class-token row supplied by the surrounding
transformer layer. The PPEG map itself acts on patch rows only; the class token
is concatenated back afterward.

### 5.3 Effective local stencil

Ignoring boundary truncation and the nonlinear transformer between PPEG blocks,
one PPEG application gives a channelwise linear stencil with offsets in the
union of the three square supports.

```math
\mathcal N_{7}
=
\{(u,v):|u|\le 3,\ |v|\le 3\}.
```

The 5 and 3 supports are subsets of the 7 support, but they have independent
parameters. The operator can therefore express a broad local filter plus
separate medium- and short-range corrections. It is not equivalent to one
ordinary 7 by 7 depthwise convolution unless the three kernels are tied in a
particular way.

The identity term also gives a direct path for every patch feature:

```math
\frac{\partial U_{i,r,a,b}^{\mathrm{ppeg}}}
{\partial U_{i,r,a,b}^{0}}
\supseteq 1.
```

This residual does not guarantee stable optimization, because the convolutional
terms can still amplify or cancel the identity path. It does guarantee that
the parameterization contains the no-convolution map at zero convolution
weights.

## 6. What PPEG changes mathematically

### 6.1 The permutation action

For an unordered bag, a permutation matrix (P) acts on rows:

```math
H_i
\longmapsto
PH_i,
\qquad
P\in\{0,1\}^{n_i\times n_i}.
```

The input features are the same multiset after this action. A set encoder must
satisfy invariance at the final readout:

```math
f(PH_i)=f(H_i).
```

Self-attention without positional information is row-permutation equivariant.
PPEG is different because the permutation is applied to tokens while the grid
coordinates remain fixed. Let (Gamma_i) denote the fixed raster assignment.
Then, in general,

```math
\mathrm{PPEG}_{\Gamma_i}(\mathcal R(PH_i))
\ne
P\,\mathrm{PPEG}_{\Gamma_i}(\mathcal R(H_i)).
```

The right side permutes outputs after the spatial operation. The left side
changes which feature occupies each convolutional neighborhood. This is the
precise sense in which PPEG breaks arbitrary set symmetry.

### 6.2 Conditional equivariance, not absolute coordinates

PPEG has a weaker and more useful symmetry: for translations that remain inside
the grid and ignore boundary changes, a shared convolution commutes with the
translation operator.

```math
\mathrm{DWConv}_{K}(\tau_{a,b}V)
\approx
\tau_{a,b}\mathrm{DWConv}_{K}(V).
```

The approximation sign is necessary. Finite zero padding makes the boundary a
special location, so exact translation equivariance fails at the edges. The
three-scale sum inherits the same boundary dependence.

There is no exact invariance to arbitrary rotations, reflections, irregular
scales, or permutations unless those transformations are explicitly represented
in the data or learned through augmentation.

### 6.3 Locality is injected before global mixing

The convolution creates a local context operator. Self-attention then provides a
global interaction operator. For one patch row (r), the causal dependency set
after a PPEG block contains a local neighborhood even if the attention matrix
were later pruned.

```math
\mathrm{Dep}(u_{i,r}^{\mathrm{ppeg}})
\supseteq
\{u_{i,s}^{0}:s\text{ lies in the }7\times7\text{ raster neighborhood of }r\}.
```

Conversely, a transformer layer can connect distant tokens even when their
grid distance is large. PPEG is therefore a local inductive bias, not a
replacement for global attention.

If PPEG appears in two layers separated by a transformer block, the local
operator is applied to a feature that already contains global information. A
convolutional neighborhood at the second application can consequently contain
information from distant patches, despite the convolution's finite spatial
stencil.

## 7. PPEG versus standard positional encodings

### 7.1 Learned absolute vectors

A standard absolute table assigns a vector (e_{a,b}) to each grid location.

```math
U_{a,b}
\longmapsto
U_{a,b}+e_{a,b}.
```

This distinguishes locations directly, but a table depends on a fixed grid
size and is awkward for variable-size WSI bags. PPEG shares its local kernels
across positions and can be applied to different square sizes. It gains this
flexibility by encoding relative local structure rather than a unique learned
identity for every coordinate.

### 7.2 Relative attention bias

A relative bias modifies a pairwise attention logit according to a coordinate
difference.

```math
\ell_{ab}
=
\frac{q_a^{\mathsf T}k_b}{\sqrt d}
+\beta(g_a-g_b).
```

PPEG does not directly modify a pairwise logit. It first changes the token
values with a channelwise spatial filter, after which attention operates on the
changed values and queries. The two mechanisms can represent different
functions even if both use the same grid.

### 7.3 Coordinate concatenation

Another option is to append normalized coordinates to the feature:

```math
\bar h_{a,b}
=
\begin{bmatrix}
h_{a,b}\\
u_{a,b}/W\\
v_{a,b}/H
\end{bmatrix}.
```

This exposes absolute location to a learned projection but does not itself
create a local neighborhood. PPEG and coordinate concatenation therefore have
different inductive biases: one is local and translation-shared, the other is
absolute and pointwise.

## 8. The physical WSI interpretation

### 8.1 When the raster is meaningful

Suppose patches were extracted from a regular lattice with fixed patch width
(p) and stride (s). If the sampler preserves row and column indices, then a
3 by 3 or 7 by 7 neighborhood approximates a physical neighborhood of radius
proportional to (s).

```math
\mathcal N_{K}(j)
=
\left\{
k:
\left|u_k-u_j\right|\le \frac{K-1}{2}s,
\quad
\left|v_k-v_j\right|\le \frac{K-1}{2}s
\right\}.
```

The approximation becomes poor when the sampler removes many cells, when
patches overlap irregularly, or when rows are sorted by a key unrelated to
coordinates.

### 8.2 Tissue filtering is not geometric masking

A tissue filter produces an occupancy indicator (m_{a,b}).

```math
m_{a,b}
=
\mathbf 1\{\text{lattice cell }(a,b)\text{ is retained}\}.
```

The ideal masked local operator would condition every convolution and attention
step on (m). PPEG as implemented does not take a tissue mask as an explicit
argument. An absent cell is therefore not automatically treated as an absent
cell. If the preprocessing deletes it and compacts the remaining tokens, its
neighbors can become adjacent in the imposed grid even when they were separated
by a large physical gap.

This creates two distinct representations:

```math
\text{compact sequence}
\quad\text{versus}\quad
\text{occupancy-preserving grid}.
```

They have the same retained patch features but different PPEG outputs. Any
claim that PPEG "uses WSI geometry" must specify which representation was
actually supplied.

### 8.3 Magnification and units

Coordinates measured in pixels at one magnification do not have the same metric
as coordinates measured at another magnification. If the patch stride is (s_1)
at level 1 and (s_2) at level 2, then a fixed kernel radius represents

```math
r_1=\frac{K-1}{2}s_1
\qquad\text{or}\qquad
r_2=\frac{K-1}{2}s_2
```

in different physical units. PPEG has no built-in scale normalization. A model
trained with one extraction protocol can therefore learn a spatial prior tied
to that protocol rather than to a magnification-invariant tissue scale.

## 9. Boundary conditions and padding artifacts

### 9.1 Zero boundaries

For a location near a grid edge, some convolutional taps read zeros. Let
(mathcal N_K(a,b)) be the nominal neighborhood and (mathcal I_i) the set of
valid grid cells. The effective operator is

```math
\bigl[\mathrm{DWConv}_{K}(V)\bigr]_{r,a,b}
=
b_r^{(K)}
+
\sum_{(u,v)\in\mathcal N_K(a,b)\cap\mathcal I_i}
W^{(K)}_{r,u,v}V_{r,a-u,b-v}.
```

The number and pattern of contributing cells depend on location. Thus the
network can infer whether a token lies near an artificial edge even if edge
location has no biological meaning.

### 9.2 Square completion boundary

The outer square boundary is also determined by bag cardinality. Two slides
with similar tissue but different n_i can receive different
padding and different boundary geometry. The network can then use bag size
through the shape of the completed grid, in addition to any explicit attention
or normalization dependence on (n_i).

### 9.3 Repeated-token boundary

The repeated completion tokens are placed at the end of the row-major sequence
by the official code. They occupy the final cells of the square and can form a
contiguous strip or corner, depending on (delta_i). Define the duplicate set

```math
\mathcal D_i
=
\{r:n_i<r\le L_i\}.
```

Then a PPEG neighborhood intersecting (mathcal D_i) receives a duplicated
feature rather than a missing value. This can create an artificial local region
whose signal is correlated with the first patches in the bag.

## 10. Two counterexamples

### 10.1 Same multiset, different raster

Take four scalar patch features and a two by two grid.

```math
U
=
\begin{bmatrix}
a\\b\\c\\d
\end{bmatrix},
\qquad
U'
=
\begin{bmatrix}
a\\c\\b\\d
\end{bmatrix}.
```

The bags have the same empirical measure. Under a nonconstant depthwise
convolution, the top-right and bottom-left neighborhoods see different values,
so generally

```math
\mathrm{PPEG}(\mathcal R_2(U))
\ne
\mathrm{PPEG}(\mathcal R_2(U')).
```

This is desirable if the two arrangements reflect different tissue geometry.
It is harmful if the ordering difference is only a serialization accident.

### 10.2 Same tissue, different compacting rule

Consider a three by three physical lattice with the center cell removed. A
geometry-preserving representation keeps eight occupied coordinates and a
missing center. A compact sequence removes the center and then fills a square
of side three by repeating one token. The retained feature multiset is the same,
but the local stencil differs at every location near the gap.

```math
\{(a,b):m_{a,b}=1\}
\quad\text{is fixed, but}\quad
\mathcal R_{3}\circ\mathcal P
\quad\text{changes with the compaction rule}.
```

The correct conclusion is not that PPEG is invalid. It is that its geometry is
the geometry of the supplied raster, and preprocessing is part of the model.

## 11. Relation to TransMIL's reported ablation

The TransMIL paper reports a CAMELYON16 comparison in which removing PPEG
reduces slide-level performance relative to the full model. The reported AUCs
are approximately 0.8416 without PPEG and 0.9309 with PPEG in the ablation
table.

This result supports the claim that the spatial module matters in that training
and evaluation protocol. It does not identify which mechanism caused the gain.
Possible contributors include:

- useful local tissue adjacency;
- regularization from shared convolutional weights;
- a different optimization path through the residual map;
- bag-size and boundary shortcuts;
- the interaction between PPEG and the two transformer layers.

The ablation is therefore evidence for the value of the complete PPEG-equipped
architecture, not a causal proof that physical coordinates alone explain the
gain.

A sharper experiment would hold the feature multiset fixed and compare:

```math
\mathrm{AUC}_{\mathrm{true\ raster}}
-
\mathrm{AUC}_{\mathrm{random\ raster}}.
```

It would also report the effect of zero padding versus repeated-token
completion, because those are different operators.

## 12. C/R/G/S placement

The C/R/G/S decomposition makes the PPEG choice explicit.

| Component | TransMIL PPEG realization |
|---|---|
| Context operator | Three depthwise local convolutions on a square raster, combined with an identity residual; global transformer attention is supplied by the surrounding TransLayer. |
| Readout operator | Class-token readout after contextual transformer layers, not a direct convolutional pooling statistic. |
| Geometry | A row-major square grid induced by token order and square completion; physical coordinates are not explicit model inputs. |
| Surviving statistic | Local channelwise spatial responses plus globally mixed class-token evidence; arbitrary permutation identity is not preserved. |

The important placement is therefore not simply "Transformer MIL." It is

```math
\text{compact bag}
\xrightarrow{\text{square completion}}
\text{raster}
\xrightarrow{\text{local PPEG}}
\text{geometry-conditioned tokens}
\xrightarrow{\text{global attention}}
\text{class-token statistic}.
```

## 13. Failure modes stated as testable hypotheses

### 13.1 Serialization failure

If token order is shuffled independently of physical coordinates, PPEG should
lose or reverse its benefit. Test by applying one fixed random permutation per
slide at inference time while keeping the feature multiset unchanged.

```math
\Delta_{\mathrm{shuffle}}
=
\mathcal M(\text{true order})
-
\mathcal M(\text{shuffled order}).
```

A large negative value indicates reliance on the supplied order. It does not
distinguish useful geometry from a training-time order shortcut.

### 13.2 Gap compaction failure

Compare a coordinate-aware occupancy-preserving grid with a compacted sequence
for the same retained patches. If predictions change substantially, the model is
sensitive to preprocessing geometry.

### 13.3 Duplicate completion failure

Replace repeated-token completion with zero completion, masked completion, and
random-feature completion. Report the change in both performance and class-token
gradients. A model whose prediction is dominated by the completion convention
has learned a padding prior.

### 13.4 Boundary shortcut

Translate or crop the same tissue region within a larger grid. A shift-sensitive
prediction indicates that boundary conditions are functioning as a location
signal.

### 13.5 Scale shortcut

Extract the same tissue at a second magnification while preserving the physical
field of view. A drop that is not explained by the feature encoder indicates a
kernel-scale mismatch.

## 14. Minimal sanity suite

The following tests are small enough to run before interpreting a PPEG result.

### Test A: identity initialization

Set all convolution weights and biases to zero. The PPEG map should be the
identity on patch tokens.

```math
W^{(7)}=W^{(5)}=W^{(3)}=0,
\qquad
b^{(7)}=b^{(5)}=b^{(3)}=0
\Longrightarrow
\mathrm{PPEG}(V)=V.
```

### Test B: channel separation

Perturb one channel while holding all other channels fixed. Before transformer
mixing, the direct PPEG contribution should remain in that channel because the
convolutions are depthwise.

### Test C: local support

Perturb one interior grid cell and measure the immediate PPEG output. Nonzero
changes should be restricted to the corresponding 7 by 7 neighborhood in the
same channel, up to numerical tolerance.

### Test D: permutation stress

Apply a random token permutation before the reshape. A PPEG-equipped model is
not expected to be invariant. The test is useful precisely because a large
change reveals geometry dependence.

### Test E: completion stress

Choose bags with the same first (n_i) features but varying (n_i) near perfect
squares. Check whether predictions jump when (delta_i) changes.

```math
n_i=N^2-1,\ N^2,\ N^2+1
```

The feature content should be held as constant as possible while the completion
rule changes.

### Test F: coordinate-order agreement

Compute the rank correlation between grid distance and physical distance.

```math
\rho_i
=
\mathrm{corr}_{j<k}
\left(
\|g_{ij}-g_{ik}\|_2,
\|c_{ij}-c_{ik}\|_2
\right).
```

Low (ho_i) means that a PPEG result cannot honestly be described as learning
physical WSI geometry without additional evidence.

## 15. Design consequences

### 15.1 If the input is genuinely a set

PPEG adds an unverified order-dependent prior. A set-only baseline should be
reported so that the benefit of geometry is not inferred from a comparison
against a different capacity or optimization regime.

### 15.2 If coordinates are available

The model should preserve them through rasterization or use an explicit
coordinate-aware operator. A useful extension is to pass an occupancy mask and
coordinate offsets to the local operator rather than treating compact order as
geometry.

### 15.3 If the WSI is irregular

A graph, sparse convolution, or relative-coordinate attention may represent the
occupied domain more faithfully than square completion. The tradeoff is that
these operators introduce their own graph-construction or distance-scale
assumptions.

### 15.4 If bag sizes are very large

PPEG is cheap relative to dense self-attention because depthwise convolution has
linear spatial cost. It does not solve the memory cost of the global attention
layer. In TransMIL, the Nyström approximation addresses the latter; PPEG and
Nyström solve different bottlenecks.

## 16. Bottom line

PPEG is best understood as a learned, multiscale, depthwise local operator on a
square completion of the patch sequence. Its mathematical effect is to replace
arbitrary set symmetry with conditional geometry: the output depends on which
feature occupies which raster cell, while shared convolutional weights provide
limited translation structure away from boundaries.

The phrase "spatially aware WSI transformer" is justified only when the token
ordering or rasterization preserves the physical coordinate relation. Otherwise
the model is still spatially biased, but the learned space is the preprocessing
space. The correct C/R/G/S description is therefore:

```math
\boxed{
\text{order-dependent raster geometry}
+
\text{local depthwise context}
+
\text{global attention}
+
\text{class-token readout}
}
```

That is the surviving statistic the PPEG choice makes possible, and also the
assumption that must be stress-tested before attributing its gains to tissue
spatial structure.

## References

- Shao et al., "TransMIL: Transformer based Correlated Multiple Instance
  Learning for Whole Slide Image Classification," NeurIPS 2021.
  https://arxiv.org/abs/2106.00908
- Official TransMIL implementation, including the PPEG module and square token
  completion.
  https://github.com/szq0214/TransMIL
