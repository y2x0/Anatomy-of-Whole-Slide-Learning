# HIPT: Hierarchical DINO Distillation And Slide Tokens

Primary source: [Chen et al., Hierarchical Image Pyramid Transformer (HIPT)](https://arxiv.org/abs/2206.02647).

HIPT is not merely a ViT applied to a large image. Its representation is a
composition of separately pretrained Transformer maps whose sequence length is
kept near 256 at the patch and region stages. The paper's central object is the
recursive map

```math
x_{\mathrm{WSI}}
\longrightarrow
\left\{
    x_{4096}^{(k)}
\right\}_{k=1}^{M_i}
\longrightarrow
\left\{
    \mathrm{CLS}_{4096}^{(k)}
\right\}_{k=1}^{M_i}
\longrightarrow
z_i.
```

The scale changes the context available to each token, while the Transformer
at each level supplies the context operator.

## 1. The WSI Pyramid

At a fixed 20x objective, the paper treats a WSI as a nested sequence of
visual tokens:

```math
16\times16
\subset
256\times256
\subset
4096\times4096
\subset
x_{\mathrm{WSI}}.
```

The hierarchy is not a claim that the slide is naturally a perfectly balanced
tree. The number of 4096 by 4096 regions varies with tissue area, and the paper
reports a slide-level sequence length ranging from 1 to 256. Let:

```math
\mathcal R_i
=
\left\{
    R_{ik}
    :
    k=1,\ldots,M_i
\right\},
\qquad
R_{ik}\in\mathbb R^{4096\times4096\times3}.
```

Each region is partitioned into 256 non-overlapping 256 by 256 images:

```math
R_{ik}
=
\left\{
    P_{ikj}
    :
    j=1,\ldots,256
\right\},
\qquad
P_{ikj}\in\mathbb R^{256\times256\times3}.
```

Each 256 by 256 image is partitioned into 256 non-overlapping 16 by 16
visual tokens:

```math
P_{ikj}
=
\left\{
    X_{ikj\ell}
    :
    \ell=1,\ldots,256
\right\},
\qquad
X_{ikj\ell}\in\mathbb R^{16\times16\times3}.
```

Thus one 4096 by 4096 region contains:

```math
256\times256
=
65{,}536
```

cell-scale 16 by 16 images, but only 256 256 by 256 child images are passed to
the second Transformer stage. The hierarchy is a learned compression map, not
an explicit enumeration of all pairwise cell interactions.

## 2. Exact HIPT Forward Map

The paper uses three ViT modules:

```math
\mathrm{ViT}_{256\text{-}16},
\qquad
\mathrm{ViT}_{4096\text{-}256},
\qquad
\mathrm{ViT}_{\mathrm{WSI}\text{-}4096}.
```

The first module maps 16 by 16 image tokens inside a 256 by 256 patch to a
384-dimensional class token. Define:

```math
H_{ikj}
=
\left[
    h_{ikj1};
    \ldots;
    h_{ikj256}
\right]
\in
\mathbb R^{256\times384},
```

where:

```math
\left[
    h_{ikj1};
    \ldots;
    h_{ikj256}
\right]
=
\mathrm{ViT}_{256\text{-}16}
\left(
    \left\{
        X_{ikj\ell}
    \right\}_{\ell=1}^{256}
\right),
```

and h_ikjell denotes the final token at the corresponding cell position. The
exported patch token is the class token:

```math
u_{ikj}
=
\mathrm{CLS}_{256}^{(ikj)}
\in
\mathbb R^{384}.
```

The paper's first-stage ViT uses 8 Transformer layers, 6 attention heads, and
width 384. Its class token is therefore a learned summary of 256 cell-scale
tokens rather than a feature extracted from one 16 by 16 image in isolation.

The 256 child patch tokens of region R_ik are arranged as a 16 by 16 feature
grid:

```math
U_{ik}
=
\left[
    u_{ik1};
    \ldots;
    u_{ik256}
\right]
\in
\mathbb R^{256\times384}.
```

The second module applies self-attention over this feature grid and exports a
192-dimensional region token:

```math
r_{ik}
=
\mathrm{CLS}_{4096}^{(ik)}
=
\mathrm{ViT}_{4096\text{-}256}
\left(
    U_{ik}
\right)
\in
\mathbb R^{192}.
```

Finally, the slide-level module receives the variable-length sequence of region
tokens:

```math
R_i
=
\left[
    r_{i1};
    \ldots;
    r_{iM_i}
\right]
\in
\mathbb R^{M_i\times192}.
```

The slide representation is:

```math
z_i
=
\mathrm{CLS}_{\mathrm{WSI}}^{(i)}
=
\mathrm{ViT}_{\mathrm{WSI}\text{-}4096}
\left(
    R_i
\right)
\in
\mathbb R^{192}.
```

The complete composition is therefore:

```math
z_i
=
F_3
\left(
    \left\{
        F_2
        \left(
            \left\{
                F_1
                \left(
                    \left\{
                        X_{ikj\ell}
                    \right\}_{\ell=1}^{256}
                \right)
            \right\}_{j=1}^{256}
        \right)
    \right\}_{k=1}^{M_i}
\right).
```

The tensor shapes are the important mathematical contract:

| Stage | Input object | Token sequence | Output token |
|---|---|---:|---:|
| ViT_256-16 | one 256 by 256 patch | 256 tokens of width 384 | one patch token in R^384 |
| ViT_4096-256 | one 4096 by 4096 region | 256 tokens of width 384 | one region token in R^192 |
| ViT_WSI-4096 | one WSI | M_i region tokens of width 192 | one slide token in R^192 |

This is a three-level hierarchy in the paper's image-pyramid sense, although
the expensive self-supervised aggregation is concentrated in the first two
levels.

## 3. Transformer Context At One Level

For a sequence V with N tokens and width d, a self-attention head has:

```math
Q
=
VW_Q,
\qquad
K
=
VW_K,
\qquad
V'
=
VW_V.
```

The attention weights are:

```math
A_{ab}
=
\frac{
\exp
\left(
    q_a^{\mathsf T}k_b/\sqrt{d_h}
\right)
}{
\sum_{c=1}^{N}
\exp
\left(
    q_a^{\mathsf T}k_c/\sqrt{d_h}
\right)
}.
```

The output of the class-token row is:

```math
o_{\mathrm{CLS}}
=
\sum_{b=0}^{N}
A_{0b}v_b.
```

The class token can therefore depend on all child tokens at that level in one
Transformer block. With residual connections and multiple layers, the learned
summary is not a fixed arithmetic mean:

```math
\mathrm{CLS}_{\mathrm{out}}
\neq
\frac{1}{N}
\sum_{b=1}^{N}
v_b
```

in general. The class token is a learned, query-dependent readout of the local
sequence.

At the patch stage, N=256 means each 256 by 256 patch can model interactions
among its 256 cell-scale tokens. At the region stage, N=256 means each
4096 by 4096 region can model interactions among its 256 patch-scale tokens.
At the WSI stage, N=M_i is variable and is at most 256 in the reported
setting.

The architecture replaces a flat WSI attention problem with:

```math
\mathrm{attention}
\left(
    256M_i
\right)^2
\quad\longrightarrow\quad
M_i\,\mathrm{attention}(256)
+
\mathrm{attention}(M_i).
```

The right-hand side is the computation used by the hierarchy, up to layer,
head, and width constants.

## 4. Position And The Region Grid

At the first stage, cell-scale tokens are placed in the 16 by 16 patch grid.
At the second stage, the 256 patch class tokens are rearranged into a 16 by 16
feature grid before cropping and Transformer processing:

```math
\Gamma_{ik}
\in
\mathbb R^{16\times16\times384}.
```

If u_ikj is the token at grid coordinate q_j, a positional embedding gives:

```math
\widetilde u_{ikj}
=
u_{ikj}
+
e_{\mathrm{pos}}(q_j).
```

This means the second-stage class token can distinguish the same local feature
placed at different positions in the 4096 by 4096 region.

The WSI-level stage is different. Because 4096 by 4096 regions can be missing
or irregularly arranged after tissue segmentation, the paper reports ignoring
positional embeddings at this stage. The WSI module therefore receives:

```math
R_i
=
\left[
    r_{i1};
    \ldots;
    r_{iM_i}
\right]
```

without a fixed global positional grid in the Transformer input. The slide
level can still model interactions among region tokens, but it does not receive
the same explicit absolute coordinate channel as the first two scales.

This distinction matters for representation taxonomy:

```math
\text{HIPT region object}
=
\text{ordered 16 by 16 feature grid},
```

whereas:

```math
\text{HIPT slide object}
=
\text{variable sequence of region tokens with no stage-3 position embedding}.
```

Calling both levels simply a sequence hides the different geometry contracts.

## 5. Stage 1 DINO Pretraining

HIPT first pretrains ViT_256-16 with DINO. Let t_256 and s_256 denote teacher
and student networks. For a global teacher crop v_g^a(x) and a student crop
v_s^b(x), define:

```math
q_{256}^{a}
=
\mathrm{softmax}
\left(
    \frac{
        g_t\left(t_{256}(v_g^a(x))\right)-c_t
    }{
        \tau_t
    }
\right),
```

```math
p_{256}^{b}
=
\mathrm{softmax}
\left(
    \frac{
        g_s\left(s_{256}(v_s^b(x))\right)
    }{
        \tau_s
    }
\right).
```

The cross-view loss is:

```math
\mathcal L_{256}
=
\frac{1}{M_gM_s}
\sum_{a=1}^{M_g}
\sum_{b=1}^{M_s}
\left[
    -
    \sum_{k=1}^{K}
    q_{256,k}^{a}
    \log
    p_{256,k}^{b}
\right].
```

The paper uses two global crops of size 224 by 224 and eight local crops of
size 96 by 96 for the 256 by 256 input. The teacher sees global views while
the student sees global and local views:

```math
\mathcal L_{256}
=
\frac{1}{2\cdot10}
\sum_{a=1}^{2}
\left(
    \sum_{b=1}^{2}
    H(q_{256}^{a},p_{256}^{b})
    +
    \sum_{b=1}^{8}
    H(q_{256}^{a},p_{256,\mathrm{local}}^{b})
\right).
```

The teacher is momentum-updated:

```math
\theta_t^{(r)}
=
m_r\theta_t^{(r-1)}
+
\left(1-m_r\right)
\theta_s^{(r)}.
```

The student receives the gradient. The teacher target is treated as fixed
during the student update. The DINO loss therefore encourages agreement
between views while preventing the optimization from treating two augmented
views as unrelated instances.

For a cell-scale representation, the surviving statistic is not a label
posterior. It is the class-token geometry induced by the view kernel:

```math
\mathbb E_{v_1,v_2}
\left[
    \left\|
        \mathrm{CLS}_{256}(v_1(x))
        -
        \mathrm{CLS}_{256}(v_2(x))
    \right\|_2^2
\right]
\quad\text{is pushed downward}.
```

The augmentations include horizontal flips and color jitter, with solarization
on one global view. Features that are unstable under those transformations are
penalized by the training contract.

## 6. Stage 2 Hierarchical DINO Pretraining

After stage 1, the ViT_256-16 weights are fixed and reused as the embedding map
for the second stage. For each 4096 by 4096 region, the 256 child class tokens
form the input sequence:

```math
U_{ik}
=
\left[
    \mathrm{CLS}_{256}^{(ik1)};
    \ldots;
    \mathrm{CLS}_{256}^{(ik256)}
\right].
```

The second-stage teacher and student consume different crops of the 16 by 16
feature grid. The paper uses 14 by 14 global crops and 6 by 6 local crops,
matching the 224 by 224 and 96 by 96 crop proportions from stage 1. Let
C_g^a and C_l^b be feature-grid crop operators. Then:

```math
q_{4096}^{a}
=
\mathrm{softmax}
\left(
    \frac{
        g_t
        \left(
            t_{4096}
            \left(
                C_g^a(U_{ik})
            \right)
        \right)
        -
        c_t
    }{
        \tau_t
    }
\right),
```

```math
p_{4096}^{b}
=
\mathrm{softmax}
\left(
    \frac{
        g_s
        \left(
            s_{4096}
            \left(
                C_b(U_{ik})
            \right)
        \right)
    }{
        \tau_s
    }
\right),
```

where C_b is either a global 14 by 14 or local 6 by 6 crop. The stage-2 DINO
objective has the same cross-view form:

```math
\mathcal L_{4096}
=
\frac{1}{M_gM_s}
\sum_{a=1}^{M_g}
\sum_{b=1}^{M_s}
H
\left(
    q_{4096}^{a},
    p_{4096}^{b}
\right).
```

The second stage learns a geometry over arrangements of already learned patch
tokens. It is not equivalent to applying DINO independently to the 256 by 256
images. The input to stage 2 is:

```math
\mathrm{CLS}_{256}
\left(
    P_{ikj}
\right),
```

not raw pixels from one child patch.

The stage-2 model uses a 4-layer, 3-head ViT with width 192. A region token
therefore compresses a 16 by 16 spatial field of 384-dimensional child tokens
into a 192-dimensional vector:

```math
\mathbb R^{16\times16\times384}
\xrightarrow{\ \mathrm{ViT}_{4096\text{-}256}\ }
\mathbb R^{192}.
```

The paper also applies dropout with probability 0.10 to the views at this
stage. The second-stage student is trained to predict teacher geometry from a
partially perturbed feature grid, not from a perfectly deterministic copy.

## 7. Slide-Level Aggregation And Fine-Tuning

The WSI stage receives one 192-dimensional token per 4096 by 4096 region:

```math
R_i
=
\left[
    r_{i1};
    \ldots;
    r_{iM_i}
\right]
\in
\mathbb R^{M_i\times192}.
```

The slide Transformer is a 2-layer, 3-head ViT with width 192. Its output
class token is the HIPT slide representation:

```math
z_i
=
F_{\mathrm{WSI}}
\left(
    R_i
\right)
\in
\mathbb R^{192}.
```

Unlike stages 1 and 2, the paper does not pretrain this final aggregation layer
with the same large-scale DINO procedure. Following hierarchical pretraining,
the lower two subnetworks are frozen and only the lightweight WSI module is
fine-tuned for the slide task:

```math
\phi_1,\phi_2
\ \text{fixed},
\qquad
\phi_3
\ \text{optimized on slide labels}.
```

For a classification target, the task loss can be written:

```math
\mathcal L_{\mathrm{task}}^{\mathrm{cls}}
=
-
\sum_{c=1}^{C}
y_{ic}
\log
\mathrm{softmax}_c
\left(
    Wz_i+b
\right).
```

For survival, the paper evaluates a survival cross-entropy objective. A
discrete-time version with interval hazard q_ik is:

```math
\mathcal L_{\mathrm{task}}^{\mathrm{surv}}
=
-
\sum_{k=1}^{K}
\left[
    m_{ik}d_{ik}\log q_{ik}
    +
    m_{ik}(1-d_{ik})\log(1-q_{ik})
\right],
```

where m_ik masks intervals after censoring and d_ik marks the event interval.
HIPT supplies z_i; the survival head and censoring likelihood determine the
task-level statistical model.

This separation is essential. HIPT's hierarchical visual representation is not
itself a Cox model, a discrete hazard model, or a survival curve.

## 8. Complexity: Why The Hierarchy Exists

Let N_i=256M_i be the number of 256 by 256 patch tokens that a flat WSI
Transformer would process. One flat self-attention layer has quadratic token
cost:

```math
\mathcal C_{\mathrm{flat}}(i)
\propto
N_i^2
=
(256M_i)^2.
```

HIPT computes local and regional attention separately:

```math
\mathcal C_{\mathrm{hier}}(i)
\propto
M_i\cdot256^2
+
M_i^2.
```

The ratio is:

```math
\frac{
\mathcal C_{\mathrm{hier}}(i)
}{
\mathcal C_{\mathrm{flat}}(i)
}
=
\frac{
M_i256^2+M_i^2
}{
256^2M_i^2
}
=
\frac{1}{M_i}
+
\frac{1}{256^2}.
```

For large M_i, this is much smaller than one. The gain is not free: the
architecture has inserted a region bottleneck, and it cannot represent every
fine-scale cross-region interaction with the same capacity as a flat model.

If the number of regions is one, the hierarchy has no cross-region problem:

```math
M_i=1
\Longrightarrow
\mathcal C_{\mathrm{hier}}(i)
\propto
256^2+1.
```

If M_i is large, the stage-3 term becomes material:

```math
M_i^2
\not\ll
M_i256^2
\quad\text{when}\quad
M_i
\text{ is comparable to }256^2.
```

The hierarchy is computationally attractive in the paper's regime, but its
cost should be evaluated with the actual number of tissue regions, not only the
maximum sequence length.

## 9. What The Region Bottleneck Erases

Let F_2 map a region's 256 child tokens to one region token:

```math
r_{ik}
=
F_2
\left(
    U_{ik}
\right).
```

Any two child arrangements with the same F_2 output are indistinguishable to
the WSI stage:

```math
F_2(U_{ik})
=
F_2(U'_{ik})
\Longrightarrow
F_3(R_i)
=
F_3(R'_i)
```

whenever the two slides differ only through that region and all other region
tokens are equal.

This is a learned sufficient-statistic condition, not a guarantee that the
region token is sufficient for the downstream task. The class token is trained
to preserve information useful for DINO-style region discrimination, not
necessarily information needed for a rare prognostic lesion.

The same logic applies recursively:

```math
\begin{aligned}
U_{ik}
&\longrightarrow r_{ik}
&&\text{discards within-region information},\\
R_i
&\longrightarrow z_i
&&\text{discards across-region information}.
\end{aligned}
```

The surviving statistic is therefore scale-dependent:

```math
S_{\mathrm{HIPT}}
=
\left(
    S_{16\to256},
    S_{256\to4096},
    S_{4096\to\mathrm{WSI}}
\right).
```

The first two terms are pretrained self-distilled Transformer summaries. The
last term is a task-adapted slide summary.

## 10. Hierarchical Attention As A Path Statistic

Let a^(1)_jell be the stage-1 attention from patch class token j to cell token
ell, a^(2)_kj be stage-2 attention from region class token k to patch token j,
and a^(3)_ik be stage-3 attention from the slide class token to region token k.

An idealized hierarchical relevance score for cell (i,k,j,ell) is:

```math
A_{ikj\ell}
=
a^{(3)}_{ik}
a^{(2)}_{kj}
a^{(1)}_{j\ell}.
```

This is the probability of a path under a product-of-attention approximation.
For a real Transformer, residual connections, multiple heads, layer
normalization, and nonlinear feed-forward blocks mean that this product is not
the exact gradient or causal attribution. It is a structured visualization
statistic.

The region-level marginal is:

```math
A_{ik}^{\mathrm{region}}
=
\sum_{j=1}^{256}
\sum_{\ell=1}^{256}
A_{ikj\ell}.
```

The fine-scale marginal for a slide is:

```math
A_{ikj\ell}^{\mathrm{slide}}
=
a^{(3)}_{ik}
a^{(2)}_{kj}
a^{(1)}_{j\ell}.
```

The factorization explains why HIPT can visualize both fine cell-scale
phenotypes and coarse tumor-stroma organization. It also explains the
interpretability limitation: a high product score identifies a high-attention
path, not necessarily a sufficient causal explanation of the slide label.

## 11. Paper-Level Training Contract

The paper's pretraining and transfer contract can be written as:

```math
\begin{aligned}
\text{Stage 1:}\quad
&\theta_{256}^{\star}
=
\arg\min_{\theta_{256}}
\mathcal L_{256}
\left(
    \theta_{256}
\right),
\\
\text{Stage 2:}\quad
&\theta_{4096}^{\star}
=
\arg\min_{\theta_{4096}}
\mathcal L_{4096}
\left(
    \theta_{4096};
    \theta_{256}^{\star}
\right),
\\
\text{Stage 3:}\quad
&\theta_{\mathrm{WSI}}^{\star}
=
\arg\min_{\theta_{\mathrm{WSI}}}
\mathcal L_{\mathrm{task}}
\left(
    \theta_{\mathrm{WSI}};
    \theta_{256}^{\star},
    \theta_{4096}^{\star}
\right).
\end{aligned}
```

The semicolon indicates that lower-level weights are fixed in the next
optimization problem. This is not end-to-end training on slide labels:

```math
\nabla_{\theta_{256}}
\mathcal L_{\mathrm{task}}
=
0,
\qquad
\nabla_{\theta_{4096}}
\mathcal L_{\mathrm{task}}
=
0
```

under the paper's frozen-transfer setup.

The reported pretraining scale is:

```math
10{,}678\ \mathrm{WSIs}
\longrightarrow
408{,}218\ \mathrm{regions}_{4096}
\longrightarrow
104\ \mathrm{million\ patches}_{256}.
```

The stage-1 model is trained for 400,000 iterations with batch size 256,
AdamW, base learning rate 0.0005, warmup, and cosine decay. The stage-2 model
is trained for 200,000 iterations from pre-extracted stage-1 class tokens.
These details matter because stage 2 is not learning from raw pixels and is not
optimized jointly with the final slide task.

## 12. C/R/G/S Placement

| Component | HIPT instantiation |
|---|---|
| C context operator | ViT self-attention within 256-cell patches, within 4096 regions, and across WSI region tokens |
| R readout operator | class-token extraction at each stage; final WSI class token for the slide representation |
| G geometry | 384-dimensional patch-token space, 192-dimensional region-token space, and 192-dimensional slide-token space |
| S surviving statistic | DINO-stable local morphology, learned region-scale interactions, and task-adapted cross-region composition |

The hierarchy is a representation choice before it is a computational trick:

```math
\mathrm{slide}
=
\mathrm{Transformer}_{\mathrm{WSI}}
\circ
\mathrm{Transformer}_{4096}
\circ
\mathrm{Transformer}_{256}
\left(
    \mathrm{cells}
\right).
```

The architecture places context at three scales, but each scale has a separate
information bottleneck.

## 13. Failure Modes

### 13.1 Region Boundary Failure

If a prognostic interaction crosses a 4096 by 4096 region boundary, stage 2
cannot model it in one attention window. The WSI stage sees only two compressed
tokens:

```math
\left(
    U_{ik},
    U_{i\ell}
\right)
\longrightarrow
\left(
    r_{ik},
    r_{i\ell}
\right).
```

Fine-scale evidence requiring joint attention over both child grids is
available only through the much smaller region-level interaction.

### 13.2 Frozen-Encoder Failure

The lower stages are frozen during task fine-tuning. If the DINO geometry
collapses a task-relevant distinction, the final WSI module cannot reconstruct
it:

```math
F_1(X)
=
F_1(X')
\Longrightarrow
F_3\circ F_2\circ F_1(X)
=
F_3\circ F_2\circ F_1(X').
```

The task head can only reweight the available tokens.

### 13.3 Stage-3 Position Failure

Ignoring WSI-level positional embeddings makes the final Transformer less
sensitive to absolute region coordinates. This can be useful under irregular
tissue segmentation, but it weakens the representation for tasks requiring
absolute layout:

```math
R_i
\mapsto
\Pi R_i
\Longrightarrow
F_{\mathrm{WSI}}(R_i)
\approx
F_{\mathrm{WSI}}(\Pi R_i).
```

The approximation becomes exact under a permutation-equivariant implementation
with no coordinate side channel. Real Transformer details can modify it, but no
explicit global coordinate channel remains in the reported stage-3 input.

### 13.4 Mean-KNN Evaluation Failure

The paper evaluates pre-extracted embeddings with mean WSI representations and
K-nearest-neighbor classification. For region tokens:

```math
\bar r_i
=
\frac{1}{M_i}
\sum_{k=1}^{M_i}
r_{ik}.
```

This is a useful representation probe, but it is not the same object as the
fine-tuned WSI class token:

```math
\bar r_i
\neq
z_i.
```

A strong KNN result on mean region features and a strong task-fine-tuned HIPT
result answer different questions.

### 13.5 Small Slide-Level Sample Size

The final slide module has fewer training examples than the patch and region
stages. Its parameter count is small, but its optimization still operates on a
patient-level sample:

```math
n_{\mathrm{slides}}
\ll
n_{\mathrm{patches}}.
```

Freezing the lower stages reduces task-level variance, but it also prevents
slide labels from correcting all lower-level errors.

### 13.6 Unsupported Causal Interpretation

The hierarchical attention product A_ikjell is a visualization statistic. It
does not imply:

```math
\Pr
\left(
    Y_i
    \mid
    X_{ikj\ell}
\right)
\text{ changes by the displayed attention weight}.
```

Attention measures the operator's routing preference, not an intervention
effect.

## 14. Sanity Checks

### Same Patch Histogram, Different Region Layout

Let two regions contain the same 256 child vectors but in different spatial
arrangements. If stage 2 includes positional embeddings, then:

```math
U'
=
\Pi U,
\qquad
\Pi\neq I
\Longrightarrow
F_2(U')
\text{ may differ from }
F_2(U).
```

If positional embeddings are removed and the Transformer is permutation
equivariant, then the class-token readout is permutation-invariant:

```math
F_2(\Pi U)
=
F_2(U).
```

This is the exact experiment for checking whether the region module uses layout
or only a multiset of patch features.

### Same Region Tokens, Different Region Coordinates

At the WSI stage, construct:

```math
R'_i
=
\Pi R_i.
```

The same region vectors in a different order should produce less change under
the reported stage-3 input contract than under a coordinate-aware hierarchical
model.

### Rare Lesion Inside A Compressed Region

Inject a single rare lesion token into one region and compare:

```math
\Delta z_i
=
F_3
\left(
    R_i
\text{ with lesion}
\right)
-
F_3
\left(
    R_i
\text{ without lesion}
\right).
```

The magnitude of Delta z_i measures end-to-end survival of a local perturbation
through both class-token bottlenecks. It is more informative than inspecting
the attention map at only one scale.

## 15. Bottom Line

HIPT represents a WSI as a nested sequence of visual tokens:

```math
16\times16
\xrightarrow{\ \mathrm{ViT}_{256\text{-}16}\ }
256\times256
\xrightarrow{\ \mathrm{ViT}_{4096\text{-}256}\ }
4096\times4096
\xrightarrow{\ \mathrm{ViT}_{\mathrm{WSI}\text{-}4096}\ }
\mathrm{WSI}.
```

Its mathematical contribution is the factorization of long-range slide
representation into manageable self-attention problems with a learned class
token at every scale. The source-faithful facts are:

- 256 cell-scale tokens per 256 by 256 patch;
- 256 patch-scale tokens per 4096 by 4096 region;
- 384-dimensional patch tokens and 192-dimensional region and slide tokens;
- DINO pretraining at patch and region levels;
- fixed lower levels and lightweight WSI-level fine-tuning;
- no explicit stage-3 positional embeddings in the reported implementation.

The hierarchy improves context length and computational feasibility by
compressing information. It therefore creates a precise design question:

```math
\text{Which prognostic interactions survive }
F_1,
F_2,
\text{and }
F_3
\text{ at their respective scales?}
```

That question is the mathematical boundary between hierarchical representation
learning and ordinary MIL pooling.
