# CTransPath: SRCL, CNN Stem, And Swin Context

Primary source: [Wang et al., CTransPath](https://pubmed.ncbi.nlm.nih.gov/35952419/).
The paper's method is a hybrid CNN-Swin backbone trained with
semantically-relevant contrastive learning, or SRCL. It should not be reduced
to an ordinary MoCo queue objective.

The method has three separate mathematical objects:

```math
\text{image encoder}
\quad
f_\theta,
\qquad
\text{positive-mining memory bank}
\quad
\mathcal M,
\qquad
\text{contrastive objective}
\quad
\mathcal L_{\mathrm{SRCL}}.
```

The hybrid architecture determines patch geometry. SRCL determines which
patches are pulled together and which are pushed apart.

## 1. Patch Dataset And Views

The unlabeled pretraining dataset is:

```math
\mathcal D_u
=
\left\{
    x_i
    :
    x_i\in\mathbb R^{H\times W\times3}
\right\}_{i=1}^{N}.
```

The paper pretrains on approximately 15 million unlabeled patches from TCGA
and PAIP. The reported TCGA component contains 29,763 20x WSIs and
14,325,848 non-overlapping 1024 by 1024 patches. PAIP contributes 1,254,414
additional unlabeled patches.

For a randomly selected patch x, construct two augmented views:

```math
x_1=T_1(x),
\qquad
x_2=T_2(x).
```

The reported pathology-oriented augmentations include random cropping,
Gaussian blur, and hue and saturation shifts in HSV space. The loss therefore
declares the two views equivalent even when the transformation changes a
feature that might matter for a downstream task.

The online and momentum-target branches produce:

```math
y_1=f_\theta(x_1),
\qquad
\widehat y_2=f_\xi(x_2),
\qquad
y_2=f_\theta(x_2),
\qquad
\widehat y_1=f_\xi(x_1).
```

Projection heads give:

```math
z_1=g_\theta(y_1),
\qquad
z_2=g_\theta(y_2),
\qquad
\widehat z_1=g_\xi(\widehat y_1),
\qquad
\widehat z_2=g_\xi(\widehat y_2).
```

The online vector is an anchor. A target vector refreshes the memory bank. The
shared-target branch processes the original unaltered patch and queries the
bank for semantically similar samples.

## 2. CNN To Swin Context

CTransPath replaces Swin's usual patch partition with a convolutional stem.
The stem has three consecutive convolutions with kernel sizes:

```math
3\times3,
\qquad
3\times3,
\qquad
1\times1.
```

For input X, the stem produces:

```math
F
=
\mathcal C_\theta(X)
\in
\mathbb R^{(H/4)\times(W/4)\times C}.
```

The downsampling factor is four. The local feature field is divided into
M by M windows:

```math
I_r
\in
\mathbb R^{M\times M\times C},
\qquad
r=1,\ldots,
\frac{H}{4M}\frac{W}{4M}.
```

For one window:

```math
Q=I_rW_Q,
\qquad
K=I_rW_K,
\qquad
V=I_rW_V.
```

Window attention is:

```math
\mathrm{W\text{-}MSA}(I_r)
=
\mathrm{softmax}
\left(
    \frac{QK^{\mathsf T}}{\sqrt{d_h}}
    +
    B
\right)V.
```

The shifted-window operator changes the partition by:

```math
\left(
\left\lfloor\frac{M}{2}\right\rfloor,
\left\lfloor\frac{M}{2}\right\rfloor
\right).
```

A two-layer Swin block is:

```math
\widehat Z^\ell
=
\mathrm{W\text{-}MSA}
\left(
\mathrm{LN}(Z^{\ell-1})
\right)
+
Z^{\ell-1},
```

```math
Z^\ell
=
\mathrm{MLP}
\left(
\mathrm{LN}(\widehat Z^\ell)
\right)
+
\widehat Z^\ell,
```

```math
\widehat Z^{\ell+1}
=
\mathrm{SW\text{-}MSA}
\left(
\mathrm{LN}(Z^\ell)
\right)
+
Z^\ell,
```

```math
Z^{\ell+1}
=
\mathrm{MLP}
\left(
\mathrm{LN}(\widehat Z^{\ell+1})
\right)
+
\widehat Z^{\ell+1}.
```

The CNN supplies local morphology, W-MSA supplies within-window context, and
SW-MSA permits information to cross the original window boundaries:

```math
f_\theta
=
\mathcal T_{\mathrm{Swin}}
\circ
\mathcal C_\theta.
```

This context operator acts inside a patch. It is not a WSI graph and does not
connect two independently extracted slide patches.

## 3. Momentum Target And Memory Bank

The target branch follows the online branch by exponential moving average:

```math
\xi^{(r)}
=
m\xi^{(r-1)}
+
(1-m)\theta^{(r)}.
```

The target features are stop-gradient during the online update:

```math
\nabla_\theta\widehat z_1
=
0,
\qquad
\nabla_\theta\widehat z_2
=
0.
```

At iteration r, the memory bank is:

```math
\mathcal M^{(r)}
=
\left\{
    c_q^{(r)}
    :
    q=1,\ldots,Q
\right\}.
```

It is populated by target features from previous minibatches and refreshed
after the current iteration:

```math
\mathcal M^{(r+1)}
=
\mathrm{QueueUpdate}
\left(
\mathcal M^{(r)},
\left\{
\widehat z_{\mathrm{target}}^{(r)}
\right\}
\right).
```

The bank is independent of the current minibatch for support search. It is not
correct to describe every bank entry as a current-step negative in SRCL.

For query z and bank vector c_q:

```math
D(z,c_q)
=
\frac{
z^{\mathsf T}c_q
}{
\|z\|_2\|c_q\|_2
}.
```

Let q_(1),...,q_(Q) sort the bank by descending D. The mined support is:

```math
\mathcal P_S(z)
=
\left\{
q_{(1)},\ldots,q_{(S)}
\right\}.
```

The paper uses S=4. These are pseudo-positives, because cosine similarity does
not prove shared diagnosis, patient identity, or even shared tissue site.

## 4. Positive And Negative Sets

For anchor z_1, the conventional augmented positive is z_hat_2. The full
positive set is:

```math
\mathcal A(z_1)
=
\left\{
\widehat z_2
\right\}
\cup
\left\{
c_q:q\in\mathcal P_S(z_1)
\right\}.
```

Therefore:

```math
|\mathcal A(z_1)|
=
S+1.
```

The negative set is formed from different views in the current minibatch:

```math
\mathcal N_1
=
\left\{
n_1,\ldots,n_N
\right\}.
```

Two distinct roles are easy to confuse:

| Object | Source | Role in SRCL |
|---|---|---|
| target feature | previous or current target branch | conventional positive and bank update |
| memory bank | previous minibatches | top-S pseudo-positive search |
| current minibatch | current online/target views | negative denominator |

The memory bank is therefore a retrieval mechanism for positives, not an
all-negative MoCo queue in the final SRCL denominator.

## 5. Exact SRCL Loss

For anchor z, positive set A, and negative set N, define:

```math
\mathcal L_2(z;A,N)
=
-
\log
\frac{
\displaystyle
\sum_{a\in A}
\exp
\left(
\frac{a^{\mathsf T}z}{\tau}
\right)
}{
\displaystyle
\sum_{a\in A}
\exp
\left(
\frac{a^{\mathsf T}z}{\tau}
\right)
+
\sum_{n\in N}
\exp
\left(
\frac{n^{\mathsf T}z}{\tau}
\right)
}.
```

The two-view SRCL objective is:

```math
\mathcal L_{\mathrm{SRCL}}
=
\frac{1}{2}
\mathcal L_2
\left(
z_1;
\mathcal A(z_1),
\mathcal N_1
\right)
+
\frac{1}{2}
\mathcal L_2
\left(
\widehat z_2;
\mathcal A(\widehat z_2),
\mathcal N_2
\right).
```

For the second term, the roles of the branches are swapped and z_1 is the
conventional positive. A one-positive InfoNCE loss would instead use:

```math
\mathcal L_{\mathrm{one}}(z_1)
=
-
\log
\frac{
\exp
\left(
\widehat z_2^{\mathsf T}z_1/\tau
\right)
}{
\exp
\left(
\widehat z_2^{\mathsf T}z_1/\tau
\right)
+
\sum_{n\in\mathcal N_1}
\exp
\left(
n^{\mathsf T}z_1/\tau
\right)
}.
```

SRCL changes the numerator from one positive mass to S+1 positive masses.
That changes the gradient. Conditional on a fixed support:

```math
\frac{\partial\mathcal L_2}{\partial z}
=
-
\frac{
\sum_{a\in A}
\exp(a^{\mathsf T}z/\tau)a
}{
\tau\sum_{a\in A}
\exp(a^{\mathsf T}z/\tau)
}
+
\frac{
\sum_{w\in A\cup N}
\exp(w^{\mathsf T}z/\tau)w
}{
\tau\sum_{w\in A\cup N}
\exp(w^{\mathsf T}z/\tau)
}.
```

The first term is the positive-set barycenter. The second is the
all-candidate barycenter. If mined positives point in conflicting directions,
their weighted barycenter can have small norm even when each individual
similarity is high.

## 6. Discrete Support Selection

For fixed bank vectors, the loss is differentiable in z conditional on
P_S(z). The support map is discontinuous at ties:

```math
D(z,c_q)
=
D(z,c_{q'})
\Longrightarrow
\mathcal P_S(z)
\text{ can switch}.
```

Thus SRCL is a piecewise-smooth objective:

```math
\mathcal L_{\mathrm{SRCL}}(z)
=
\mathcal L_{\mathrm{SRCL}}
\left(
z;\mathcal P_S(z)
\right).
```

The gradient used by backpropagation treats the selected support as fixed for
that step:

```math
\nabla_z\mathcal L_{\mathrm{SRCL}}
\Big|_{\mathcal P_S(z)\ \mathrm{fixed}}.
```

The top-S search is an outer discrete update. It is not a differentiable
attention layer and no gradient is propagated through the identity of the
retrieved memory entries.

## 7. Temperature And Training Curriculum

The reported contrastive temperature is:

```math
\tau=0.2.
```

The logit for candidate w is:

```math
\ell_w
=
\frac{w^{\mathsf T}z}{\tau},
\qquad
\frac{\partial\ell_w}{\partial z}
=
\frac{w}{\tau}.
```

The method uses a five-epoch conventional contrastive warmup before mining
supports. It then uses SRCL for the rest of the 100-epoch pretraining run.
The schedule is:

```math
\mathcal L^{(r)}
=
\begin{cases}
\mathcal L_{\mathrm{one}},
&
r\le r_{\mathrm{warm}},\\
\mathcal L_{\mathrm{SRCL}},
&
r>r_{\mathrm{warm}}.
\end{cases}
```

The warmup is mathematically a support-construction curriculum. Before the
feature geometry is usable, top-S retrieval can be mostly noise. The curriculum
delays the pseudo-label mechanism until a rough angular geometry exists.

The pretraining setup reports batch size 1024, AdamW, initial learning rate
0.00015, cosine decay, and a 40-epoch learning-rate warmup. The size of the
memory bank Q and the minibatch size N affect different parts of the method:

```math
Q
\longrightarrow
\text{candidate diversity},
\qquad
N
\longrightarrow
\text{negative support}.
```

Increasing Q does not guarantee semantic purity. Increasing N does not remove
false negatives from adjacent or same-tissue patches.

## 8. Transfer To WSI Learning

For slide i, the frozen encoder gives:

```math
h_{ij}
=
f_{\theta^\star}(x_{ij}),
\qquad
j=1,\ldots,n_i.
```

A generic attention MIL readout is:

```math
e_{ij}
=
w_a^{\mathsf T}
\tanh
\left(
V_a h_{ij}+b_a
\right),
```

```math
\alpha_{ij}
=
\frac{\exp(e_{ij})}{
\sum_{\ell=1}^{n_i}\exp(e_{i\ell})
},
\qquad
z_i
=
\sum_{j=1}^{n_i}
\alpha_{ij}h_{ij}.
```

The complete frozen-feature pipeline is:

```math
\widehat y_i
=
g_\omega
\left(
\mathcal R_\psi
\left(
\left\{
f_{\theta^\star}(x_{ij})
\right\}_{j=1}^{n_i}
\right)
\right).
```

CTransPath's coordinates do not enter f_theta. Any spatial inductive bias must
be supplied by R_psi, for example through coordinate-aware attention, graph
edges, a hierarchy, or a slide Transformer.

The paper's weakly supervised WSI experiments freeze the pretrained backbone
and train a CLAM-style bag model. A strong result therefore supports the
transferability of the patch geometry; it does not establish that SRCL is a
slide-level aggregator.

## 9. What SRCL Preserves

The view term pushes:

```math
\mathbb E_{T_1,T_2}
\left[
\left\|
f_\theta(T_1x)-f_\theta(T_2x)
\right\|_2^2
\right]
\text{ downward}.
```

The mining term additionally pushes:

```math
D
\left(
g_\xi(x_{\mathrm{orig}}),
c_q
\right)
\text{ high}
\Longrightarrow
g_\theta(f_\theta(Tx))
\approx
c_q
```

The method combines augmentation invariance and cross-instance morphology
alignment. It does not require a mined positive to share patient, slide, organ,
or diagnosis with the anchor.

If all S mined positives have approximately the same direction a_bar:

```math
\sum_{s=1}^{S+1}
\exp
\left(
\frac{a_s^{\mathsf T}z}{\tau}
\right)
\approx
(S+1)
\exp
\left(
\frac{\bar a^{\mathsf T}z}{\tau}
\right).
```

The multiplicity increases positive mass. If the directions conflict:

```math
\left\|
\sum_{s=1}^{S+1}
\pi_s a_s
\right\|_2
\ll
1,
```

then the positive gradient can cancel. False positives therefore cause both
semantic error and optimization instability.

## 10. Failure Modes

### 10.1 False Pseudo-Positive

Cosine retrieval is not a label oracle:

```math
\arg\max_q D(z,c_q)
\not=
\arg\max_q
\Pr
\left(
\text{same diagnostic factor}
\mid
z,c_q
\right).
```

Visually similar tissue can have different molecular or prognostic meaning.

### 10.2 Memory Staleness

A bank vector was generated by an earlier target network:

```math
c_q
=
g_{\xi^{(r_q)}}(x_q),
\qquad
r_q<r.
```

The current query uses xi^(r). Momentum reduces representation drift but does
not remove it.

### 10.3 False Negatives

The current minibatch denominator treats different indices as negatives:

```math
i\neq j
\Longrightarrow
\text{negative term}.
```

In pathology:

```math
\text{different patch index}
\not\Longrightarrow
\text{different morphology}.
```

Adjacent tissue, repeated structures, and same-slide regions can be repelled.

### 10.4 Augmentation Erasure

If an augmentation destroys a rare feature:

```math
\text{diagnostic feature changed by }T
\Longrightarrow
\text{feature treated as nuisance}.
```

This is a direct consequence of the view-equivalence assumption.

### 10.5 Local-Context Ceiling

The Swin backbone does not connect independent WSI patches:

```math
\nexists\ \text{cross-patch attention edge in }f_\theta.
```

Long-range tumor-stroma organization must be represented by the downstream
slide operator.

### 10.6 Site Shortcut

If scanner or stain is stable in the pretraining corpus:

```math
\text{site similarity}
\longrightarrow
\text{memory similarity}
\longrightarrow
\text{pseudo-positive alignment}.
```

External-site evaluation and stain perturbation are required before claiming
universal pathology geometry.

## 11. Sanity Checks

### Mined-Support Purity

When audit labels are available, measure:

```math
\mathrm{purity}_S
=
\frac{1}{S}
\sum_{q\in\mathcal P_S(z)}
\mathbf 1
\left\{
\ell(c_q)=\ell(z)
\right\}.
```

This is not part of training. It tests whether the cosine neighborhood is a
reasonable proxy for the intended pathology factor.

### View Destruction

Compare two views that preserve or destroy a small lesion:

```math
\Delta_{\mathrm{view}}
=
\left\|
f_{\theta^\star}(T_1x)
-
f_{\theta^\star}(T_2x)
\right\|_2.
```

Small Delta_view after lesion destruction indicates that the contrastive
contract has erased that lesion from the representation.

### Layout Intervention

For slides with identical patch features and permuted coordinates, a set
readout must satisfy:

```math
\mathcal R_{\mathrm{set}}
\left(
\left\{h_{ij}\right\}_{j=1}^{n_i}
\right)
=
\mathcal R_{\mathrm{set}}
\left(
\left\{h_{i\pi(j)}\right\}_{j=1}^{n_i}
\right).
```

Any layout sensitivity must be added after CTransPath.

## 12. C/R/G/S Placement

| Component | CTransPath and SRCL |
|---|---|
| C context operator | CNN local field, Swin window attention, shifted-window attention, and memory-neighborhood mining |
| R readout operator | projection head during SRCL; downstream MIL or task head for a slide |
| G geometry | normalized cosine space for mining and contrastive logits |
| S surviving statistic | augmentation-invariant local morphology plus a top-S feature-neighborhood relation |

The memory bank belongs to training-time support construction. It does not
become a spatial graph or a slide representation at inference.

The training map is:

```math
x
\xrightarrow{\ T_1,T_2\ }
(x_1,x_2)
\xrightarrow{\ f_\theta,f_\xi\ }
(y_1,\widehat y_2)
\xrightarrow{\ g_\theta,g_\xi\ }
(z_1,\widehat z_2)
\xrightarrow{\ \mathcal M\ }
\mathcal P_S(z)
\xrightarrow{\ \mathcal L_{\mathrm{SRCL}}\ }
\theta.
```

The inference map is:

```math
x_{ij}
\xrightarrow{\ f_{\theta^\star}\ }
h_{ij}
\xrightarrow{\ \mathcal R_{\mathrm{WSI}}\ }
z_i.
```

## 13. Bottom Line

CTransPath is:

```math
\text{CNN-local morphology}
+
\text{shifted-window Swin context}
+
\text{top-S cross-instance positive mining}.
```

The memory bank supplies candidate pseudo-positives; the current minibatch
supplies the contrastive negative support. The method relies on both
augmentation equivalence and cosine-neighborhood semantic validity:

```math
\text{augmentation equivalence}
\qquad\text{and}\qquad
\text{feature-neighborhood validity}.
```

The first can erase focal diagnostic information. The second can introduce
false positives, stale supports, and gradient cancellation. The correct
paper-level statement is therefore not “MoCo with a Swin encoder.” It is a
specific composition of patch context, memory retrieval, and set-valued
contrastive supervision.
