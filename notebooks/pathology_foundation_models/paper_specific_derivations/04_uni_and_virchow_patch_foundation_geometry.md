# UNI And Virchow: Patch Foundation Geometry

Primary sources: [UNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/),
[Virchow](https://arxiv.org/abs/2309.07778), and
[DINOv2](https://arxiv.org/abs/2304.07193).

This note isolates the mathematical object exported by two pathology patch
encoders. It does not treat a patch foundation model as a slide model. A
whole-slide prediction still requires a separate readout over the exported
patch vectors.

## 1. The Transfer Contract

Represent slide `i` by tissue-containing patches and their coordinates:

```math
\mathcal S_i
=
\left\{
    (x_{ij},c_{ij})
    :
    j=1,\ldots,n_i
\right\},
\qquad
x_{ij}\in\mathbb R^{H\times W\times 3},
\quad
c_{ij}\in\mathbb R^2.
```

A patch encoder produces one vector per patch:

```math
h_{ij}
=
f_\phi(x_{ij})
\in
\mathbb R^d.
```

The coordinate does not enter this map unless it is explicitly appended to the
image or supplied to a coordinate-aware encoder. The usual frozen-feature WSI
pipeline is therefore:

```math
\left\{
    (x_{ij},c_{ij})
\right\}_{j=1}^{n_i}
\xrightarrow{\ f_\phi\ }
\left\{
    (h_{ij},c_{ij})
\right\}_{j=1}^{n_i}
\xrightarrow{\ \mathcal R_\psi\ }
z_i
\xrightarrow{\ g_\omega\ }
\widehat y_i.
```

The factorization makes the transfer boundary explicit:

```math
\widehat y_i
=
g_\omega
\left(
    \mathcal R_\psi
    \left(
        \left\{
            (f_\phi(x_{ij}),c_{ij})
        \right\}_{j=1}^{n_i}
    \right)
\right).
```

The patch pretraining objective constrains `f_phi`; it does not, by itself,
select whether `R_psi` is mean pooling, attention, a graph operator, or a
hierarchical readout.

## 2. UNI: Broad Patch Geometry

The UNI paper trains a ViT-L/16 encoder on Mass-100K: more than 100 million
tissue patches from 100,426 diagnostic H&E WSIs spanning 20 major tissue types.
The paper's model is the original UNI ViT-L/16 model. Its public repository
later also exposes a UNI-2 ViT-G/14 configuration; the latter must not be used
as the dimension of the original UNI feature.

For a 224 by 224 crop and a 16 by 16 image patch, tokenization gives:

```math
N_{\mathrm{UNI}}
=
\left(\frac{224}{16}\right)^2
=
14^2
=
196
```

patch tokens. If `u_0` is the class token and `u_1,...,u_196` are the final
hidden tokens of the ViT-L encoder, the exported original-UNI feature is the
class-token representation:

```math
h^{\mathrm{UNI}}(x)
=
u_0
\in
\mathbb R^{1024}.
```

The 1024-dimensional value is the original UNI ViT-L/16 output. A current
repository configuration with

```math
\mathrm{vit\_giant\_patch14\_224}
\quad\text{and}\quad
d=1536
```

refers to UNI-2 and is a different encoder. Confusing the two changes the
feature space, downstream parameter count, and any claimed reproducibility.

### 2.1 The DINOv2 Contract

UNI uses the DINOv2 training algorithm. The relevant contract is not a single
supervised class label; it is consistency between teacher outputs on global
views and student outputs on global and local views, together with masked image
modeling and regularization terms from the DINOv2 recipe.

Let `V_g(x)` be the set of global crops and `V_l(x)` the set of local crops.
For a teacher view `u` and student view `v`, define centered and sharpened
teacher coordinates and student coordinates:

```math
q_t^{u}
=
\mathrm{softmax}
\left(
    \frac{g_t^{\mathrm{img}}(u)-c_t}{\tau_t}
\right),
\qquad
p_s^{v}
=
\mathrm{softmax}
\left(
    \frac{g_s^{\mathrm{img}}(v)}{\tau_s}
\right).
```

The image-level self-distillation term is:

```math
\mathcal L_{\mathrm{img}}
=
-
\frac{1}{|V_g(x)|}
\sum_{u\in V_g(x)}
\frac{1}{|V_g(x)|+|V_l(x)|-1}
\sum_{\substack{v\in V_g(x)\cup V_l(x)\\v\neq u}}
\sum_{k=1}^{K}
q_{t,k}^{u}
\log p_{s,k}^{v}.
```

The target center is updated from teacher outputs across the distributed batch,
and the target encoder is updated by exponential moving average rather than by
backpropagation:

```math
\theta_t^{(r)}
=
m_r\theta_t^{(r-1)}
+
\left(1-m_r\right)\theta_s^{(r)}.
```

The masked image-modeling branch supplies a patch-token objective. If `M` is a
set of masked token positions and `y_j` is the target patch token at position
`j`, a schematic form is:

```math
\mathcal L_{\mathrm{patch}}
=
-
\frac{1}{|M|}
\sum_{j\in M}
\sum_{k=1}^{K}
q_{t,j,k}
\log p_{s,j,k}.
```

The DINOv2 family objective can therefore be represented as:

```math
\mathcal L_{\mathrm{DINOv2}}
=
\mathcal L_{\mathrm{img}}
+
\lambda_{\mathrm{patch}}
\mathcal L_{\mathrm{patch}}
+
\lambda_{\mathrm{KoLeo}}
\mathcal L_{\mathrm{KoLeo}}.
```

This equation is a decomposition of the DINOv2 training contract, not a claim
that the UNI paper introduced a new UNI-specific loss. The important
representation consequence is an invariance pressure:

```math
\mathbb E_{T_1,T_2\sim K(\cdot\mid x)}
\left[
    \left\|
        f_\phi(T_1x)-f_\phi(T_2x)
    \right\|_2^2
\right]
\quad\text{is pushed downward},
```

where `K` is the crop, resize, color, and masking view kernel. Information that
is systematically removed or made equivalent by this kernel is not recovered
by adding more slides:

```math
\text{scale increases information about the training distribution}
\not\Rightarrow
\text{scale reverses an imposed view invariance}.
```

### 2.2 UNI's Patch-to-Slide Baseline

The UNI paper evaluates the exported features with downstream methods including
attention-based MIL. For a frozen patch representation, the standard attention
readout has:

```math
a_{ij}
=
\frac{
\exp
\left(
    w_a^{\mathsf T}
    \tanh
    \left(V_a h_{ij}^{\mathrm{UNI}}+b_a\right)
\right)
}{
\sum_{\ell=1}^{n_i}
\exp
\left(
    w_a^{\mathsf T}
    \tanh
    \left(V_a h_{i\ell}^{\mathrm{UNI}}+b_a\right)
\right)
},
\qquad
\sum_{j=1}^{n_i}a_{ij}=1.
```

```math
z_i^{\mathrm{ABMIL}}
=
\sum_{j=1}^{n_i}
a_{ij}h_{ij}^{\mathrm{UNI}}.
```

This factorization is useful for interpreting the paper's result: an improved
patch geometry can make a simple ABMIL head competitive with more elaborate
slide architectures. It does not show that the attention operator is
unnecessary for every encoder or every task.

UNI also evaluates few-shot prototype-style slide classification. Given class
support patches `H_c`, the class prototype is:

```math
\mu_c
=
\frac{1}{|H_c|}
\sum_{h\in H_c}h,
\qquad
\widetilde\mu_c
=
\frac{\mu_c}{\|\mu_c\|_2}.
```

For a query patch, cosine assignment is:

```math
\widehat c(h)
=
\arg\max_c
\frac{h^{\mathsf T}\widetilde\mu_c}{\|h\|_2}.
```

A slide-level nearest-patch score using the top `K` query patches is:

```math
R_{i,c}^{(K)}
=
\frac{1}{K}
\sum_{j\in\mathrm{TopK}_c(i)}
\frac{
    h_{ij}^{\mathsf T}\widetilde\mu_c
}{
    \|h_{ij}\|_2
},
\qquad
\widehat y_i
=
\arg\max_c R_{i,c}^{(K)}.
```

The operation preserves evidence from a small number of high-similarity
patches. It is not equivalent to ABMIL: the prototype is estimated from a
support set, while attention learns a task-specific score function.

## 3. Virchow: Correlated WSI Sampling And Token Concatenation

Virchow is a 632-million-parameter ViT-H pathology model trained with the
DINOv2 recipe on approximately 1.5 million WSIs from 119,629 patients and 17
high-level tissue types. It uses 224 by 224 inputs and a 14 by 14 image patch,
so each image contains 256 patch tokens:

```math
N_{\mathrm{Virchow}}
=
\left(\frac{224}{14}\right)^2
=
16^2
=
256.
```

The final hidden state has width 1280. Write the final class token as `c(x)`
and the patch tokens as `p_1(x),...,p_256(x)`. Virchow exports the concatenated
statistic:

```math
e^{\mathrm{Virchow}}(x)
=
\left[
    c(x)
    \,;
    \overline p(x)
\right],
\qquad
\overline p(x)
=
\frac{1}{256}
\sum_{r=1}^{256}p_r(x),
\qquad
e^{\mathrm{Virchow}}(x)\in\mathbb R^{2560}.
```

The exported vector is therefore not just a class token and not a flat mean of
the whole WSI. It contains two patch-level statistics from one input crop: a
learned global summary and the first moment of the local token field.

### 3.1 The Sampling Distribution Is Part Of The Model

Virchow samples one WSI per GPU and 256 foreground tiles from that WSI. Let
`S_g` be the slide assigned to GPU `g` and let `X_{g,r}` be its `r`-th sampled
tile:

```math
S_g
\sim
P_{\mathrm{slide}},
\qquad
X_{g,1},\ldots,X_{g,256}
\sim
P(\cdot\mid S_g).
```

The correct joint law is:

```math
P(X_{g,1:256})
=
\int
\prod_{r=1}^{256}
P(X_{g,r}\mid S_g=s)
\,dP_{\mathrm{slide}}(s),
```

under conditional independence given the sampled slide. Its pairwise covariance
contains a within-slide term:

```math
\mathrm{Cov}(X_{g,r},X_{g,r'})
=
\mathrm{Cov}_{S_g}
\left(
    \mathbb E[X_{g,r}\mid S_g]
\right)
\quad
\text{for }r\neq r'.
```

This is different from drawing 256 independent tiles from the marginal mixture
over all WSIs. The batch can therefore contain strong slide-level correlation,
which changes the statistics seen by centering, prototype assignment, and
feature-spacing regularization.

For the mean of the 256 sampled tiles, the conditional variance is:

```math
\mathrm{Var}
\left(
    \frac{1}{256}
    \sum_{r=1}^{256}X_{g,r}
\right)
=
\frac{1}{256}
\Sigma_{\mathrm{within}}
+
\frac{255}{256}
\Sigma_{\mathrm{between}},
```

where `Sigma_within` is the average conditional covariance and
`Sigma_between` is the covariance of slide-specific means. The second term
does not vanish as quickly as the iid formula would suggest when the sampled
tiles share a slide. This is the mathematical reason that “256 tiles per GPU”
is not merely a memory-management detail.

### 3.2 Prototype Coordinates And Exported Geometry

Virchow uses 131,072 prototype coordinates in its DINOv2-style heads:

```math
K_{\mathrm{Virchow}}
=
131{,}072.
```

For a teacher output `z_t` and student output `z_s`, the categorical target and
prediction have the form:

```math
q_t
=
\mathrm{softmax}
\left(
    \frac{z_t-c_t}{\tau_t}
\right),
\qquad
p_s
=
\mathrm{softmax}
\left(
    \frac{z_s}{\tau_s}
\right),
\qquad
q_t,p_s\in\Delta^{131072-1}.
```

The loss is cross entropy in prototype space:

```math
\mathcal L_{\mathrm{proto}}
=
-
\sum_{k=1}^{131072}
q_{t,k}\log p_{s,k}.
```

The prototype coordinates are training-time comparison coordinates. They are
not the 2560 coordinates exported for downstream WSI modeling. A downstream
user who treats the prototype index as a semantic class is adding an
interpretation that the objective does not provide.

Virchow's reported teacher-temperature schedule can be written as:

```math
\tau_t(r)
=
0.04
+
0.03
\min
\left(
    1,
    \frac{r}{186000}
\right),
```

with a reported learning-rate warmup extending to 495,000 iterations. A larger
teacher temperature spreads the target mass over more prototype coordinates;
for a target distribution with entropy `H(q_t)`, the gradient with respect to a
student logit is:

```math
\frac{\partial\mathcal L_{\mathrm{proto}}}{\partial z_{s,k}}
=
\frac{1}{\tau_s}
\left(
    p_{s,k}-q_{t,k}
\right).
```

Temperature and centering therefore alter the optimization geometry even
though the exported embedding dimension remains 2560.

## 4. Why The Virchow Concatenation Matters

Let two patches have exported embeddings:

```math
e_a
=
[c_a;\overline p_a],
\qquad
e_b
=
[c_b;\overline p_b].
```

Their cosine similarity is:

```math
\mathrm{cos}(e_a,e_b)
=
\frac{
    c_a^{\mathsf T}c_b
    +
    \overline p_a^{\mathsf T}\overline p_b
}{
    \sqrt{\|c_a\|_2^2+\|\overline p_a\|_2^2}
    \sqrt{\|c_b\|_2^2+\|\overline p_b\|_2^2}
}.
```

The two blocks contribute according to their norms. Concatenation is not the
same as an equal-weight average of class-token and patch-token similarities
unless both blocks are separately normalized and explicitly reweighted.

If a downstream model first applies a linear map `W`, it can change this block
balance:

```math
W[c;\overline p]
=
W_c c
+
W_p\overline p.
```

Consequently, a linear probe, cosine retrieval head, and attention MIL head do
not see identical Virchow geometry. The normalization convention belongs in
the experimental specification.

## 5. Slide Readout After Either Encoder

For encoder `b` in `{UNI, Virchow}`, define:

```math
h_{ij}^{(b)}
=
f_{\phi_b}(x_{ij}).
```

Mean pooling preserves the empirical first moment:

```math
z_i^{\mathrm{mean},(b)}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
h_{ij}^{(b)}.
```

Attention pooling preserves a learned weighted first moment:

```math
z_i^{\mathrm{attn},(b)}
=
\sum_{j=1}^{n_i}
\alpha_{ij}^{(b)}h_{ij}^{(b)},
\qquad
\alpha_{ij}^{(b)}
\ge 0,
\qquad
\sum_{j=1}^{n_i}\alpha_{ij}^{(b)}=1.
```

Neither readout is spatially aware unless coordinates enter the score or the
context operator. A coordinate-aware version is:

```math
\alpha_{ij}^{(b)}
=
\frac{
\exp
\left(
    a_\psi(h_{ij}^{(b)},c_{ij})
\right)
}{
\sum_{\ell=1}^{n_i}
\exp
\left(
    a_\psi(h_{i\ell}^{(b)},c_{i\ell})
\right)
}.
```

The same patch vectors can therefore produce different slide objects under
different context and readout choices:

```math
\mathcal S_i
\xrightarrow{\ f_{\phi_b}\ }
H_i^{(b)}
\xrightarrow[
    \text{coordinates optional}
]{\ \mathcal C_\psi\ }
\xrightarrow{\ \mathcal R_\psi\ }
z_i.
```

The foundation model contributes a patch geometry; it does not remove the
aggregation design problem.

## 6. What Information Survives

UNI's original export is a class-token statistic of a 224 by 224 crop. Virchow's
export is a class token concatenated with a first moment over 256 patch tokens.
For a slide containing `n_i` patch images, both models still expose only the
collection of patch vectors to the downstream readout:

```math
\mathcal H_i^{(b)}
=
\left\{
    h_{ij}^{(b)}
\right\}_{j=1}^{n_i}.
```

If two slides have the same empirical measure in patch-feature space, every
permutation-invariant readout over those features gives the same output:

```math
\widehat\mu_i^{(b)}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\delta_{h_{ij}^{(b)}},
\qquad
\widehat\mu_i^{(b)}=\widehat\mu_{i'}^{(b)}
\Longrightarrow
\mathcal R(\mathcal H_i^{(b)})
=
\mathcal R(\mathcal H_{i'}^{(b)}).
```

Coordinates, neighborhood relations, and region boundaries survive only if the
downstream operator receives them or if the feature extractor itself was given
that context. A stronger patch encoder cannot reconstruct layout that was never
encoded in the input to the readout.

## 7. C/R/G/S Placement

For both encoders, the decomposition is:

| Component | UNI | Virchow |
|---|---|---|
| `C` context operator | crop/mask/view equivalence inside DINOv2; no WSI adjacency in the exported patch | crop/mask/view equivalence plus within-slide sampling correlations during training; no WSI adjacency in one exported patch |
| `R` readout operator | ViT class token for one crop; downstream ABMIL or prototype readout for a slide | class token concatenated with mean patch token for one crop; downstream slide readout for a WSI |
| `G` geometry | original UNI feature space in `R^1024`; cosine, linear, or learned task geometry after transfer | concatenated `R^2560` space; block norms and normalization affect cosine and probe geometry |
| `S` surviving statistic | crop-invariant patch morphology plus the class-token summary | crop-invariant morphology, global token, and first moment of local tokens |

The key comparison is not “which model has more dimensions.” It is:

```math
\begin{aligned}
\mathrm{UNI:}
&\quad
\text{broad patch distribution}
\longrightarrow
\text{class-token geometry},
\\
\mathrm{Virchow:}
&\quad
\text{correlated WSI sampling}
\longrightarrow
\text{class token plus local first moment}.
\end{aligned}
```

## 8. Failure Modes And Sanity Checks

### 8.1 Confusing UNI With UNI-2

The original paper's ViT-L/16 feature and the later repository's ViT-G/14
feature are different objects:

```math
d_{\mathrm{UNI\text{-}L}}
=
1024
\neq
1536
=
d_{\mathrm{UNI\text{-}2}}.
```

Loading one checkpoint and reporting the other model's dimension invalidates
feature-shape, parameter-count, and replication claims.

### 8.2 Patch Representation Is Not Slide Representation

Two slides can have identical patch-feature histograms but different spatial
layouts. A set readout cannot distinguish them:

```math
\widehat\mu_A^{(b)}
=
\widehat\mu_B^{(b)},
\qquad
\text{layout}(A)\neq\text{layout}(B)
\Longrightarrow
\mathcal R_{\mathrm{set}}(A)
=
\mathcal R_{\mathrm{set}}(B).
```

This is not a defect in UNI or Virchow alone. It is a missing geometry channel
in the downstream readout.

### 8.3 The 256-Tile Mean Is Not An IID Estimate

For correlated tiles from one slide, the variance of the sampled mean contains
the between-slide term shown above. Treating 256 tiles as 256 independent
observations can overstate effective sample size and understate uncertainty.

### 8.4 Concatenation Can Hide A Block Imbalance

If the class-token block has much larger norm than the mean-patch block, cosine
retrieval is dominated by the class token. If a linear probe learns a larger
operator norm on the patch block, the opposite can happen:

```math
\|W_c c\|_2
\gg
\|W_p\overline p\|_2
\quad\text{or}\quad
\|W_p\overline p\|_2
\gg
\|W_c c\|_2.
```

A serious transfer report should state block normalization, probe training, and
whether the vectors were centered or whitened.

### 8.5 Rare-Region Coverage

Let `A` be a rare diagnostic event and `q_b(A)` its probability under the patch
sampling distribution used to build encoder `b`'s training set. The chance that
`m` sampled training patches contain no instance of `A` is:

```math
\Pr\left(N_A=0\right)
=
\left(1-q_b(A)\right)^m.
```

For small `q_b(A)`, increasing `m` helps, but only if the event is actually
represented in the sampling distribution. Downstream attention cannot recover
a morphology absent from the exported patch support.

## 9. Bottom Line

UNI and Virchow should be placed in the same patch-foundation family but not
treated as interchangeable feature maps. The original UNI object is a
1024-dimensional ViT-L/16 class-token representation learned from a broad
multi-tissue patch distribution. Virchow exports a 2560-dimensional
concatenation of a ViT-H/14 class token and the mean of 256 local patch tokens,
while its training sampler introduces explicit within-slide dependence.

The mathematical transfer question is therefore:

```math
\text{Which invariances and patch statistics does }f_\phi\text{ preserve?}
\quad\text{and}
\quad
\text{which slide statistics does }\mathcal R_\psi\text{ preserve?}
```

The answers are separate. Foundation-model scale changes the first map; it does
not eliminate the second design problem.
