# CONCH And PLIP: Image-Text Geometry And WSI Readout

Primary sources: [CONCH](https://arxiv.org/abs/2307.12914) and
[PLIP](https://pubmed.ncbi.nlm.nih.gov/37592105/).

CONCH and PLIP belong to the pathology vision-language family, but they are
not instances of one identical training objective. Both learn a shared image
and text geometry. CONCH then adds a CoCa-style autoregressive captioning
objective and a multimodal decoder. PLIP fine-tunes a CLIP architecture with
symmetric image-text contrastive learning.

The useful common contract is:

```math
u_i
=
\frac{f_{\mathrm{img}}(x_i)}
{\left\|f_{\mathrm{img}}(x_i)\right\|_2},
\qquad
v_i
=
\frac{f_{\mathrm{text}}(w_i)}
{\left\|f_{\mathrm{text}}(w_i)\right\|_2},
\qquad
s_{ij}
=
u_i^{\mathsf T}v_j.
```

The differences begin in how f_img and f_text are built, what losses constrain
them, how captions are generated, and how a tile-level score becomes a slide
prediction.

## 1. Paired Data As A Statistical Object

For paired image-text data:

```math
\mathcal D
=
\left\{
    (x_i,w_i)
\right\}_{i=1}^{B},
\qquad
x_i\in\mathcal X,
\quad
w_i=(w_{i,1},\ldots,w_{i,T_i}).
```

The image-text pair is not a disease label. It is a noisy semantic relation:

```math
(x_i,w_i)
\sim
P_{\mathrm{pair}}(x,w).
```

The learned representation is shaped by the pair distribution. A caption can
describe diagnosis, morphology, specimen type, stain, educational context, or
an artifact. The contrastive loss does not know which portion of the caption is
causal for the image.

### CONCH Data

CONCH constructs 1.79 million image-text pairs from educational and PubMed
Central Open Access sources, then filters to a human-only pretraining set of
1,170,647 pairs. The source paper reports an effective global batch size of
1536, 448 by 448 image inputs, and 40 epochs.

### PLIP Data

PLIP trains on OpenPath, containing 208,414 image-text pairs:

```math
208{,}414
=
116{,}504_{\mathrm{tweets}}
+
59{,}869_{\mathrm{replies}}
+
32{,}041_{\mathrm{PathLAION}}.
```

The captions originate from pathology-focused medical Twitter posts, replies,
and public image-text data. The source paper reports a median caption length of
17 words and a manual quality audit, while also acknowledging residual noise.

The different pair distributions imply different inherited geometry:

```math
P_{\mathrm{CONCH}}(x,w)
\neq
P_{\mathrm{PLIP}}(x,w)
\Longrightarrow
f_{\mathrm{CONCH}}
\text{ and }
f_{\mathrm{PLIP}}
\text{ need not preserve the same semantics}.
```

## 2. PLIP: CLIP Fine-Tuning On OpenPath

PLIP uses a ViT-B/32 image encoder with 224 by 224 input images and a text
Transformer with maximum sequence length 76 tokens. Images are resized to a
maximum dimension of 512 and randomly cropped to 224 by 224 before encoding.
Both image and text encoders output 512-dimensional vectors.

Write:

```math
h_i^{\mathrm{PLIP}}
=
F_{\theta}(x_i),
\qquad
r_i^{\mathrm{PLIP}}
=
G_{\theta}(h_i)
\in
\mathbb R^{512},
```

and:

```math
t_i^{\mathrm{PLIP}}
=
H_{\phi}(w_i)
\in
\mathbb R^{512}.
```

Normalize both vectors:

```math
u_i
=
\frac{r_i^{\mathrm{PLIP}}}
{\left\|r_i^{\mathrm{PLIP}}\right\|_2},
\qquad
v_i
=
\frac{t_i^{\mathrm{PLIP}}}
{\left\|t_i^{\mathrm{PLIP}}\right\|_2}.
```

The paired similarity matrix is:

```math
S_{ij}
=
u_i^{\mathsf T}v_j
\in[-1,1].
```

With learnable logit scale gamma, the image-to-text loss is:

```math
\mathcal L_{\mathrm{I\to T}}^{\mathrm{PLIP}}
=
-
\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{
\exp(\gamma S_{ii})
}{
\sum_{j=1}^{B}
\exp(\gamma S_{ij})
}.
```

The text-to-image loss is:

```math
\mathcal L_{\mathrm{T\to I}}^{\mathrm{PLIP}}
=
-
\frac{1}{B}
\sum_{j=1}^{B}
\log
\frac{
\exp(\gamma S_{jj})
}{
\sum_{i=1}^{B}
\exp(\gamma S_{ij})
}.
```

The symmetric objective is:

```math
\mathcal L_{\mathrm{PLIP}}
=
\frac{1}{2}
\left(
\mathcal L_{\mathrm{I\to T}}^{\mathrm{PLIP}}
+
\mathcal L_{\mathrm{T\to I}}^{\mathrm{PLIP}}
\right).
```

The diagonal pair is treated as positive and the remaining minibatch pairs as
negatives:

```math
i=j
\Longrightarrow
\text{paired target},
\qquad
i\neq j
\Longrightarrow
\text{in-batch negative}.
```

This creates the usual false-negative risk. Two different OpenPath pairs can
describe the same tissue morphology, while two captions from the same broad
topic can describe different image evidence.

### 2.1 PLIP Zero-Shot Classification

For candidate class text c, define a normalized prompt vector:

```math
v_c
=
\frac{
f_{\mathrm{text}}
\left(
\mathrm{template}(c)
\right)
}{
\left\|
f_{\mathrm{text}}
\left(
\mathrm{template}(c)
\right)
\right\|_2
}.
```

For a tile x:

```math
s_c(x)
=
\gamma
u(x)^{\mathsf T}v_c.
```

The zero-shot prediction is:

```math
\widehat y(x)
=
\arg\max_c
s_c(x).
```

The text vector defines the classifier direction. The score is not automatically
a calibrated pathology probability:

```math
\gamma u(x)^{\mathsf T}v_c
\not\Rightarrow
\Pr(Y=c\mid x).
```

### 2.2 PLIP Linear Probe

For labeled tiles, a frozen PLIP image vector can feed a linear classifier:

```math
\widehat y
=
\mathrm{softmax}
\left(
W u(x)+b
\right).
```

The linear probe changes the readout geometry without changing the pretrained
image embedding:

```math
\nabla_{\theta_{\mathrm{PLIP}}}
\mathcal L_{\mathrm{probe}}
=
0.
```

This is a different transfer question from zero-shot prediction. Zero-shot
classification uses text-defined directions; a linear probe learns directions
from task labels.

## 3. CONCH: CoCa-Style Alignment And Captioning

CONCH uses a ViT-B image backbone with 12 Transformer layers, 12 attention
heads, width 768, MLP hidden width 3072, 16 by 16 image tokens, and learned
absolute positional embeddings.

For a 448 by 448 image, the number of image tokens before the class token is:

```math
N_{\mathrm{img}}
=
\left(
\frac{448}{16}
\right)^2
=
28^2
=
784.
```

Let the final visual feature map be:

```math
H_i
\in
\mathbb R^{784\times768}.
```

### 3.1 Contrastive Pooler

The contrastive pooler uses one learned query q_c:

```math
u_i^{\mathrm{raw}}
=
\mathrm{AttnPool}_{q_c}(H_i)
\in
\mathbb R^{768}.
```

After projection and normalization:

```math
u_i
=
\frac{
W_{\mathrm{contrast}}u_i^{\mathrm{raw}}
}{
\left\|
W_{\mathrm{contrast}}u_i^{\mathrm{raw}}
\right\|_2
}
\in
\mathbb R^{512}.
```

The single learned query produces a global image token for image-text alignment
and retrieval.

### 3.2 Captioning Pooler

The captioning pooler uses 256 learned queries:

```math
C_i
=
\mathrm{AttnPool}_{Q_{\mathrm{caption}}}(H_i)
\in
\mathbb R^{256\times768}.
```

These visual tokens are provided to the multimodal decoder through
cross-attention. The captioning pooler preserves a set of local visual states
rather than one global contrastive vector.

### 3.3 Text Encoder And Decoder

The CONCH text encoder and multimodal decoder are GPT-style causal Transformers
with 12 layers, width 768, and hidden width 3072. The text encoder appends a
learned class token whose output is used for contrastive alignment:

```math
v_i
=
\frac{
W_{\mathrm{text}}
\mathrm{CLS}
\left(
H_{\phi}(w_i)
\right)
}{
\left\|
W_{\mathrm{text}}
\mathrm{CLS}
\left(
H_{\phi}(w_i)
\right)
\right\|_2
}
\in
\mathbb R^{512}.
```

The multimodal decoder predicts the next caption token using causal self
attention and cross-attention to C_i.

## 4. CONCH Pretraining Objective

For a minibatch of B image-caption pairs, the contrastive terms are:

```math
\mathcal L_{\mathrm{I\to T}}^{\mathrm{CONCH}}
=
-
\frac{1}{B}
\sum_{i=1}^{B}
\log
\frac{
\exp(\gamma u_i^{\mathsf T}v_i)
}{
\sum_{j=1}^{B}
\exp(\gamma u_i^{\mathsf T}v_j)
},
```

```math
\mathcal L_{\mathrm{T\to I}}^{\mathrm{CONCH}}
=
-
\frac{1}{B}
\sum_{j=1}^{B}
\log
\frac{
\exp(\gamma v_j^{\mathsf T}u_j)
}{
\sum_{i=1}^{B}
\exp(\gamma v_j^{\mathsf T}u_i)
}.
```

For caption tokens w_i,t, the autoregressive loss is:

```math
\mathcal L_{\mathrm{cap}}
=
-
\frac{1}{B}
\sum_{i=1}^{B}
\sum_{t=1}^{T_i+1}
\log
p_\psi
\left(
w_{i,t}
\mid
w_{i,0:t-1},
x_i
\right).
```

CONCH uses an equal-weighted combination:

```math
\mathcal L_{\mathrm{CONCH}}
=
\frac{1}{2}
\left(
\mathcal L_{\mathrm{I\to T}}^{\mathrm{CONCH}}
+
\mathcal L_{\mathrm{T\to I}}^{\mathrm{CONCH}}
\right)
+
\mathcal L_{\mathrm{cap}}.
```

The contrastive pooler and text class token define the shared retrieval
geometry. The captioning pooler and multimodal decoder define conditional
generation geometry:

```math
u_i
\in
\mathbb R^{512},
\qquad
C_i
\in
\mathbb R^{256\times768}.
```

These are distinct exported objects.

## 5. Prompt Geometry And Ensembling

For candidate concept c and prompt template r:

```math
v_{c,r}
=
\frac{
f_{\mathrm{text}}
\left(
\mathrm{template}_r(c)
\right)
}{
\left\|
f_{\mathrm{text}}
\left(
\mathrm{template}_r(c)
\right)
\right\|_2
}.
```

The prompt-specific tile score is:

```math
s_{i,c,r}
=
\gamma
u_i^{\mathsf T}v_{c,r}.
```

The prompt-specific classifier is:

```math
\widehat c_i
=
\arg\max_c
s_{i,c,r}.
```

Prompt ensembling averages class directions:

```math
\bar v_c
=
\frac{1}{R}
\sum_{r=1}^{R}
v_{c,r},
\qquad
\widetilde v_c
=
\frac{\bar v_c}{\|\bar v_c\|_2}.
```

The ensembled score is:

```math
\bar s_{i,c}
=
\gamma
u_i^{\mathsf T}\widetilde v_c.
```

Different prompt templates can define different classifier directions even when
they are linguistically close:

```math
v_{c,r}
\neq
v_{c,r'}
\Longrightarrow
\widehat c_i^{(r)}
\text{ may differ from }
\widehat c_i^{(r')}.
```

Prompt ensembling reduces variance only when the vectors average toward a stable
semantic direction.

## 6. CONCH WSI Readout: MI-Zero Top-K Pooling

For WSI i, independently embed N_i tiles:

```math
\mathcal T_i
=
\left\{
x_{ij}
:
j=1,\ldots,N_i
\right\},
\qquad
u_{ij}
=
\frac{
f_{\mathrm{img}}(x_{ij})
}{
\left\|f_{\mathrm{img}}(x_{ij})\right\|_2
}.
```

For class c, compute tile scores:

```math
s_{ijc}
=
\gamma
u_{ij}^{\mathsf T}\widetilde v_c.
```

Let the class-specific top-K index set be:

```math
\mathcal I_{i,c}^{(K)}
=
\underset{
A\subseteq\{1,\ldots,N_i\},
|A|=K
}{\arg\max}
\sum_{j\in A}
s_{ijc}.
```

The slide score is:

```math
S_{i,c}^{(K)}
=
\frac{1}{K}
\sum_{j\in\mathcal I_{i,c}^{(K)}}
s_{ijc}.
```

The slide prediction is:

```math
\widehat y_i
=
\arg\max_c
S_{i,c}^{(K)}.
```

The source paper evaluates:

```math
K\in\{1,5,10,50,100\}.
```

This is a top-K order statistic, not mean MIL:

```math
S_{i,c}^{(K)}
\neq
\frac{1}{N_i}
\sum_{j=1}^{N_i}
s_{ijc}.
```

The operator assumes that the slide label is supported by a small number of
high-scoring tiles. It naturally favors sparse positive evidence and ignores
most low-scoring tissue.

## 7. Supervised WSI Transfer

When the image encoder is frozen and slide labels are available, a downstream
ABMIL readout can operate on CONCH tile vectors:

```math
h_{ij}
=
W_{\mathrm{tile}}u_{ij}
b_{\mathrm{tile}}.
```

With gated attention:

```math
e_{ij}
=
w_a^{\mathsf T}
\left[
\tanh(V_a h_{ij})
\odot
\sigma(U_a h_{ij})
\right].
```

```math
\alpha_{ij}
=
\frac{\exp(e_{ij})}{
\sum_{\ell=1}^{N_i}\exp(e_{i\ell})
},
\qquad
z_i
=
\sum_{j=1}^{N_i}
\alpha_{ij}h_{ij}.
```

The full transfer map is:

```math
\widehat y_i
=
g_\omega
\left(
\mathcal R_\psi
\left(
\left\{
f_{\mathrm{img}}(x_{ij})
\right\}_{j=1}^{N_i}
\right)
\right).
```

The contrastive text vector is not part of this supervised WSI readout unless
the downstream model explicitly retains text.

## 8. Retrieval Geometry

For a text query q and image database vectors:

```math
v_q
=
\frac{f_{\mathrm{text}}(q)}
{\|f_{\mathrm{text}}(q)\|_2},
\qquad
\mathrm{score}(q,i)
=
u_i^{\mathsf T}v_q.
```

The top-K retrieval set is:

```math
\mathcal R_K(q)
=
\underset{
A\subseteq\{1,\ldots,M\},
|A|=K
}{\arg\max}
\sum_{i\in A}
u_i^{\mathsf T}v_q.
```

Recall@K evaluates whether the correct paired item appears in this set. A
retrieval result measures the learned data geometry plus query wording; it is
not a proof of clinical equivalence.

## 9. CONCH And PLIP Comparison

| Property | PLIP | CONCH |
|---|---|---|
| source data | 208,414 OpenPath image-text pairs | 1,170,647 human-only image-caption pairs |
| image backbone | ViT-B/32, 224 by 224 input | ViT-B/16, 448 by 448 pretraining input |
| shared vector | 512-dimensional image and text vectors | 512-dimensional one-query image vector and text CLS |
| alignment | symmetric CLIP-style contrastive loss | symmetric contrastive loss |
| generation | absent from the pretraining objective | 256-query image pooler and autoregressive captioning |
| WSI operator in source paper | tile-level image-text and transfer geometry | MI-Zero top-K tile pooling and ABMIL transfer |

The comparison is not a ranking. It identifies which operators exist in the
source training graphs.

## 10. Failure Modes

### 10.1 Prompt Instability

If prompt vectors differ:

```math
v_{c,r}
\neq
v_{c,r'}
\Longrightarrow
\arg\max_c u^{\mathsf T}v_{c,r}
\text{ may differ from }
\arg\max_c u^{\mathsf T}v_{c,r'}.
```

Prompt sensitivity should be reported, not hidden behind one favorable
template.

### 10.2 Top-K Prevalence Blindness

Two slides with identical top-K scores but different prevalence obtain the same
score:

```math
\left\{
s_{ijc}:j\in\mathcal I_{i,c}^{(K)}
\right\}
=
\left\{
s_{i'jc}:j\in\mathcal I_{i',c}^{(K)}
\right\}
\Longrightarrow
S_{i,c}^{(K)}
=
S_{i',c}^{(K)}.
```

Top-K is appropriate for sparse evidence and inappropriate when burden or area
is the target statistic.

### 10.3 Captions Are Not Dense Labels

The caption objective models:

```math
\Pr
\left(
w_t
\mid
w_{<t},
x
\right),
```

not a pixel-level annotation. A frequent caption phrase can dominate language
geometry while referring only coarsely to the image.

### 10.4 CONCH Pooler Confusion

The contrastive and captioning poolers are different:

```math
u_i
=
\mathrm{AttnPool}_{1}(H_i),
\qquad
C_i
=
\mathrm{AttnPool}_{256}(H_i).
```

Using captioning tokens as if they were the retrieval vector changes the model
contract.

### 10.5 In-Batch False Negatives

In the contrastive denominator:

```math
i\neq j
\Longrightarrow
\exp
\left(
\gamma u_i^{\mathsf T}v_j
\right)
\text{ enters the negative support}.
```

Repeated morphology or duplicate educational examples can be repelled.

### 10.6 Tile Independence

MI-Zero computes:

```math
s_{ijc}
=
\gamma
u_{ij}^{\mathsf T}\widetilde v_c
```

independently for each tile. No tile-to-tile context enters before top-K
pooling. A spatially organized pattern can be lost if its tiles are not
independently close to the text direction.

### 10.7 Pair Distribution Shift

Training and evaluation pair distributions differ:

```math
P_{\mathrm{train}}(x,w)
\neq
P_{\mathrm{test}}(x,w)
\Longrightarrow
\text{zero-shot score calibration can shift}.
```

The existence of a text interface does not guarantee clinically calibrated
semantics for an unseen phrase.

## 11. Sanity Checks

### Prompt Dispersion

For class c and R prompts, measure:

```math
\mathrm{Disp}(c)
=
\frac{1}{R}
\sum_{r=1}^{R}
\left\|
v_{c,r}
-
\widetilde v_c
\right\|_2^2.
```

Large dispersion predicts prompt-sensitive decisions.

### K Sensitivity

Evaluate:

```math
K
\longmapsto
S_{i,c}^{(K)}
```

for K in the source set. A prediction that changes substantially with K relies
on a sparse-evidence assumption.

### Spatial Permutation

Permute tile coordinates while preserving tile vectors:

```math
\left\{
(u_{ij},c_{ij})
\right\}_{j=1}^{N_i}
\longrightarrow
\left\{
(u_{ij},c_{i\pi(j)})
\right\}_{j=1}^{N_i}.
```

MI-Zero top-K pooling should be unchanged. A coordinate-aware WSI model should
not be expected to remain unchanged.

### Caption Ablation

Remove the captioning term:

```math
\mathcal L_{\mathrm{CONCH}}^{\mathrm{no\ cap}}
=
\frac{1}{2}
\left(
\mathcal L_{\mathrm{I\to T}}
+
\mathcal L_{\mathrm{T\to I}}
\right).
```

The difference between this ablation and full CONCH separates alignment gains
from autoregressive generation gains.

## 12. C/R/G/S Placement

| Component | PLIP | CONCH |
|---|---|---|
| C context operator | ViT-B/32 image tokenization and text Transformer context | ViT-B/16 image context, learned-query pooling, text Transformer, and decoder cross-attention |
| R readout operator | 512-dimensional normalized image-text vector; downstream linear or WSI readout | one-query 512-dimensional retrieval vector, 256-query caption token set, or downstream WSI readout |
| G geometry | symmetric cosine image-text space | symmetric cosine image-text space plus conditional decoder geometry |
| S surviving statistic | pathology-image and caption alignment under OpenPath | global image-text alignment, local caption tokens, and top-K tile evidence |

The common image-text vector is one part of CONCH. Treating CONCH as PLIP with
more data misses the decoder and the second attentional pooler.

## 13. Bottom Line

PLIP learns:

```math
\text{OpenPath image-text pairs}
\longrightarrow
\text{ViT-B/32 and text Transformer}
\longrightarrow
\text{symmetric 512-dimensional cosine geometry}.
```

CONCH learns:

```math
\text{human-only pathology captions}
\longrightarrow
\left\{
\begin{aligned}
\text{one-query contrastive image token},\\
\text{256-query caption image tokens}
\end{aligned}
\right\}
\longrightarrow
\text{alignment plus autoregressive captioning}.
```

For CONCH WSI zero-shot classification:

```math
\text{tile image-text scores}
\longrightarrow
\text{class-specific top-K pooling}
\longrightarrow
\text{slide prediction}.
```

That operator preserves high-scoring sparse evidence and discards prevalence,
coordinate arrangement, and low-scoring tile interactions. The correct
mathematical question is:

```math
\text{Which semantic direction does the text prompt define,}
\quad
\text{and which tile statistic does the WSI readout preserve?}
```

The encoder supplies cross-modal geometry. The top-K or MIL readout still
decides what a whole slide means.
