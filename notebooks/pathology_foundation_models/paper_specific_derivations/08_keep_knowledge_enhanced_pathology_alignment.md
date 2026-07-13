# KEEP: Knowledge-Enhanced Pathology Alignment

Reference: [KEEP](https://arxiv.org/html/2412.13126v1).

KEEP does not merely append a knowledge string to each caption. It uses a
disease graph in three distinct places: knowledge-encoder pretraining,
semantic-group construction, and group-level vision-language alignment.

## 1. Disease Knowledge Graph

Let the disease graph be:

```math
\mathcal G_{\mathrm{disease}}
=
\left(
\mathcal D,
\mathcal E_{\mathrm{hypernym}},
\mathcal A
\right),
```

where `D` is the disease-entity set, `E_hypernym` contains parent-child
relations, and `A_i` contains synonyms, definitions, and disease-chain text
for entity `i`.

For attribute `a_{ip}` of entity `i`, the BERT-based knowledge encoder gives a
normalized representation:

```math
u_{ip}
=
\frac{
\Phi_k(a_{ip})
}{
\left\lVert
\Phi_k(a_{ip})
\right\rVert_2
}.
```

The knowledge encoder is trained before vision-language alignment so that
attributes of the same disease are close while attributes from different
diseases are separated.

## 2. Disease-Attribute Metric Loss

For disease entity `i`, the hard positive similarity is the max-min quantity:

```math
S_i^{+}
=
\max_{p}
\min_{q\neq p}
u_{ip}^{\top}u_{iq},
```

and the hardest negative similarity is:

```math
S_i^{-}
=
\max_{j\neq i}
\max_{p,q}
u_{ip}^{\top}u_{jq}.
```

The paper uses smooth approximations to these max/min operators. The resulting
AdaSP-style metric loss is:

```math
\mathcal L_{\mathrm{KG}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\log
\left(
1
+
\exp
\left(
\frac{S_i^{-}-S_i^{+}}{\tau_{\mathrm{KG}}}
\right)
\right).
```

The target is the ordering `S_i^{+} > S_i^{-}`. This is not ordinary CLIP
classification over one caption per image.

## 3. Knowledge-Guided Semantic Groups

After image curation and caption refinement, KEEP forms semantic groups. Let
group `i` contain `M` sampled image-caption pairs, with sampling with
replacement:

```math
\mathcal B_i
=
\left\{
\left(x_{i1},c_{i1}\right),
\ldots,
\left(x_{iM},c_{iM}\right)
\right\}.
```

The normalized visual and knowledge-text embeddings are:

```math
v_{ik}
=
\frac{\Phi_v(x_{ik})}{\left\lVert\Phi_v(x_{ik})\right\rVert_2},
\qquad
t_{im}
=
\frac{\Phi_k(c_{im})}{\left\lVert\Phi_k(c_{im})\right\rVert_2}.
```

Each group contributes an `M` by `M` positive similarity block:

```math
\mathcal S_{ii}
=
\left[
t_{im}^{\top}v_{ik}
\right]_{m,k=1}^{M}.
```

The positive statistic is the least-hard positive, not the hardest individual
pair:

```math
S_i^{+}
=
\max_{k}
\min_{m}
t_{im}^{\top}v_{ik}.
```

The outer maximum chooses the image whose worst caption match is strongest.
This is a compromise between averaging all positives and selecting one noisy
hard positive.

## 4. Hypernym-Aware Negative Geometry

Let `I_{ij}` indicate whether semantic group `j` is a valid negative for group
`i`. KEEP sets this indicator to zero when groups share a disease label or are
connected through the relevant hypernym paths. Define:

```math
\mathcal N_i
=
\left\{
(j,m):
j\neq i,
I_{ij}=1,
1\le m\le M
\right\}.
```

The hard negative statistic is:

```math
S_i^{-}
=
\max_{k}
\max_{(j,m)\in\mathcal N_i}
t_{jm}^{\top}v_{ik}.
```

The semantic-group loss is:

```math
\mathcal L_{\mathrm{sem}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\log
\left(
1
+
\exp
\left(
\frac{S_i^{-}-S_i^{+}}{\tau_{\mathrm{sem}}}
\right)
\right).
```

The group relation is positive over multiple images and captions, while the
disease hierarchy removes some nominal negatives.

## 5. Combined Training Map

The knowledge encoder is initialized by the disease metric-learning stage; the
visual encoder is initialized by UNI; and the vision-language stage uses the
semantic-group loss above. Abstractly:

```math
\mathcal L_{\mathrm{KEEP}}
=
\mathcal L_{\mathrm{sem}}
\quad
\text{with}
\quad
\Phi_k
\text{ initialized by }
\mathcal L_{\mathrm{KG}}.
```

The paper reports a ViT-L/16 visual encoder, a PubMedBERT-based text encoder,
batch size `128`, `32` semantic groups, `4` image-caption pairs per group,
temperature `0.04`, and `10` vision-language pretraining epochs. These are
reported training choices, not consequences of the loss family.

## 6. Knowledge Changes the Prior

The graph and curated groups change the effective text-side prior:

```math
P(H\mid T,\mathcal G_{\mathrm{disease}})
\neq
P(H\mid T).
```

This can improve rare-concept coverage, but an incomplete graph, an incorrect
hypernym edge, or a noisy caption can impose a misleading alignment direction.
KEEP therefore changes both the representation objective and the sampling law;
its gain cannot be attributed to the text encoder alone.

## 7. Tile-Level And WSI Readout

At inference, a tile `x_{ij}` is compared with prompt embedding `t_c`:

```math
q_{ij}(c)
=
\frac{
\exp
\left(
\Phi_v(x_{ij})^{\top}t_c/\tau
\right)
}{
\sum_{a=1}^{C}
\exp
\left(
\Phi_v(x_{ij})^{\top}t_a/\tau
\right)
}.
```

The paper's tumor-ratio readout assigns a slide-level class by tile votes. A
simple form is:

```math
\widehat r_i(c)
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}
\mathbf 1
\left[
\arg\max_a q_{ij}(a)=c
\right].
```

This readout assumes tiles are independent and equally informative, assumes no
class prior, and relies on large `n_i` for a law-of-large-numbers
approximation. Those assumptions are not properties of the KEEP encoder. The
zero-shot WSI prediction is a patch classifier plus a ratio readout, not a
learned spatial slide encoder.

## 8. Evaluation

Knowledge-enhanced zero-shot performance should be tested against plain text
prompts, paraphrases, held-out diseases, and external institutions. Higher
accuracy alone does not show that the learned explanations are medically valid.

The decisive ablations are:

```text
remove the knowledge metric pretraining;
remove semantic grouping;
allow hypernym-related groups as negatives;
replace least-hard positives with ordinary pairwise CLIP;
replace tumor-ratio readout with a learned slide aggregator.
```

These isolate knowledge, group geometry, false-negative handling, and the
downstream aggregation claim.

## C/R/G/S Placement

```text
C:
    BERT knowledge encoder plus image and text encoders; no learned spatial WSI
    context in the zero-shot tile-to-slide readout

R:
    normalized image-text similarity followed by tile votes or tumor ratio

G:
    disease-hierarchy text geometry and group-block image-text geometry

S:
    disease attributes, curated semantic groups, hypernym-aware negatives, and
    tile-level prompts at inference
```
