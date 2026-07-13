# GECKO Aggregation, Alignment, and C/R/G/S

## 1. Deep Branch

GECKO projects patch features and obtains softmax attention:

```math
\widetilde f_j=H(f_j),
\qquad
\alpha_j=A_p(\widetilde f_j),
\qquad
\sum_j\alpha_j=1.
```

The deep WSI embedding is

```math
F_{\mathrm{WSI}}
=\sum_{j=1}^N\alpha_j\widetilde f_j.
```

## 2. Concept Branch

A differentiable perturbed top-K operator selects rows of the concept matrix
using deep-branch patch attention, producing

```math
\widetilde M\in\mathbb R^{K\times C}.
```

An MLP-Mixer and gated network compute concept gates

```math
\beta=G(\mathrm{Mixer}(\widetilde M^{\top}))
\in(0,1)^C.
```

The matrix is rescaled coordinatewise and mean pooled:

```math
\widehat M_{jc}=\beta_c\widetilde M_{jc},
\qquad
M_{\mathrm{WSI},c}
=\frac1K\sum_{j=1}^K\widehat M_{jc}.
```

The final coordinate preserves a named concept axis, but its gate depends
nonlinearly on the entire selected concept matrix. A large coordinate is gated
concept activation, not an additive class contribution.

## 3. Contrastive Alignment

After projecting both branches to a shared space, GECKO uses a symmetric CLIP
loss. For batch size `B`,

```math
\mathcal L_{a\to b}
=-\frac1B\sum_{i=1}^B
\log
\frac{\exp(\mathrm{sim}(a_i,b_i)/\tau)}
{\sum_{r=1}^B\exp(\mathrm{sim}(a_i,b_r)/\tau)},
```

```math
\mathcal L
=\frac12
(\mathcal L_{F\to M}+\mathcal L_{M\to F}).
```

Alignment makes paired slide embeddings distinguishable from batch negatives.
It does not force every deep dimension to correspond one-to-one with a concept
coordinate, nor guarantee that all batch negatives are clinically dissimilar.

## 4. C/R/G/S Placement

| Method | Context `C` | Readout `R` | Geometry `G` | Supervision `S` | Explanation target |
|---|---|---|---|---|---|
| TCAV | local class gradient at layer `l` | sign frequency over class samples | CAV direction in activation space | concept examples and random references | directional sensitivity |
| ACE | SLIC, activation embedding, K-means | TCAV ranking of discovered clusters | pixel locality plus latent Euclidean geometry | class images and random references | global discovered-concept sensitivity |
| ProtoMIL | sparse encoding and attention | weighted concept mean | FM feature and sparse dictionary geometry | reconstruction, sparsity, slide labels | exact linear logit credit |
| GECKO | VLM similarities, top-K, mixer, gates | mean of gated concept rows | shared vision-language geometry | task lexicon and contrastive pairs | slide concept activation |

## 5. Unified Failure Principle

Every concept explanation can fail at four interfaces:

```math
\text{examples or prompts}
\xrightarrow{\text{concept construction}}
C
\xrightarrow{\text{representation geometry}}
z_C
\xrightarrow{\text{aggregation}}
u_C
\xrightarrow{\text{task functional}}
E_C.
```

The required report names the concept source, reference distribution, layer or
encoder, aggregation rule, target score, local or global scope, sign semantics,
statistical unit, semantic validation, completeness test, and intervention
operator.

