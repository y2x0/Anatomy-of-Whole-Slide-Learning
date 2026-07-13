# GECKO Vision-Language Concept Prior

GECKO constructs an explicit patch-by-concept matrix from a pathology lexicon
and a pretrained vision-language model.

## 1. Lexicon and Shared Space

An LLM proposes task-specific textual pathology descriptions. A VLM text
encoder produces

```math
T=[t_1,\ldots,t_C]^{\top}\in\mathbb R^{C\times D},
```

while its vision encoder maps `N` WSI patches to

```math
F=[f_1,\ldots,f_N]^{\top}\in\mathbb R^{N\times D}.
```

The concept prior is the cosine-similarity matrix

```math
M_{jc}
=\frac{f_j^{\top}t_c}
{\|f_j\|_2\|t_c\|_2},
\qquad
M\in\mathbb R^{N\times C}.
```

Each coordinate is readable because it is indexed by a text description. Its
numeric value is VLM alignment, not a calibrated probability that the
histologic concept is present.

## 2. Task Conditioning

Concepts are requested per downstream class and selected for visual
discriminativeness. Therefore the prior is

```math
M=M(X;\mathcal L_{\mathrm{task}},E_{\mathrm{VLM}},Q_{\mathrm{LLM}}).
```

It is not unsupervised with respect to task semantics even when no slide label
is used in contrastive pretraining.

## 3. Geometric Ambiguities

Cosine similarity is invariant to positive scaling but sensitive to prompt
direction:

```math
\mathrm{sim}(f,t)
=\mathrm{sim}(f,\alpha t)
\quad\text{for }\alpha>0.
```

Synonymous prompts need not yield identical directions. A high score can arise
from morphology correlated with the phrase in VLM pretraining rather than the
literal diagnostic criterion.

## 4. Grounding Audit

For concept `c`, validate patch scores against expert labels `y_jc` using
ranking, calibration, and subgroup tests:

```math
p(M_{jc}\mid y_{jc}=1)
\quad\text{versus}\quad
p(M_{jc}\mid y_{jc}=0).
```

Also test prompt paraphrases, negations, stain shifts, scanners, and external
institutions. Readable coordinates without these checks are named latent
similarities, not verified pathology measurements.

