# CPLIP Bag Construction Map

Primary anchor:

- Javed et al. "CPLIP: Zero-Shot Learning for Histopathology with
  Comprehensive Vision-Language Alignment." CVPR 2024.
  https://arxiv.org/abs/2406.05205

CPLIP converts one unlabeled histology image into a matched text bag and visual
bag before applying MIL-NCE. The supervision relation is produced by a chain
of pretrained models, a pathology dictionary, generation, retrieval, and
thresholding.

## Dictionary Assignment

Let the refined pathology dictionary be:

```math
\mathcal{V}
=
\left\{
v_1,\ldots,v_D
\right\}.
```

For seed image `h_i`, MI-Zero supplies the assignment:

```math
\widehat v_i
=
\arg\max_{v\in\mathcal{V}}
\mathrm{sim}
\left(
f_0(h_i),g_0(v)
\right).
```

This discrete variable is the root of both bags:

```math
Z_i
=
\widehat v_i.
```

An assignment error propagates into text generation, text retrieval, and image
retrieval.

## Text Bag

The initial text bag combines four mechanisms:

```math
\widetilde{\mathcal{B}}_i^t
=
\left\{
\widehat v_i
\right\}
\cup
\mathcal{G}_{\mathrm{desc}}
\left(
\widehat v_i
\right)
\cup
\mathcal{G}_{\mathrm{cause}}
\left(
\widehat v_i
\right)
\cup
\mathcal{G}_{\mathrm{symptom}}
\left(
\widehat v_i
\right)
\cup
\mathcal{R}_{I\rightarrow T}
\left(
h_i
\right).
```

The paper obtains five alternate descriptions, three causes, three symptoms,
and five PLIP-retrieved descriptions in addition to the selected prompt.

## Text Pruning

For threshold `\delta_t`, retain:

```math
\mathcal{B}_i^t
=
\left\{
t\in\widetilde{\mathcal{B}}_i^t:
\mathrm{sim}
\left(
f_0(h_i),g_0(t)
\right)
\ge
\delta_t
\right\}.
```

The same pretrained geometry both proposes and validates members. Generated
descriptions outside its similarity region are removed even if clinically
valid.

Because generation and retrieval are pipeline operations rather than observed
annotations, it is more precise to write the resulting bags as random sets:

```math
\mathcal{B}_i^t
\sim
\mathsf{K}_t
\left(
\cdot
\mid
\widehat v_i,h_i;\omega_t
\right),
```

```math
\mathcal{B}_i^v
\sim
\mathsf{K}_v
\left(
\cdot
\mid
\mathcal{B}_i^t,h_i;\omega_v
\right).
```

Here `\mathsf{K}_t` includes language-model generation, retrieval, and
thresholding, while `\mathsf{K}_v` includes text-to-image and image-to-image
retrieval. Even if retrieval is deterministic after fixing the encoders, the
generated text and any tie-breaking or sampling rule remain part of the
induced supervision distribution.

## Visual Bag

Visual concepts are retrieved from the selected dictionary prompt, enriched
descriptions, and the seed image:

```math
\mathcal{B}_i^v
=
\mathcal{R}_{T\rightarrow I}
\left(
\mathcal{B}_i^t
\right)
\cup
\mathcal{R}_{I\rightarrow I}
\left(
h_i
\right).
```

The paper reports up to 21 visual concepts and up to 17 textual descriptions.
No additional visual-bag pruning is applied after text-bag pruning.

## Dependency Graph

The construction is not two independent noisy views. It is a coupled graph:

```math
h_i
\longrightarrow
\widehat v_i
\longrightarrow
\widetilde{\mathcal{B}}_i^t
\longrightarrow
\mathcal{B}_i^t
\longrightarrow
\mathcal{B}_i^v.
```

There is also a direct edge:

```math
h_i
\longrightarrow
\mathcal{B}_i^v
```

through image-to-image retrieval. Conditional independence assumptions such
as:

```math
\mathcal{B}_i^v
\perp
\mathcal{B}_i^t
\mid
Z_i
```

do not hold under this construction.

## Bag Relation Versus Instance Relation

CPLIP observes:

```math
\mathcal{B}_i^v
\longleftrightarrow
\mathcal{B}_i^t.
```

This does not establish that every Cartesian-product pair is valid:

```math
\forall
x\in\mathcal{B}_i^v,
\forall
t\in\mathcal{B}_i^t,
\qquad
x\sim t.
```

The MIL-NCE numerator nevertheless assigns positive exponential mass to the
entire Cartesian product. Its log-sum-exp can tolerate some bad members, but
the supervision assumption is stronger than bag-level compatibility alone.

## Failure Cascade

If dictionary assignment has error probability `\varepsilon_Z`, subsequent
steps cannot make total bag correctness exceed:

```math
p
\left(
\text{correct bag semantics}
\right)
\le
1-\varepsilon_Z
```

without an explicit correction mechanism capable of overturning `Z_i`.
Retrieval enrichment increases diversity conditional on the seed assignment;
it does not independently verify that assignment.
