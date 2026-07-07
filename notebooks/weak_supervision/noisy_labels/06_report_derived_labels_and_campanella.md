# Report-Derived Labels And Campanella-Style Supervision

Clinical weak supervision often uses reported diagnoses as labels.

Reference: [Campanella et al. clinical weak supervision](https://pmc.ncbi.nlm.nih.gov/articles/PMC7418463/).

The observed object is not:

```math
Y_{\mathrm{patch}}.
```

It is usually:

```math
\widetilde Y_{\mathrm{case}}
=
E_{\mathrm{report}}(R_{\mathrm{case}}),
```

where $R_{\mathrm{case}}$ is clinical/pathology text or metadata and
$E_{\mathrm{report}}$ is an extraction rule.

Campanella et al. style training uses reported diagnoses at scale rather than
manual pixel annotations. That scale helps, but the supervision channel has
several layers.

## Case-To-Slide Mapping

A case can contain multiple slides:

```math
\mathcal{S}_{\mathrm{case}}
=
\{s_1,\ldots,s_m\}.
```

The case label may be copied to each slide:

```math
\widetilde Y_{s}
=
\widetilde Y_{\mathrm{case}}
\qquad
s\in\mathcal{S}_{\mathrm{case}}.
```

This is correct only if every slide contains the diagnostic evidence or if the
slide-level aggregator is allowed to handle irrelevant slides.

## Report Extraction

The report extractor maps text to labels:

```math
E_{\mathrm{report}}:
R_{\mathrm{case}}
\to
\widetilde Y.
```

Errors can come from:

```text
negation
uncertainty
history versus current diagnosis
specimen adequacy
template text
clinical context
multiple diagnoses in one report
```

This is not the same as symmetric class-conditional noise.

## Negative Curation

If a negative label means:

```text
no positive diagnosis was extracted
```

that is not the same as:

```math
Y=0.
```

Negative curation determines whether the negative class is clean, contaminated,
or selected by workflow.

## Aggregation Unit

The model may score tiles, slides, or cases:

```math
\text{tile}
\to
\text{slide}
\to
\text{case}.
```

If training labels are case-level but the model reports slide- or patch-level
heatmaps, then localization is weaker than classification.

## C/R/G/S Placement

```text
G:
    case/slide grouping and tissue sampling structure

C:
    tile features are learned under report-derived slide/case labels

R:
    top-k, RNN, attention, or MIL aggregation maps tile evidence to slide/case prediction

S:
    report-extracted diagnosis, not direct slide-local truth
```

## Dense Summary

Report-derived supervision should be written as:

```math
U_{\mathrm{case}}
\to
R_{\mathrm{case}}
\to
\widetilde Y_{\mathrm{case}}
\to
\widetilde Y_{\mathrm{slide}}.
```

Each arrow can introduce noise, selection, or unit mismatch.
