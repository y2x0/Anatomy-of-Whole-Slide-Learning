# Patient Leakage, False Positives, And Identifiability

## Slide Is Not Patient

```math
\left\{
\mathcal{X}_{i1},\ldots,
\mathcal{X}_{im_i}
\right\}
```

can all belong to patient `i`. Slide-level splitting permits:

```math
P_{\mathrm{train}}
\cap
P_{\mathrm{test}}
\ne
\varnothing.
```

Cross-slide objectives can exploit patient-specific stain, preparation, or
tissue signatures and inflate external performance.

## Same-Patient Negatives

```math
P_i=P_j,
\qquad
i\ne j.
```

If distinct slides are treated as unrelated candidates, the objective can
repel shared patient biology. The correct relation unit may be patient,
specimen, block, slide, or region.

## Weak Positive Nonidentifiability

For a positive bag, many patch assignments satisfy:

```math
Y_i
=
\max_j y_{ij}
=
1.
```

Slide labels do not identify:

```math
T_i
=
\left\{
j:y_{ij}=1
\right\}.
```

Attention-selected positives add an inductive assumption; they do not make
latent patch labels observed.

## Shared Benign Distribution

```math
\mu_i
=
\pi_i\mu_i^{\mathrm{lesion}}
+
\left(
1-\pi_i
\right)
\mu_i^{\mathrm{benign}}.
```

```math
\mathbb{E}_{\mu_i}[H]
=
\pi_i
\mathbb{E}_{\mu_i^{\mathrm{lesion}}}[H]
+
\left(
1-\pi_i
\right)
\mathbb{E}_{\mu_i^{\mathrm{benign}}}[H].
```

At low lesion prevalence, full-slide means are dominated by benign tissue.
Selected subbags help only when attention recall is high.

## Institution Shortcut

```math
I(Y;S)>0
```

allows same-label attraction to organize slides by site-related appearance.
Cross-slide supervision cannot distinguish causal morphology from predictive
cohort variables.

## Identifiable Claim

```math
p
\left(
R=1
\mid
z_i,z_j
\right)
```

is compatibility under the constructed relation. It is not lesion
correspondence or causal biomarker identification.
