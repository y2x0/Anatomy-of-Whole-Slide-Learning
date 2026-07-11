# Patient Leakage, False Positives, And Identifiability

## Slide Is Not Patient

Let patient `P_i` contribute slides:

```math
\left\{
\mathcal{X}_{i1},\ldots,
\mathcal{X}_{im_i}
\right\}.
```

If train and validation splitting occurs at slide level, the same patient can
enter both sets:

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

If two slides from one patient have different labels or are treated as distinct
instances, a contrastive relation may repel them despite shared biology:

```math
P_i=P_j,
\qquad
i\ne j.
```

The correct unit for splitting and relation design can be patient, specimen,
block, slide, or region depending on the task.

## Weak Positive Nonidentifiability

For a positive bag, many latent patch assignments satisfy:

```math
Y_i
=
\max_j y_{ij}
=
1.
```

Any nonempty subset can be the positive set. Slide labels alone do not identify:

```math
T_i
=
\left\{
j:y_{ij}=1
\right\}.
```

Attention-selected cross-slide positives add an inductive assumption; they do
not resolve identifiability from observed labels alone.

## Shared Benign Distribution

Model slide distribution as:

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

For small lesion prevalence, distances between full-slide means are
dominated by benign tissue:

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

Selected-subbag contrast tries to increase effective lesion prevalence, but succeeds only
when attention recall is high.

## Institution Shortcut

If label and site are correlated:

```math
I(Y;S)>0,
```

same-label attraction can compress slides by institution-related appearance.
Cross-slide supervision does not distinguish causal morphology from any
predictive cohort variable.

## Identifiable Claim

What the objective directly identifies is compatibility under its constructed
relation:

```math
p
\left(
R=1
\mid
z_i,z_j
\right).
```

It does not identify lesion correspondence, causal biomarkers, or patient
prognosis without additional assumptions.
