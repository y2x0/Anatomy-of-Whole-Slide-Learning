# HIPPO Set Interventions

Primary anchor:

- Kaczmarzyk et al. "Explainable AI for Computational Pathology Identifies
  Model Limitations and Tissue Biomarkers." 2024.
  https://arxiv.org/abs/2409.03080

HIPPO abbreviates Histopathology Interventions of Patches for Predictive
Outcomes.

## MIL Set Function

Let patch embeddings be:

```math
P
=
\left\{
p_1,\ldots,p_n
\right\}.
```

A permutation-invariant model satisfies:

```math
F(P)
=
F
\left(
\pi P
\right)
```

for every permutation `pi`.

## Exclusion

For selected subregion patch set `R`:

```math
P^{-R}
=
P\setminus R,
```

```math
\Delta^{-}(R)
=
F(P)
-
F
\left(
P^{-R}
\right).
```

This quantifies the model effect of removing that represented tissue.

## Inclusion

For donor region `A`:

```math
P^{+A}
=
P\cup A,
```

```math
\Delta^{+}(A)
=
F
\left(
P^{+A}
\right)
-
F(P).
```

The effect depends on donor multiplicity and recipient context.

## HIPPO-Knowledge

Expert annotations or a declared histologic hypothesis define `R` or `A`.
This tests whether model behavior changes under removal or addition of known
tissue categories.

## HIPPO-Attention

Let top-attention subset be:

```math
R_{\mathrm{attn}}
=
\mathrm{TopK}
\left(
\left\{
\alpha_i
\right\},k
\right).
```

HIPPO-attention evaluates:

```math
F(P)
-
F
\left(
P\setminus R_{\mathrm{attn}}
\right).
```

This directly tests whether highly attended tissue is influential under
deletion.

## Valid-Input Claim

Variable-size set models accept resampled bags without synthetic pixels or
inpainting. The modified set is valid for the function signature. It can still
be statistically unusual due to altered tissue proportions and cardinality.

## Causal Boundary

HIPPO identifies controlled effects on model output:

```math
\mathrm{do}
\left(
P=P\setminus R
\right).
```

Calling this a causal model explanation is appropriate. Calling it a causal
effect of tissue on clinical outcome requires additional biological assumptions.
