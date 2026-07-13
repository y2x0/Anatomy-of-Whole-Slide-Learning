# Prompt Ensembles And Semantics

Prompt ensembling averages multiple text queries for the same class.

Let prompts for class
```math
c
```
be:

```math
p_{c1},\ldots,p_{cM}.
```

Their text embeddings are:

```math
v_{cm}
=
f_T(p_{cm}).
```

A prompt ensemble prototype can be:

```math
\bar v_c
=
\frac{1}{M}
\sum_{m=1}^{M}
\frac{v_{cm}}{\|v_{cm}\|_2}.
```

The score is:

```math
s_c(x)
=
f_I(x)^\top \bar v_c.
```

## Variance Across Prompts

Prompt sensitivity can be measured by:

```math
\mathrm{Var}_m
\left[
f_I(x)^\top v_{cm}
\right].
```

High variance means the class boundary is unstable under wording.

## Semantic Granularity

Two prompts may correspond to different latent targets:

```text
adenocarcinoma
poorly differentiated adenocarcinoma
tumor epithelium
invasive front
```

These are not interchangeable. The prompt defines the target concept.

## Dense Summary

Prompt ensembling reduces wording variance, but it does not solve semantic
mismatch. If the prompts describe different pathology objects, averaging them
averages different tasks.
