# UNI and Virchow: Patch Foundation Geometry

References: [UNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/),
[Virchow](https://arxiv.org/abs/2309.07778).

## 1. Shared Patch-Level Contract

UNI and Virchow learn broad pathology patch representations before downstream
aggregation:

```math
h=f_\phi(x_{\mathrm{patch}}).
```

The downstream WSI model receives a set or sequence of such vectors. The patch
encoder itself does not decide how a slide is represented.

## 2. Self-Supervised Distinction

UNI and Virchow are related by their patch-level foundation-model role, not by
one shared loss equation. Their exact pretraining pipelines, data mixtures,
architectures, and objective terms must be taken from their respective source
papers. The following is only a comparison abstraction for a model that really
combines view invariance, masking, and regularization:

```math
\mathcal L
=\lambda_{\mathrm{inv}}\mathcal L_{\mathrm{view}}
+\lambda_{\mathrm{mask}}\mathcal L_{\mathrm{masked}}
+\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{regularize}}.
```

It should not be read as a claim that UNI and Virchow optimize this same sum or
that every term is present in both models. The safe paper-level statement is
that both provide patch representations whose downstream geometry must be
measured separately from the slide readout.

The exact weights and implementation determine which patch variation survives;
the model name alone does not specify that statistic.

## 3. Transfer Boundary

With frozen `f_phi`, a WSI readout is

```math
z_i=\mathcal R(\{f_\phi(x_{ij})\}_j;G_i),
```

so downstream attention, graph, hierarchy, or survival heads add a second
inductive bias. A patch encoder benchmark cannot be read as a slide-encoder
benchmark.

## 4. Failure Mode

Patch-level training can overrepresent common tissue and underrepresent rare
prognostic regions. A slide sampler or downstream aggregator must be audited
for the resulting coverage imbalance.
