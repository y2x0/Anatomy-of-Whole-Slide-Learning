# UNI and Virchow: Patch Foundation Geometry

## 1. Shared Patch-Level Contract

UNI and Virchow learn broad pathology patch representations before downstream
aggregation:

```math
h=f_\phi(x_{\mathrm{patch}}).
```

The downstream WSI model receives a set or sequence of such vectors. The patch
encoder itself does not decide how a slide is represented.

## 2. Self-Supervised Distinction

The papers use large-scale self-supervised visual pretraining with teacher,
student, and/or masked-token components. A generic combined objective is

```math
\mathcal L
=\lambda_{\mathrm{inv}}\mathcal L_{\mathrm{view}}
+\lambda_{\mathrm{mask}}\mathcal L_{\mathrm{masked}}
+\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{regularize}}.
```

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

