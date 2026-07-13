# Catastrophic Forgetting And Small Cohorts

Fine-tuning on a small cohort can overwrite broad pretrained knowledge.

Let
```math
P_0
```
 be the pretraining distribution and
```math
P_T
```
the downstream task
distribution. Pretraining optimized:

```math
\phi_0
\approx
\arg\min_\phi
\mathbb{E}_{x\sim P_0}
\ell_0(f_\phi,x).
```

Fine-tuning optimizes:

```math
\phi_T
=
\arg\min_\phi
\mathbb{E}_{(x,y)\sim P_T}
\ell_T(f_\phi(x),y).
```

If
```math
P_T
```
 is small or biased,
```math
\phi_T
```
may improve the task validation set while
damaging general morphology representation.

## Overfitting Signal

With small
```math
N
```
and many parameters:

```math
\dim(\phi)
\gg
N,
```

many parameter updates can fit labels. The model may encode:

```text
site
scanner
stain
artifact
case mix
```

instead of pathology.

## Forgetting As Geometry Change

A neighbor relation in pretrained space:

```math
x_a
\sim_0
x_b
```

may break after fine-tuning:

```math
\|f_{\phi_T}(x_a)-f_{\phi_T}(x_b)\|
\gg
\|f_{\phi_0}(x_a)-f_{\phi_0}(x_b)\|.
```

This can improve the downstream label but harm reuse, retrieval, or
interpretability.

## Dense Summary

Full fine-tuning trades broad pretrained invariance for task specificity. The
smaller and noisier the task cohort, the more dangerous that trade becomes.
