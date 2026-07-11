# The Negative-Free Statistical Object

Primary anchors:

- Caron et al. "Emerging Properties in Self-Supervised Vision Transformers."
  ICCV 2021. https://arxiv.org/abs/2104.14294
- Zhou et al. "iBOT: Image BERT Pre-Training with an Online Tokenizer." ICLR
  2022. https://arxiv.org/abs/2111.07832
- Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision."
  TMLR 2024. https://arxiv.org/abs/2304.07193

## Views Without Candidate Negatives

For image `X`, sample views:

```math
V_s
\sim
q_s(\cdot\mid X),
\qquad
V_t
\sim
q_t(\cdot\mid X).
```

A student distribution and stop-gradient teacher distribution are:

```math
p_s
=
p_{\theta_s}(\cdot\mid V_s),
\qquad
p_t
=
\mathrm{sg}
\left[
p_{\theta_t}(\cdot\mid V_t)
\right].
```

The primitive loss is cross-entropy:

```math
\mathcal{L}_{t\rightarrow s}
=
H(p_t,p_s)
=
-\sum_{k=1}^{K}
p_{t,k}
\log p_{s,k}.
```

No other image appears in the denominator. The target distribution itself is
nonstationary because the teacher follows the student.

## Teacher Dynamics

The student is optimized by gradient descent:

```math
\theta_s^{(r+1)}
=
\theta_s^{(r)}
-
\eta_r
\nabla_{\theta_s}
\mathcal{L}^{(r)}.
```

The teacher is an exponential moving average:

```math
\theta_t^{(r+1)}
=
\lambda_r
\theta_t^{(r)}
+
\left(
1-\lambda_r
\right)
\theta_s^{(r+1)}.
```

Unrolling gives a temporal ensemble:

```math
\theta_t^{(r)}
=
\left(
\prod_{a=0}^{r-1}\lambda_a
\right)
\theta_t^{(0)}
+
\sum_{b=1}^{r}
\left[
\left(
1-\lambda_{b-1}
\right)
\prod_{a=b}^{r-1}
\lambda_a
\right]
\theta_s^{(b)}.
```

The teacher is not an independently trained oracle. It is a lagged average of
student states.

## Constant Solution

If every view maps to one distribution `c`:

```math
p_s(\cdot\mid v)
=
p_t(\cdot\mid v')
=
c
```

for all `v,v'`, then student and teacher agree perfectly in the KL component:

```math
\mathrm{KL}(p_t\|\|p_s)
=
0.
```

Cross-view agreement alone therefore does not exclude collapse.

## What Supplies Repulsion

DINO introduces batch centering and teacher sharpening. iBOT forces masked
patch positions to predict teacher token distributions from visible context.
DINOv2 combines image-level and patch-level distillation with assignment
balancing and feature-spreading regularization.

These are distributional constraints:

```math
\text{centering},
\quad
\text{entropy control},
\quad
\text{masked conditional prediction},
\quad
\text{nearest-neighbor spacing}.
```

## Pathology Boundary

All objectives operate initially on sampled image tiles or regions. A WSI
prediction still requires:

```math
\left\{
x_{ij}
\right\}_{j=1}^{n_i}
\xrightarrow{f_{\theta}}
\left\{
h_{ij}
\right\}_{j=1}^{n_i}
\xrightarrow{\mathcal{R}_{\mathrm{slide}}}
z_i.
```

Negative-free patch pretraining does not define the slide readout.
