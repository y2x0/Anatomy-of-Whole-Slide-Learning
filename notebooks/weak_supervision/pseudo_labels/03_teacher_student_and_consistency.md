# Teacher-Student And Consistency

Teacher-student methods create pseudo-labels from a teacher model.

Let teacher parameters be $\phi$ and student parameters be $\theta$.

The teacher prediction is:

```math
q_i
=
P_\phi(Y\mid H_i,G_i).
```

The student prediction is:

```math
p_i
=
P_\theta(Y\mid H_i,G_i).
```

## Distillation Loss

A soft pseudo-label objective is:

```math
\mathcal{L}_{\mathrm{KD}}
=
\sum_i
\mathrm{KL}
\left(
q_i
\;\|\;
p_i
\right).
```

With temperature $T$:

```math
q_i^{(T)}
=
\mathrm{softmax}(o_i^\phi/T),
\qquad
p_i^{(T)}
=
\mathrm{softmax}(o_i^\theta/T).
```

Higher temperature exposes class similarities but can also soften wrong teacher
beliefs.

## Mean Teacher

The teacher can be an exponential moving average of the student:

```math
\phi_t
=
\alpha\phi_{t-1}
+
(1-\alpha)\theta_t.
```

The consistency loss is:

```math
\mathcal{L}_{\mathrm{cons}}
=
\left\|
P_\theta(Y\mid t_1(H_i))
-
P_\phi(Y\mid t_2(H_i))
\right\|_2^2.
```

This assumes augmentations $t_1,t_2$ preserve the label.

## Instance-Level Teacher

A teacher may generate instance pseudo-labels:

```math
\widehat z_{ij}
=
\arg\max_c
P_\phi(Z_{ij}=c\mid h_{ij}).
```

or soft targets:

```math
q_{ij}
=
P_\phi(Z_{ij}\mid h_{ij}).
```

The student instance loss is:

```math
\mathcal{L}_{\mathrm{instKD}}
=
\sum_{i,j}
\mathrm{KL}(q_{ij}\|p_{\theta,ij}).
```

This is stronger than bag supervision and inherits the teacher's errors.

## Noisy Student View

Noisy student training adds augmentation and model noise:

```math
\widehat y_i
=
\arg\max q_i,
```

```math
\mathcal{L}
=
\mathrm{CE}
\left(
P_\theta(Y\mid \mathrm{Aug}(H_i)),
\widehat y_i
\right).
```

The model learns invariance to perturbations while matching teacher labels.

## C/R/G/S Placement

```text
G:
    teacher may use a different geometry from the student

C:
    student representation is shaped by teacher outputs

R:
    teacher may supervise slide, region, pseudo-bag, or instance readouts

S:
    original labels plus teacher-generated soft or hard labels
```

## Dense Summary

Teacher-student learning replaces direct supervision:

```math
Y
```

with a learned supervision distribution:

```math
q_\phi(\cdot\mid H,G).
```

It is powerful when the teacher is smoother or better informed than the student,
and dangerous when the teacher's mistakes are treated as ground truth.
