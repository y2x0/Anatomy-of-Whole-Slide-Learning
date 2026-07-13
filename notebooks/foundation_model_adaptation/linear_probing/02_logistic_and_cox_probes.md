# Logistic And Cox Probes

Linear probing is not only for classification. It also appears in survival
modeling.

## Logistic Probe

For binary classification:

```math
\widehat p_i
=
\mathrm{sigmoid}(w^\top z_i+b).
```

The empirical loss is:

```math
\mathcal{L}_{\mathrm{logistic}}
=
-
\sum_i
\left[
y_i\log\widehat p_i
+
(1-y_i)\log(1-\widehat p_i)
\right].
```

The probe learns a direction
```math
w
```
in frozen representation space.

## Multiclass Probe

For
```math
C
```
classes:

```math
o_i
=
Wz_i+b,
\qquad
\widehat p_i
=
\mathrm{softmax}(o_i).
```

Each class has a direction:

```math
w_c^\top z_i+b_c.
```

The class decision compares linear scores:

```math
\widehat y_i
=
\arg\max_c
(w_c^\top z_i+b_c).
```

## Cox Probe

For survival, a linear Cox probe uses:

```math
\eta_i
=
w^\top z_i.
```

The partial likelihood loss is:

```math
\mathcal{L}_{\mathrm{Cox}}
=
-
\sum_{i:\delta_i=1}
\left[
\eta_i
-
\log
\sum_{k:T_k\ge T_i}
\exp(\eta_k)
\right].
```

This tests whether frozen representations contain a linear prognostic risk
direction.

## What Changes Across Tasks

The frozen representation is the same:

```math
z_i
=
F_{\phi_0}(X_i).
```

The target functional differs:

```text
classification:
    class posterior

survival:
    risk ordering or hazard parameter

retrieval:
    similarity metric or query score
```

## Dense Summary

Linear probes are task heads with no representational adaptation:

```math
F_{\phi_0}
\text{ fixed},
\qquad
w
\text{ trained}.
```

Their success means the task was already readable from the frozen embedding
under the chosen linear functional.
