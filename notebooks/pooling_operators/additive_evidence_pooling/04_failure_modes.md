# Additive Evidence Failure Modes

Additive pooling is interpretable, but the sum assumption can be wrong.

## 1. Double Counting Correlated Patches

Nearby patches are correlated. If the same lesion appears in many overlapping
tiles:

```math
r_i
=
\sum_j e_{ij}
```

may count the same biological evidence multiple times.

## 2. Patch Count Confounding

If larger tissue area produces more patches:

```math
n_i\uparrow
\quad
\Rightarrow
\quad
r_i\uparrow
```

even when evidence density is unchanged. This is correct only if total tissue
burden is the target.

## 3. Missing Interactions

Additive pooling assumes:

```math
r_i
=
\sum_j e(u_{ij}).
```

It cannot represent:

```math
r_i
=
\sum_{j,k} e_2(u_{ij},u_{ik})
```

unless pairwise interactions are encoded into
```math
u_{ij}
```
before summation.

## 4. Calibration Of Evidence Units

Evidence scores are learned:

```math
e_{ij}=e_\theta(u_{ij}).
```

They are interpretable only up to the scale implied by the head and loss. A
patch with evidence
```math
2
```
is not biologically twice as diseased unless the model
and calibration justify that interpretation.

## 5. Negative Evidence Ambiguity

Negative evidence can mean:

```text
benign tissue
artifact opposing the class
absence of expected morphology
model shortcut
```

A signed map is faithful to the model, not automatically faithful to pathology.

## Dense Summary

Additive pooling fails when:

```text
evidence is not additive,
patch counts are nuisance,
or interactions matter more than individual contributions.
```

Its strength is exact model credit assignment. Its weakness is the assumption
that the slide score should decompose into independent patch scores.
