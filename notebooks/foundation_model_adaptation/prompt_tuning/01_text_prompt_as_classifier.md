# Text Prompt As Classifier

Vision-language pathology models align image and text embeddings.

For image $x$ and class prompt $p_c$:

```math
u
=
f_I(x),
\qquad
v_c
=
f_T(p_c).
```

The zero-shot class score is:

```math
s_c(x)
=
\frac{u^\top v_c}{\tau}.
```

The predicted class is:

```math
\widehat y
=
\arg\max_c s_c(x).
```

## Prompt-Defined Boundary

Between classes $a$ and $b$, the decision boundary is:

```math
u^\top v_a
=
u^\top v_b.
```

Equivalently:

```math
u^\top(v_a-v_b)
=
0.
```

Thus the prompt pair defines a linear boundary in image embedding space.

## Task Information

The task information is in:

```math
p_c.
```

Different wording changes:

```math
v_c
=
f_T(p_c),
```

and therefore changes the classifier.

## Pathology Caveat

If the prompt describes a case-level diagnosis but the image crop contains
background or nondiagnostic tissue:

```math
p_c
\text{ describes }
Y_{\mathrm{case}},
\qquad
x
\text{ shows }
U_{\mathrm{local}},
```

then prompt classification is solving a mismatched relation.

## Dense Summary

Text prompting is not label-free. It moves supervision into language:

```math
\text{class name}
\to
\text{prompt}
\to
v_c
\to
\text{decision boundary}.
```
