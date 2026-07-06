# Foundation Latent Representation Failure Modes

Foundation representations are powerful because they import large-scale
pretraining. They fail when the imported geometry does not match the downstream
question.

## 1. Pretraining Bias

The pretrained encoder is optimized for:

```math
\ell_{\text{pretrain}},
```

not directly for:

```math
\ell_{\text{task}}.
```

If pretraining emphasizes tissue type, stain, magnification, or caption
semantics, then those directions may dominate the latent space.

Downstream linear probing:

```math
\widehat y_i
=
Wz_i+b
```

can only use directions already linearly accessible in $z_i$.

## 2. Patch-Foundation Illusion

A strong patch encoder:

```math
h_{ij}=E_{\text{FM}}(x_{ij})
```

does not solve slide representation:

```math
\{h_{ij}\}_{j=1}^{n_i}
\not\mapsto
z_i
```

without an aggregation or slide encoder.

The hard part may move from feature extraction to:

```text
which patch features interact, survive, or get ignored
```

## 3. Latent Mean Collapse

Using mean pooling over foundation features:

```math
z_i
=
\frac{1}{n_i}
\sum_jh_{ij}
```

can erase rare morphology even when the patch features are strong.

The encoder may separate rare patterns locally:

```math
h_{ij^\star}
\quad
\text{distinct}
```

but the slide statistic gives it weight:

```math
\frac{1}{n_i}.
```

Foundation features do not remove aggregation failure modes.

## 4. Prompt Mismatch

For text-aligned models:

```math
p(y=c\mid S_i)
\propto
\exp(z_i^\top t_c/\tau).
```

But:

```math
t_c
=
F_{\text{text}}(\operatorname{prompt}(c)).
```

If the prompt fails to express the relevant pathology concept, then the
classifier is mis-specified.

Different prompts can define different decision boundaries:

```math
t_c^{(1)}\ne t_c^{(2)}.
```

## 5. Domain Shift In Latent Space

Let source and target distributions be:

```math
p_{\text{src}}(z,y),
\qquad
p_{\text{tgt}}(z,y).
```

Even if image-level representations transfer well on average, downstream tasks
can fail when:

```math
p_{\text{src}}(y\mid z)
\ne
p_{\text{tgt}}(y\mid z).
```

The same latent neighborhood may have different clinical meaning across cohorts.

## 6. Overconfident Semantic Alignment

Image-text alignment creates a shared space, but language labels are coarse and
reports are incomplete. A report may omit visual details that matter for a new
task.

The model may learn:

```math
z_i\approx t_i
```

for report-level semantics while discarding morphology not mentioned in text.

## Diagnostic Questions

1. Is the foundation model patch-level or slide-level?
2. What pretraining objective shaped the latent geometry?
3. Is the downstream task linearly accessible from frozen features?
4. What aggregation is used after patch encoding?
5. Do prompts encode the actual pathology distinction?
6. Does latent similarity remain valid under cohort shift?

## Dense Summary

The foundation-latent assumption is:

```text
large-scale pretraining produced a useful geometry for this downstream slide task
```

It fails when:

```text
pretraining geometry, aggregation, prompt semantics, or target cohort differ from the task's true structure
```
