# Decision Checklist

Every foundation-model adaptation paper should state these choices.

## 1. Native Representation

What does the foundation model output?

```text
patch embeddings
slide embeddings
image-text embeddings
retrieval memory
multimodal generative latent
```

## 2. Trainable Set

State:

```math
\Theta_{\mathrm{train}}
\subseteq
(\phi,\psi,\omega,\pi,M).
```

Which parameters receive gradients?

## 3. Adaptation Constraint

If the encoder changes, write:

```math
\phi
=
\phi_0+\Delta_\eta,
\qquad
\Delta_\eta\in\mathcal{A}.
```

Then define
```math
\mathcal{A}
```
:

```text
low-rank
adapter bottleneck
prompt tokens
all parameters
memory update
```

## 4. Task Object

State the target:

```text
slide class
patch concept
region property
survival risk
retrieval relevance
report generation
```

The task object must match the representation object.

## 5. Readout

If using patch FMs, state the slide readout:

```math
z_i
=
\mathcal{R}(\{f_{\phi_0}(x_{ij})\}_{j=1}^{n_i}).
```

The readout is not a minor implementation detail. It defines what slide
statistic survives.

## 6. Drift

If adapting the encoder, report:

```math
\mathbb{E}_x
\|f_{\phi_0+\Delta}(x)-f_{\phi_0}(x)\|^2.
```

or an equivalent representation-drift diagnostic.

## 7. Failure Counterexample

Give a toy pair:

```math
F_{\phi_0}(X)
=
F_{\phi_0}(X'),
\qquad
Y
\ne
Y'.
```

If such pairs are plausible, head-only adaptation cannot solve the task.

## Dense Summary

Foundation adaptation is clear when the reader can fill:

```math
(\phi_0,\Theta_{\mathrm{train}},\mathcal{A},\mathcal{R},S,\text{drift},\text{failure mode}).
```

Without those pieces, "we used a foundation model" is not a mathematical
statement.
