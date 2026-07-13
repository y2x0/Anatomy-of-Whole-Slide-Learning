# Open Design Axes and Reporting Rule

## 1. Design Axes

```text
patch versus slide objective;
appearance versus morphology target;
single stain versus multistain view;
image-only versus language or knowledge alignment;
local versus long-range context;
retrieval versus parametric readout;
teacher transfer versus independent student geometry;
in-domain versus external-cohort validation.
```

The unexplored method space is the Cartesian product of these choices, subject
to compute and data constraints.

## 2. Minimum PFM Report

Every foundation-model paper should report:

```text
pretraining data unit; view construction; positive/negative or mask rule;
teacher or text source; objective; encoder and context geometry; sampling unit;
downstream readout; patient-level split; external-shift test; calibration or
retrieval metric; and failure modes expected from the objective.
```

## 3. Final Placement

```math
\text{PFM claim}
\mapsto
(\mathfrak P,\Phi,\mathcal R,\mathcal H,\text{evaluation},\text{shift}).
```

Without this tuple, “foundation model” is a scale label rather than a
mathematical description.

## Pretraining Contract As An Object

Let a foundation-model contract be

```math
\mathfrak P
=
\left(\mathcal D,\mathcal V,\mathcal Q,\mathcal L_{\mathrm{pre}},
\mathcal O,\mathcal G\right),
```

where D is the data distribution, V is the view or masking mechanism, Q is the
positive/negative or teacher-target construction, L_pre is the objective, O is
the output object, and G is the context or geometry available during training.
The learned encoder is a function of the whole contract:

```math
\phi^{\star}
\in
\underset{\phi}{\arg\min}
\mathbb E_{x\sim\mathcal D,
v\sim\mathcal V,
q\sim\mathcal Q}
\left[
\mathcal L_{\mathrm{pre}}(\phi;x,v,q,\mathcal G)
\right].
```

Two models with the same architecture but different contracts need not induce
the same geometry. Conversely, different architectures can induce similar
downstream statistics if their contracts impose similar equivalence classes.

## Downstream Compatibility

Let T be the downstream task functional and let R be the chosen adaptation and
readout. The relevant question is not whether the encoder is large, but whether

```math
T(X)\approx
\mathcal H_{\psi}
\left(
\mathcal R_{\omega}(\Phi_{\phi^{\star}}(X))
\right)
```

can be achieved without requiring distinctions that the pretraining contract
identifies away. A probe failure establishes a limitation of the adaptation class
R and H; it does not by itself prove that T is absent from the frozen features.

The minimum report should therefore separate pretraining invariances from
downstream invariances and state which parameters are allowed to move.
