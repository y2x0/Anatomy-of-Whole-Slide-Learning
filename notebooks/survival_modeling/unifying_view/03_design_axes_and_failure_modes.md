# Design Axes And Failure Modes

## Axis 1: Risk Object

```text
scalar risk:
    simple, ranks only

discrete hazards:
    time-aware, bin-dependent

continuous hazards:
    expressive, integration-heavy

PMF:
    direct distribution, needs normalization

CIF:
    competing-risk probability
```

Failure mode:

```text
metric or interpretation does not match output object
```

## Axis 2: Slide Representation

```text
set:
    MIL

graph:
    spatial context

sequence:
    state-space or ordered transformer

hierarchy:
    region/slide/multiscale

distribution:
    prototypes

memory/retrieval:
    foundation-model reasoning
```

Failure mode:

```text
representation discards survival-relevant morphology
```

## Axis 3: Time Conditioning

```text
shared z:
    same slide statistic for all times

z(t):
    time-specific morphology

z(k):
    discrete horizon-specific morphology
```

Failure mode:

```text
early and late risk mechanisms collapse into one embedding
```

## Axis 4: Event Conditioning

```text
shared z:
    same statistic for all causes

z_c:
    cause-specific statistic

z_kc:
    time-cause-specific statistic
```

Failure mode:

```text
event-specific morphology is hidden by shared pooling
```

## Axis 5: Modality Fusion

```text
late:
    independent modality summaries

tensor:
    multiplicative interactions

co-attention:
    token-level alignment

pathway:
    biologically structured omics

hyperbolic:
    hierarchy-aware geometry
```

Failure mode:

```text
one modality dominates; fusion adds complexity without information
```

## Formal Coordinate System

The axes become mathematically useful only after the slide object and risk object
are separated. Let the representation pipeline be

```math
H_i
\xrightarrow{\mathcal C_{\theta}(\,\cdot\,;G_i)}
\widetilde H_i
\xrightarrow{\mathcal R_{\theta}(\,\cdot\,;G_i)}
z_i
\xrightarrow{\mathcal H_{\psi}}
r_i.
```

The output space of the head determines the risk object:

```math
r_i\in\mathbb R
\quad\text{(scalar risk)},
\qquad
r_i=(h_{i1},\ldots,h_{iK})\in[0,1]^K
\quad\text{(discrete hazards)},
```

```math
r_i=(p_{i1},\ldots,p_{iK})\in\Delta^{K-1}
\quad\text{(event-time PMF)},
```

```math
r_i=(F_{i1}(\tau_k),\ldots,F_{iC}(\tau_k))_{k=1}^{K}
\quad\text{(competing-risk CIFs)}.
```

For CIFs the admissible output is constrained by

```math
0\le F_{ic}(\tau_k)\le F_{ic}(\tau_{k+1}),
\qquad
\sum_{c=1}^{C}F_{ic}(\tau_k)\le 1.
```

Thus a change in the head is not merely a change in the final layer. It changes
the codomain, the normalization constraints, the likelihood, and the metrics
that are mathematically defined.

## Time And Event Conditioning As Representation Axes

A shared slide statistic uses one representation for every horizon and event:

```math
z_i=\mathcal R(\mathcal C(H_i;G_i)).
```

A horizon-conditioned model instead exposes a family of statistics:

```math
z_{ik}=\mathcal R_k(\mathcal C_k(H_i;G_i)),
\qquad
\widehat h_{ik}=\sigma(g_k(z_{ik})).
```

Event-conditioned readout adds a cause index:

```math
z_{ikc}=\mathcal R_{k,c}(\mathcal C_{k,c}(H_i;G_i)),
\qquad
\widehat F_{ic}(\tau_k)=g_{k,c}(z_{ikc}).
```

The expressive gain comes with a statistical cost. If the number of free
parameters grows from one slide vector to K or K times C vectors, the same
patient-level supervision must constrain more task-specific directions. A model
that reports a time- or event-specific representation should therefore report
the sharing rule, not only the output dimension.

## What The Representation Erases

Every context/readout pair induces an equivalence relation on slides:

```math
S_i\sim_{\mathcal C,\mathcal R}S_i'
\quad\Longleftrightarrow\quad
\mathcal R(\mathcal C(S_i;G_i))
=
\mathcal R(\mathcal C(S_i';G_i')).
```

Any survival head placed after that representation must give the same prediction
to equivalent slides. This makes the central failure mode precise: a rare region,
layout relation, time-specific pattern, or event-specific pattern cannot be
recovered after the C/R pair has identified it away.

The practical diagnostic is therefore a representation intervention, not just a
metric comparison:

```math
\Delta_{\mathrm{repr}}
=
\mathcal L(\widehat r(H))
-
\mathcal L(\widehat r(\mathsf T(H))),
```

where the transformation T deletes, relocates, duplicates, or reorders the
information claimed to be relevant. A survival method should state which such
transformations it intends to be invariant to.

## Dense Rule

Every WSI survival method can be described by:

```text
risk object
slide object
context operator
readout operator
geometry
loss
metric
failure mode
```

If any of these is unstated, the method is mathematically underspecified.
