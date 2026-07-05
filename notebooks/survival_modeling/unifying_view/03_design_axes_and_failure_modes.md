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
