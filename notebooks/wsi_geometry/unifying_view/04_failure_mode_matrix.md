# Failure Mode Matrix

Every geometry choice creates a blind spot.

## Matrix

| Geometry | Inductive Bias | What It Preserves | What It Erases Or Distorts | Main Failure Mode |
|---|---|---|---|---|
| none | morphology distribution is enough | feature multiset | all layout | cannot separate same-morphology different-layout slides |
| absolute coordinates | location matters | where features occur | translation and origin invariance | coordinate shortcuts |
| relative coordinates | arrangement matters | displacement and distance | absolute location | misses anatomy-specific location |
| grid | lattice neighbors matter | local grid structure | irregular tissue geometry | mask holes, aliasing, window artifacts |
| graph kNN | fixed number of neighbors matters | local topology | physical density | variable physical radius |
| graph radius | fixed physical scale matters | metric neighborhood | degree stability | variable node degree |
| graph attention | learned neighbor weighting matters | edge-conditioned context | unsupported nonedges | wrong graph support |
| hierarchy | tissue is compositional | parent-child scale structure | within-region fine variation after pooling | premature compression |
| learned topology | task reveals relations | dynamic predictive relations | fixed biological interpretation | shortcut edges and non-identifiability |

## Geometry Too Weak

Geometry is too weak when:

```math
I(Y;G\mid H)>0
```

but the model ignores or underuses
```math
G
```
.

Example:

```text
tumor-stroma interface matters, but the model only sees morphology prevalence
```

Then two slides can have:

```math
\widehat\mu_A
=
\widehat\mu_B
```

but:

```math
y_A\ne y_B.
```

No distribution-only model can solve this without geometry.

## Geometry Too Strong

Geometry is too strong when the model treats unstable spatial facts as
predictive:

```math
P_{\mathrm{train}}(Y\mid G)
\ne
P_{\mathrm{test}}(Y\mid G).
```

Examples:

```text
scanner coordinate shortcuts
tissue mask artifacts
lab-specific section placement
window origin artifacts
fixed orientation assumptions
```

## Wrong Geometry

Wrong geometry is worse than no geometry when it injects misleading context:

```math
\mathcal{N}_{G}(j)
\ne
\mathcal{N}_{G^\star}(j).
```

The context operator then computes:

```math
\widetilde h_j
=
F_\theta(\text{wrong neighborhood}).
```

This can smooth away boundaries, connect unrelated regions, or miss relevant
interfaces.

## Geometry Leakage

Geometry leakage occurs when
```math
G
```
encodes nonbiological information:

```math
G
\to
A
\to
Y,
```

where
```math
A
```
is artifact, institution, scanner, stain protocol, or tissue
processing.

The model may then learn:

```math
f(H,G)
\approx
f(G)
```

without learning relevant morphology.

## Stress Tests

A geometry-aware method should be tested by perturbing:

```text
patch order:
    should not matter unless order is explicitly meaningful

coordinate origin:
    should not matter if absolute location is irrelevant

rotation and flip:
    should not matter if orientation is irrelevant

tile size:
    should not radically alter biology

graph k or radius:
    should not make conclusions unstable

window origin:
    should not create artificial boundaries

hierarchy partition:
    should not erase rare diagnostic evidence

dynamic graph seed:
    learned topology should be stable enough to trust
```

## Dense Summary

A geometry choice should always be accompanied by:

```text
the symmetry it enforces
the spatial statistic it preserves
the shortcut it risks
the counterexample it cannot solve
```

That is the clean way to debug
```math
G
```
in C/R/G/S.
