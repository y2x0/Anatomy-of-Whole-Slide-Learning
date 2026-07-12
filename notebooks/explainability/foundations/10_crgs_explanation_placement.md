# C/R/G/S Explanation Placement

## Explanation-Aware Pipeline

For predictor:

```math
F
=
\mathcal{H}
\circ
\mathcal{R}
\circ
\mathcal{C},
```

an explanation additionally requires target `T`, reference `B`, and
intervention or attribution rule `A`:

```math
E
=
\mathcal{A}
\left(
F,X,T,B
\right).
```

## C/R/G/S Roles

```math
\mathcal{C}
=
\text{where evidence can interact before explanation},
```

```math
\mathcal{R}
=
\text{which patch, region, prototype, or concept statistics survive},
```

```math
\mathcal{G}
=
\text{score geometry attributed or perturbed},
```

```math
\mathcal{S}
=
\text{labels, concepts, references, and explanation validation targets}.
```

## Method Placement

| Explanation family | Immediate target | Reference object | Principal surviving statistic |
|---|---|---|---|
| attention | readout weights | competing instances in normalization | normalized selection mass |
| gradient saliency | local score derivative | current input | first-order sensitivity |
| integrated gradients | path-integrated derivative | explicit baseline and path | score-difference allocation |
| perturbation | finite score change | removal or replacement law | intervention response |
| prototype | similarity to learned exemplars | prototype bank | nearest or weighted resemblance |
| concept | directional sensitivity in concept space | concept examples and random controls | concept-aligned derivative |
| counterfactual | minimal decision-changing input | feasibility metric and generator | model boundary displacement |

## Explanation Of Context

If `C` couples patches, patch `i` can matter through other patch states:

```math
\frac{\partial F}
{\partial x_i}
=
\sum_j
\frac{\partial F}
{\partial h_j}
\frac{\partial h_j}
{\partial x_i}.
```

An explanation restricted to the direct path `j=i` is incomplete whenever
cross terms are nonzero.

## Explanation Of Readout

For mean pooling:

```math
z
=
\frac{1}{n}
\sum_j h_j,
```

equal readout weights do not imply equal score contributions because the head
gradient and feature directions differ.

For max pooling, only coordinatewise winners survive, so a patch can matter in
some dimensions and not others.

## Explanation Of Supervision

A model trained only with slide labels identifies predictive slide evidence.
It does not identify patch truth:

```math
Y_{\mathrm{slide}}
\not\Rightarrow
\left\{
y_j
\right\}_{j=1}^{n}.
```

Heatmap validation against lesion annotations introduces an additional
supervision object absent during training.

## Unified Failure Principle

```math
\text{prediction target}
\longrightarrow
\text{reference or perturbation semantics}
\longrightarrow
\text{explanation functional}
\longrightarrow
\text{evaluation criterion}.
```

An explanation is meaningful only when every arrow is explicit. Otherwise a
method can answer one mathematical question while being interpreted as another.
