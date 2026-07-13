# Pathology Concept Validation and Bias Discovery

Sauter et al. evaluate ACE in digital histopathology as a global, model-level
tool for discovering known induced biases, complemented by local Guided
Grad-CAM maps.

## 1. Validation Target

Let `B` denote whether a trained classifier contains a designed bias and let
`E` be the concept explanation. Technical validity for that experiment asks
whether

```math
p(E\text{ indicates bias}\mid B=1)
```

is high while false indication under `B=0` is low. This is an explanation-level
detection problem, not a test of diagnostic accuracy or causal morphology.

## 2. Bias Families

The experiments examine class sampling ratio bias, measurement bias, sampling
bias, and class-correlated bias. Abstractly, let nuisance variable `U` and label
`Y` satisfy

```math
p_{\mathrm{train}}(U\mid Y)
\ne
p_{\mathrm{target}}(U\mid Y).
```

If a discovered concept `C_U` separates nuisance states and has high class
sensitivity, ACE can reveal reliance on this shifted association.

## 3. A Stronger Diagnostic Design

For a candidate nuisance concept, stratify concept scores by nuisance and
disease status:

```math
\theta_{u,y}
=\mathbb P
[S_{C_U,y,l}(X)>0\mid U=u,Y=y].
```

A model using `U` should show systematic changes across `u` after holding `y`
fixed. Cohort-level concept rankings without this stratification can confuse
disease morphology with correlated acquisition effects.

## 4. Why Local Maps Remain Necessary

ACE aggregates across examples and may reveal that a concept matters globally,
but it does not localize where that concept affected a particular image.
Guided Grad-CAM localizes activation but requires the observer to infer the
semantic abstraction. Their targets are complementary:

```math
\text{ACE: global concept sensitivity},
\qquad
\text{Grad-CAM: local spatial attribution}.
```

Agreement increases descriptive confidence, but two associational methods
agreeing does not establish causality.

## 5. Pathology-Specific Failure Modes

- SLIC can mistake the shape of an artificial mask for a concept.
- Stain and scanner variation can dominate activation-space clusters.
- Concept patches can mix multiple cell and tissue states.
- Location-dependent bias is difficult for a location-free concept inventory.
- Cluster count and concept size change duplicates and mixed clusters.
- A bar chart of global scores can be less intuitive than expected for
  dermatopathology.

The paper supports ACE as a useful bias-discovery instrument in its evaluated
settings, not as a universal validator of pathology explanations.

