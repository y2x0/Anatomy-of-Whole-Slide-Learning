# Explainability

This notebook family asks:

```text
What does the explanation actually explain?
```

An explanation can target a score, a decision, a local approximation, a
selected patch set, a learned concept, or a counterfactual intervention. These
objects are not interchangeable.

## Current Portion

```text
foundations/
    explanation targets and conditional claims
    local functionals and reference dependence
    faithfulness, completeness, sensitivity, and implementation invariance
    explanation nonidentifiability
    perturbation distributions and off-manifold evaluation
    associational versus causal explanations
    hierarchical WSI credit assignment
    stability and robustness
    deletion, insertion, localization, and sanity-check metrics
    C/R/G/S explanation placement

attention_explanations/
    attention as a normalized computational object
    weight versus signed class contribution
    exact softmax deletion and renormalization effects
    alternative attention null spaces
    correlation, permutation, and adversarial diagnostics
    model-level attention baselines and seed stability
    CLAM class-specific heatmap construction and rendering semantics
    Additive MIL exact score credit and Shapley boundary
    WSI counterexamples and C/R/G/S synthesis

gradient_saliency/
    differential saliency and local score geometry
    gradient-times-input and Taylor credit
    saturation, shattered gradients, and smoothing
    Integrated Gradients axioms, completeness, baselines, and paths
    Grad-CAM layer-dependent activation linearization
    guided backpropagation and Guided Grad-CAM
    WSI context-mediated chain-rule paths
    LRP conservation and propagation-rule dependence
    xMIL-LRP attention propagation and signed instance evidence
    perturbation faithfulness and C/R/G/S synthesis

perturbation_explanations/
    perturbation as a model intervention
    deletion, replacement, insertion, and context recomputation
    one-removed effects and subset interactions
    LIME local surrogate geometry
    SHAP coalition values and Kernel SHAP regression
    WSI coalition semantics and computational limits
    HIPPO set interventions, necessity, sufficiency, and greedy search
    perturbation faithfulness and C/R/G/S synthesis
```

Later portions will derive prototypes, concepts, counterfactuals, graph
explanations, and survival explanations from the primary papers in the private
research map.
