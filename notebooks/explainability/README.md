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

prototype_explanations/
    latent exemplars, mixture components, attention anchors, and sparse concepts
    ProtoPNet similarity geometry, projection, and exact linear score credit
    similarity, presence, contribution, necessity, and sufficiency distinctions
    prototype purity, coverage, redundancy, utilization, and seed stability
    PANTHER GMM responsibilities, EM moments, and assignment maps
    PAMIL dual-branch prototype attention and agreement limits
    ProtoMIL sparse concepts, visual representatives, and exact concept credit
    prototype intervention semantics and C/R/G/S synthesis

concept_explanations/
    concept objects, explanation targets, and representation-relative semantics
    CAV separation geometry and reference-set dependence
    TCAV directional derivatives, population scores, and statistical units
    ACE multiresolution segmentation, clustering, and concept desiderata
    pathology concept validation and bias-discovery limits
    concept completeness, dependence, and basis nonidentifiability
    ProtoMIL sparse WSI concepts and exact linear score credit
    GECKO language-derived concept priors, aggregation, and contrastive alignment
    concept explanation C/R/G/S synthesis

counterfactual_explanations/
    counterfactual state spaces, optimization targets, and distance geometry
    Wachter target search, robust scaling, and closest-world dependence
    validity, plausibility, feasibility, actionability, and causality
    DiCE set-valued explanations and determinantal diversity
    HIPPO WSI bag deletion and addition counterfactuals
    GMM-CeFlow human-interpretable features and quadratic boundary projection
    surrogate fidelity and inverse-flow distortion
    diffusion counterfactual geometry and generator-relative plausibility
    MoPaDi linear and MIL-gradient morphology manipulation
    counterfactual validation and C/R/G/S synthesis

graph_spatial_explanations/
    graph explanation objects, node/edge/path targets, and spatial semantics
    HEAT leave-one-node localization and fixed-versus-rebuilt deletion
    message-path attribution and heterogeneous type/edge credit
    pseudo-label pooling and hierarchical cell-to-region localization
    BioX-CPath stain-aware attention, entropy, and interaction scores
    interpretable region graphs and integrated gradients
    spatial rendering, topology interventions, and graph sanity checks
    graph-spatial C/R/G/S synthesis and failure matrix

survival_explanations/
    risk, hazard, survival-curve, and competing-risk explanation targets
    Cox patch credit and WSI attention risk heatmaps
    graph survival node and topology effects
    MCAT genomic-guided co-attention
    C2MIL semantic intervention and topological causal subgraphs
    survival explanation validation and C/R/G/S synthesis

unifying_view/
    explanation target ladder and C/R/G/S ledger
    slide-object and resolution alignment
    faithfulness, plausibility, and causality separation
    failure matrix, sanity checks, design axes, and reporting rule
```

The Explainability family is now complete at the notebook level. Its final
cross-family audit will be performed after the remaining top-level families are
written.
