# Weak Supervision

This notebook family asks:

```text
Where does supervision come from?
```

In whole-slide learning, the model often observes a weak signal $S_i$ rather
than the object it would ideally supervise. A slide label, report-derived label,
partial region annotation, pseudo-label, or contrastive pair is not the same as
fully observed patch truth.

The repository-wide decomposition is:

```math
\widetilde H_i
=
\mathcal{C}(H_i;G_i,S_i),
\qquad
z_i
=
\mathcal{R}(\widetilde H_i;G_i,S_i),
\qquad
\widehat y_i
=
\mathcal{H}(z_i).
```

This folder studies $S_i$.

## Supervision As Observation Channel

Let the ideal but unobserved variables be:

```math
U_i
=
(Y_i,\{Z_{ij}\}_{j=1}^{n_i},A_i,R_i),
```

where:

```text
Y_i:
    true slide label or outcome

Z_ij:
    latent patch or instance label

A_i:
    latent attention, assignment, or causal evidence variable

R_i:
    latent region, report, or annotation object
```

The observed supervision is generated through a channel:

```math
S_i
\sim
Q(S\mid U_i,H_i,G_i).
```

Different weak-supervision methods are different assumptions about $Q$ and
about which parts of $U_i$ can be inferred from $S_i$.

## Folder Map

```text
00_problem_setup.md
    latent variables, observation channels, risk, and identifiability

bag_labels/
    slide labels as constraints on latent instance labels
    MIL assumptions and marginal likelihoods
    positive-unlabeled bag view
    identifiability failures

partial_labels/
    sparse region and patch observations
    constraint sets and partial likelihoods
    ranking and order annotations
    semi-supervised and PU risk

noisy_labels/
    noise transition matrices
    corrected risk estimators
    robust losses and co-teaching
    instance noise under bag supervision

pseudo_labels/
    self-training as latent-variable optimization
    CLAM top/bottom-k pseudo-instance constraints
    teacher-student consistency
    pseudo-bags and distillation

contrastive_labels/
    positive-pair construction
    InfoNCE and supervised contrastive losses
    slide, patch, cluster, and multimodal positives
    false positives and sampling bias

unifying_view/
    supervision channel decomposition
    C/R/G/S supervision matrix
    paper placement matrix
    noise and identifiability matrix
```

## Core Question

Every note should answer:

```text
1. What latent variables would be supervised in the ideal dataset?
2. What supervision signal is actually observed?
3. What observation channel maps latent truth to the observed signal?
4. What loss is optimized?
5. What assumptions make the loss meaningful?
6. What is unidentifiable under this supervision?
7. What failure mode follows?
```

## Anchor Papers

- Campanella et al.: clinical-grade weak supervision from reported diagnoses.
- ABMIL: bag labels supervising an attention-weighted slide statistic.
- CLAM: slide labels plus attention-derived top/bottom-k pseudo-instance
  constraints.
- DSMIL: slide labels plus self-supervised contrastive feature learning.
- DTFD-MIL: pseudo-bags and double-tier feature distillation.
- Cluster-to-Conquer: clustering structure plus slide labels.
- SC-MIL and SCL-WC: supervised or cross-slide contrastive objectives for WSI.
- LACL: lesion-aware contrastive assumptions.
- Snorkel, co-teaching, mean teacher, noisy student, and supervised contrastive
  learning as general weak-supervision machinery.

## Dense Summary

Weak supervision is not merely "less labels." It is a different statistical
problem:

```math
\text{learn }f
\text{ from }
S
\text{ when }
S
\ne
U.
```

The central difficulty is that the model is trained through an observation
channel, not through direct access to the desired latent pathology variables.
