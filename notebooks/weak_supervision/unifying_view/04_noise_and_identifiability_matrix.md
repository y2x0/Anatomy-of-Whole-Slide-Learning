# Noise And Identifiability Matrix

Weak supervision is limited by what the observed signal identifies.

## Matrix

| Supervision | Observed | Identifies Well | Does Not Identify | Extra Assumption Needed |
|---|---|---|---|---|
| Clean bag label | $Y=\Gamma(Z)$ | bag-level prediction | true instance labels | correct bag map and witness/burden assumption |
| Noisy bag label | $\widetilde Y$ | noisy-label prediction | clean $Y$, instances | noise channel $T$ or robust noise assumption |
| Partial labels | $M\odot U$ | labeled subset | unlabeled subset | missingness model |
| Region label | $\Gamma_R(Z_R)$ | region event | patch labels inside region | region bag map |
| Pseudo-label | $\widehat U$ | teacher/model belief | true latent state | pseudo-label accuracy or correction |
| Contrastive pair | $a\sim b$ | relation geometry | class probability or patch truth | valid positive and negative construction |
| Report-derived label | text-to-label output | report mention pattern | full slide truth | report extraction and clinical semantics |

## Identifiability Definition

A latent object $U$ is identifiable from $S$ if:

```math
P_{\theta}(S\mid H,G)
=
P_{\theta'}(S\mid H,G)
\quad
\forall H,G
\quad
\Longrightarrow
\quad
P_{\theta}(U\mid H,G)
=
P_{\theta'}(U\mid H,G).
```

Bag labels usually do not identify instance labels:

```math
P_\theta(Y\mid H)
=
P_{\theta'}(Y\mid H)
```

does not imply:

```math
P_\theta(Z_j\mid H)
=
P_{\theta'}(Z_j\mid H).
```

## Supervision Strength Ordering

A rough ordering is:

```text
full instance labels:
    strongest

partial instance or region labels:
    strong but mask-dependent

bag labels:
    weak, many latent explanations

noisy bag labels:
    weaker, corrupted bag event

pseudo-labels:
    can be strong but model-dependent

contrastive labels:
    strong for representation invariance, indirect for task truth
```

This is not a universal ranking. It depends on whether the supervision channel
matches the task.

## Fisher Information View

For parameters $\theta$, supervision $S$ provides Fisher information:

```math
\mathcal{I}_S(\theta)
=
\mathbb{E}
\left[
\nabla_\theta\log P_\theta(S\mid H,G)
\nabla_\theta\log P_\theta(S\mid H,G)^\top
\right].
```

Full latent labels would provide:

```math
\mathcal{I}_U(\theta)
=
\mathbb{E}
\left[
\nabla_\theta\log P_\theta(U\mid H,G)
\nabla_\theta\log P_\theta(U\mid H,G)^\top
\right].
```

Weak supervision loses information when:

```math
\mathcal{I}_S(\theta)
\prec
\mathcal{I}_U(\theta).
```

The missing directions correspond to unidentifiable latent structure.

## Failure Geometry

Weak supervision induces flat directions:

```math
\nabla_\theta
P_\theta(S\mid H,G)
=
0
```

while:

```math
\nabla_\theta
P_\theta(U\mid H,G)
\ne
0.
```

Along these directions, the model can change instance explanations without
changing observed-label likelihood.

## Dense Summary

The right question is:

```text
What does S identify?
```

Not:

```text
What do we hope S means?
```

If a latent variable is not identified by the supervision channel, interpreting
it requires extra assumptions or independent validation.
