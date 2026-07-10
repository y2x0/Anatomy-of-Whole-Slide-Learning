# False Negatives And Batch Bias

A contrastive candidate can be a valid draw from the statistical marginal and
still be a false negative for the downstream semantic relation. Statistical
correctness and biological correctness are different standards.

Primary anchors:

- Chen et al. "A Simple Framework for Contrastive Learning of Visual
  Representations." ICML 2020. https://arxiv.org/abs/2002.05709
- Khosla et al. "Supervised Contrastive Learning." NeurIPS 2020.
  https://arxiv.org/abs/2004.11362

## Statistical Negative Versus Semantic Negative

In the InfoNCE experiment, an ideal statistical negative is drawn from:

```math
K^{-}
\sim
p_K
```

independently of the anchor.

Let `U` be a latent semantic or biological state. A semantic false negative
occurs when:

```math
U(K^{-})
\sim
U(A),
```

even though `K-` was independently drawn from the marginal.

This does not violate the classical MI-bound sampler. It violates the intended
semantic interpretation that every denominator candidate should be repelled.

## Class-Collision Probability

Let anchor class distribution be:

```math
p_A(c),
```

and negative proposal class distribution be:

```math
\nu(c).
```

For independent anchor and negative draws:

```math
\Pr
\left(
Y(K^{-})=Y(A)
\right)
=
\sum_c
p_A(c)\nu(c).
```

If both use the same class marginal:

```math
p_A(c)
=
\nu(c)
=
p(c),
```

then:

```math
\Pr
\left(
Y(K^{-})=Y(A)
\right)
=
\sum_c
p(c)^2.
```

Class imbalance increases this collision probability because the squared mass
of majority classes dominates.

## Morphology Collision Is Finer Than Class Collision

Observed class can be decomposed into latent morphology:

```math
U
\sim
p(u\mid Y).
```

The clinically relevant collision probability is:

```math
\Pr
\left(
U(K^{-})
\sim
U(A)
\right),
```

which need not equal class collision.

Two patches can have different slide labels but share morphology:

```math
Y(K^{-})
\ne
Y(A),
\qquad
U(K^{-})
\sim
U(A).
```

Two slides can share a diagnosis while containing different tissue states:

```math
Y(K^{-})
=
Y(A),
\qquad
U(K^{-})
\not\sim
U(A).
```

## Contaminated Negative Proposal

Let `nu_0(k)` be an intended negative proposal and let a fraction `rho` of
candidates instead come from the positive conditional:

```math
\nu_{\rho}(k\mid a)
=
(1-\rho)\nu_0(k)
+
\rho p(k\mid a),
```

with:

```math
0\le\rho<1.
```

Define the clean ratio:

```math
r_0(a,k)
=
\frac{p(k\mid a)}{\nu_0(k)}.
```

The contaminated ratio becomes:

```math
r_{\rho}(a,k)
=
\frac{p(k\mid a)}
{(1-\rho)\nu_0(k)+\rho p(k\mid a)}
=
\frac{r_0(a,k)}
{(1-\rho)+\rho r_0(a,k)}.
```

For:

```math
0<\rho<1,
```

positive contamination implies:

```math
r_{\rho}(a,k)
\le
\frac{1}{\rho}.
```

Large clean density ratios are compressed. Contamination changes the population
critic, not only its variance.

## Why Hard False Negatives Dominate

For a denominator candidate `j`, the key-gradient magnitude is proportional to:

```math
\pi_j
=
\frac{
\exp(s_j/\tau)
}{
\sum_r\exp(s_r/\tau)
}.
```

A false negative already close to the anchor has high `pi_j` and therefore
receives strong repulsion. Lower temperature further concentrates this error.

## Batch-Conditioned Proposal

SimCLR uses other augmented samples in the minibatch as negatives. Conditional
on batch-construction variables `B`, the actual proposal is:

```math
K^{-}
\sim
\nu_K(\cdot\mid B).
```

If batches are grouped by site, organ, class, patient, acquisition time, or
scanner, then:

```math
\nu_K(\cdot\mid B)
\ne
p_K.
```

The population objective is an expectation over conditional candidate tasks:

```math
\mathbb{E}_{B}
\mathbb{E}
\left[
\mathcal{L}_{N}
\mid
B
\right],
```

not necessarily the InfoNCE objective under population-marginal negatives.

## Duplicate And Correlated Candidates

The standard candidate derivation assumes conditional independence. If several
patches come from the same slide or patient:

```math
K_j
\not\perp
K_r
\mid
A,
```

then the denominator contains correlated evidence. Increasing candidate count
does not create the same information gain as adding independent marginal
draws.

The nominal count:

```math
N
```

can substantially exceed the effective number of independent comparisons.

## Positive-Pair Violations

Augmentation contrast assumes the source information intended to survive is
shared:

```math
U(V^{(1)})
=
U(V^{(2)}).
```

If a crop removes a lesion, stain transformation erases a diagnostic pattern,
or two same-slide regions contain different tissue states, this equality can
fail. The objective then enforces invariance to information that may be needed
downstream.

## C/R/G/S Placement

```text
\mathcal{G}:
    batch-conditional candidate law and latent semantic relation

\mathcal{C}:
    encoder exposed to nuisance and morphology structure

\mathcal{R}:
    embedding in which collisions are attracted or repelled

\mathcal{S}:
    generated positive/negative relation, possibly contaminated
```

## Dense Summary

The phrase `false negative` must specify a reference relation:

```text
statistical false negative:
    candidate violates the assumed sampling law

semantic false negative:
    candidate obeys the sampling law but should not be repelled biologically
```

Pathology commonly suffers from the second even when the first is absent.
