# InfoNCE And The Mutual-Information Bound

The InfoNCE mutual-information statement is a theorem about a specific random
candidate experiment. It is not a generic interpretation of every minibatch
contrastive loss.

Primary source:

- van den Oord, Li, and Vinyals. "Representation Learning with Contrastive
  Predictive Coding." 2018. https://arxiv.org/abs/1807.03748

## Assumptions

Let:

```math
(A,K)
\sim
p_{AK}.
```

Generate a candidate experiment by sampling:

```math
I
\sim
\mathrm{Uniform}
\{1,\ldots,N\},
```

```math
K_I
\sim
p_{K\mid A}(\cdot\mid A),
```

```math
K_j
\overset{\mathrm{iid}}{\sim}
p_K,
\qquad
j\ne I.
```

The negatives must be independent draws from the population key marginal for
the standard bound below.

## InfoNCE Objective

For any critic `f`:

```math
q_f(I=i\mid A,K_{1:N})
=
\frac{
\exp(f(A,K_i))
}{
\sum_{j=1}^{N}
\exp(f(A,K_j))
}.
```

Define:

```math
\mathcal{L}_{N}(f)
=
\mathbb{E}
\left[
-\log
q_f(I\mid A,K_{1:N})
\right].
```

Then the InfoNCE bound is:

```math
I(A;K)
\ge
\log N
-
\mathcal{L}_{N}(f).
```

## Exact Classification Decomposition

Let:

```math
\mathcal{D}_N
=
(A,K_1,\ldots,K_N).
```

Cross-entropy decomposes as:

```math
\mathcal{L}_{N}(f)
=
H(I\mid\mathcal{D}_N)
+
\mathbb{E}
\left[
\mathrm{KL}
\left(
p(I\mid\mathcal{D}_N)
\,
\Vert
\,
q_f(I\mid\mathcal{D}_N)
\right)
\right].
```

Because:

```math
H(I)
=
\log N,
```

we obtain:

```math
\log N
-
\mathcal{L}_{N}(f)
=
I(I;\mathcal{D}_N)
-
\mathbb{E}
\left[
\mathrm{KL}
\left(
p(I\mid\mathcal{D}_N)
\,
\Vert
\,
q_f(I\mid\mathcal{D}_N)
\right)
\right].
```

Therefore:

```math
\log N
-
\mathcal{L}_{N}(f)
\le
I(I;\mathcal{D}_N).
```

Critic approximation error is exactly the expected posterior KL term.

## Why Index Information Is Bounded By Pair Information

Define the marginal density ratio:

```math
r(a,k)
=
\frac{p(k\mid a)}{p_K(k)}.
```

Let `Q_N` be the reference law under which all candidates are marginal:

```math
Q_N(a,k_{1:N})
=
p_A(a)
\prod_{j=1}^{N}p_K(k_j).
```

After averaging over the unknown positive index, the candidate-set marginal is:

```math
P_N(a,k_{1:N})
=
Q_N(a,k_{1:N})
\left(
\frac{1}{N}
\sum_{j=1}^{N}
r(a,k_j)
\right).
```

The difference between pair mutual information and index mutual information is:

```math
I(A;K)
-
I(I;\mathcal{D}_N)
=
\mathrm{KL}
\left(
P_N
\,
\Vert
\,
Q_N
\right)
\ge
0.
```

Hence:

```math
I(I;\mathcal{D}_N)
\le
I(A;K).
```

Combining both inequalities proves the InfoNCE bound.

## Saturation At Log N

Since cross-entropy is nonnegative:

```math
\log N
-
\mathcal{L}_{N}(f)
\le
\log N.
```

Therefore the bound cannot certify mutual information above `log N`, even if
the true mutual information is much larger.

For `N=1`:

```math
\mathcal{L}_{1}
=
0,
```

but:

```math
\log 1
-
\mathcal{L}_{1}
=
0.
```

With no competing candidate, the identification task supplies no MI lower
bound.

## Two Distinct Gaps

The bound gap is:

```math
I(A;K)
-
\left(
\log N-\mathcal{L}_{N}(f)
\right).
```

It decomposes into:

```math
\underbrace{
I(A;K)-I(I;\mathcal{D}_N)
}_{
\text{finite-candidate information gap}
}
+
\underbrace{
\mathbb{E}
\mathrm{KL}
\left(
p(I\mid\mathcal{D}_N)
\,
\Vert
\,
q_f(I\mid\mathcal{D}_N)
\right)
}_{
\text{critic approximation gap}
}.
```

More negatives can reduce the first gap, but they do not guarantee a small
critic gap.

## Proposal Mismatch

If negatives instead follow:

```math
\nu_K
\ne
p_K,
```

the optimal critic targets:

```math
\log
\frac{p(k\mid a)}{\nu_K(k)}.
```

The expected population log ratio becomes:

```math
\mathbb{E}_{p_{AK}}
\log
\frac{p(K\mid A)}{\nu_K(K)}
=
I(A;K)
+
\mathrm{KL}
\left(
p_K
\,
\Vert
\,
\nu_K
\right).
```

It can exceed mutual information. The standard `log N - loss` interpretation
is therefore invalid without correcting for the proposal.

## Dependence And False Negatives

The theorem also fails when candidate keys are not conditionally independent
or when some supposed negatives are drawn from the positive conditional:

```math
K_j
\not\sim
p_K
```

independently of `A`.

Minibatch construction, patient clustering, site-stratified sampling, and
memory queues must be analyzed as sampling mechanisms, not implementation
details.

## C/R/G/S Placement

```text
\mathcal{G}:
    joint p_AK, marginal p_K, and candidate independence

\mathcal{C}:
    critic-producing encoders

\mathcal{R}:
    finite-dimensional embedding entering f

\mathcal{S}:
    positive index and N-way cross-entropy
```

## Dense Summary

The correct statement is:

```math
\text{under iid marginal negatives,}
\qquad
I(A;K)
\ge
\log N-\mathcal{L}_{N}.
```

InfoNCE is not automatically a mutual-information estimator under arbitrary
pathology batches.
