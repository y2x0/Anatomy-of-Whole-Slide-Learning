# Sinkhorn Balanced Codes

SwAV computes codes by entropy-regularized optimal transport between minibatch
features and prototypes. The balance constraint is the central anti-collapse
operator.

Primary anchors:

- SwAV: https://arxiv.org/abs/2006.09882
- Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport."
  NeurIPS 2013. https://arxiv.org/abs/1306.0895

## Score Matrix

Let a minibatch contain `B` normalized features:

```math
Z
=
\begin{bmatrix}
z_1&
\cdots&
z_B
\end{bmatrix}
\in
\mathbb{R}^{d\times B}.
```

Let prototypes be:

```math
C
\in
\mathbb{R}^{d\times K}.
```

The prototype-feature score matrix is:

```math
S
=
C^{\top}Z
\in
\mathbb{R}^{K\times B}.
```

## Transportation Polytope

SwAV defines:

```math
\mathcal{Q}
=
\left\{
Q
\in
\mathbb{R}_{+}^{K\times B}
:
Q\mathbf{1}_B
=
\frac{1}{K}\mathbf{1}_K,
\quad
Q^{\top}\mathbf{1}_K
=
\frac{1}{B}\mathbf{1}_B
\right\}.
```

The total mass is:

```math
\mathbf{1}_K^{\top}
Q
\mathbf{1}_B
=
1.
```

Each prototype receives mass `1/K`; each sample contributes mass
`1/B`.

## Entropy-Regularized Assignment

Define:

```math
H(Q)
=
-
\sum_{k=1}^{K}
\sum_{b=1}^{B}
Q_{kb}
\log
Q_{kb}.
```

SwAV solves:

```math
Q^{\star}
\in
\arg\max_{
Q\in\mathcal{Q}
}
\left\{
\mathrm{Tr}
\left(
Q^{\top}S
\right)
+
\varepsilon H(Q)
\right\}.
```

The first term assigns mass to high-similarity prototype-feature pairs. The
second discourages an excessively sharp transport plan.

## KKT Scaling Form

Introduce row and column multipliers `\alpha_k` and `\beta_b`. For
an interior optimum:

```math
0
=
\frac{
\partial
}{
\partial Q_{kb}
}
\left[
\sum_{r,j}
Q_{rj}S_{rj}
-
\varepsilon
\sum_{r,j}
Q_{rj}
\log
Q_{rj}
+
\text{marginal terms}
\right].
```

This gives:

```math
S_{kb}
-
\varepsilon
\left(
\log Q_{kb}
+
1
\right)
+
\alpha_k
+
\beta_b
=
0.
```

Hence:

```math
Q_{kb}^{\star}
=
u_k
\exp
\left(
S_{kb}/\varepsilon
\right)
v_b
```

for positive scaling vectors `u` and `v`. In matrix form:

```math
Q^{\star}
=
\mathrm{Diag}(u)
\exp
\left(
S/\varepsilon
\right)
\mathrm{Diag}(v).
```

The exponential is elementwise.

## Sinkhorn-Knopp Iterations

Let:

```math
K_0
=
\exp
\left(
S/\varepsilon
\right).
```

Alternating row and column rescaling seeks the required marginals:

```math
u
\leftarrow
\frac{
\frac{1}{K}\mathbf{1}_K
}{
K_0v
},
```

```math
v
\leftarrow
\frac{
\frac{1}{B}\mathbf{1}_B
}{
K_0^{\top}u
},
```

where division is elementwise.

SwAV reports that three iterations are sufficient in its training setup.

## Transport Plan Versus Per-Sample Code

The transport column sums to:

```math
\sum_{k=1}^{K}
Q_{kb}^{\star}
=
\frac{1}{B}.
```

Cross-entropy requires a per-sample probability vector summing to one. Define:

```math
q_b
=
BQ_{:b}^{\star}.
```

Then:

```math
q_b
\in
\Delta^{K-1}.
```

The transport matrix `Q` and training code `q_b` differ by column
normalization. Conflating them creates a factor-of-`B` notation error.

## Balance Identity

The average per-sample code is:

```math
\frac{1}{B}
\sum_{b=1}^{B}
q_b
=
\frac{1}{B}
\sum_{b=1}^{B}
BQ_{:b}^{\star}
=
Q^{\star}\mathbf{1}_B
=
\frac{1}{K}\mathbf{1}_K.
```

Thus the batch-average target distribution is exactly uniform over
prototypes.

If all samples attempted the same one-hot code:

```math
q_b
=
e_1,
\qquad
\forall b,
```

the average would be:

```math
e_1
\ne
\frac{1}{K}\mathbf{1}_K.
```

The balance constraint excludes this categorical collapse.

## Entropy Limits

As:

```math
\varepsilon
\rightarrow
0^{+},
```

the optimizer approaches a maximum-score balanced transport plan. Assignments
become sharp when the score geometry supports a near-discrete solution.

As:

```math
\varepsilon
\rightarrow
\infty,
```

entropy dominates and:

```math
Q_{kb}^{\star}
\rightarrow
\frac{1}{KB},
```

so:

```math
q_b
\rightarrow
\frac{1}{K}\mathbf{1}_K.
```

Every sample then receives the same uniform code. Balance alone prevents
single-prototype use but high entropy can still destroy sample-specific
information.

## Hard Feasibility And Small Batches

A hard one-hot assignment with exactly equal prototype counts requires:

```math
\frac{B}{K}
\in
\mathbb{N}.
```

If:

```math
B
<
K,
```

every prototype cannot receive a distinct sample under a hard assignment.

The continuous transportation polytope remains nonempty because a sample can
split mass across prototypes. The practical problem is that exact balance then
forces highly fractional codes, weakening categorical discrimination.

## Queue For Assignment Support

For small current batch `B`, SwAV augments the assignment feature matrix
with stored features:

```math
\widetilde Z
=
\begin{bmatrix}
Z_{\mathrm{current}}
&
Z_{\mathrm{queue}}
\end{bmatrix}.
```

Sinkhorn is solved on `\widetilde Z`, but only current-sample codes enter
the training loss.

The queue changes the empirical prototype marginals used to assign current
features:

```math
q_b
=
q_b
\left(
Z_{\mathrm{current}},
Z_{\mathrm{queue}}
\right).
```

It is assignment context, not a feature-negative dictionary.

## No-Gradient Code Path

SwAV computes:

```math
Q^{\star}
=
\mathrm{Sinkhorn}
\left(
C^{\top}Z
\right)
```

under stop-gradient for the current update. Therefore:

```math
\nabla_{\theta}
Q^{\star}
=
0
```

inside backpropagation, even though `Q^{\star}` is recomputed after
parameters change.

The update does not differentiate through the optimal transport solver.

## Batch-Prior Assumption

Balance imposes:

```math
p_{\mathrm{target}}
\left(
C=k
\right)
=
\frac{1}{K}
```

within assignment context. If true tissue prevalence is:

```math
p_{\mathrm{tissue}}
\left(
U=k
\right)
\ne
\frac{1}{K},
```

prototype usage is a regularized representation prior, not an estimate of
tissue prevalence.

## C/R/G/S Placement

```text
\mathcal{G}:
    complete prototype-feature bipartite graph with prescribed row and column
    marginals

\mathcal{C}:
    entropy-regularized transport solved by Sinkhorn scaling

\mathcal{R}:
    balanced soft per-sample prototype code

\mathcal{S}:
    stopped code used as a cross-view prediction target
```

## Failure Principle

Sinkhorn prevents prototype starvation by imposing a uniform usage prior. It
does not prove that the resulting prototypes match natural or biological
cluster frequencies.
