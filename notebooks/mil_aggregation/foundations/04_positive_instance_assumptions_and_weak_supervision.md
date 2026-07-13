# Positive-Instance Assumptions and Weak Supervision

## 1. Standard MIL Label Rule

The classical positive-instance assumption is

```math
y_i=1
\Longleftrightarrow
\exists j\in\{1,\ldots,n_i\}:z_{ij}=1.
```

The negative-bag rule is

```math
y_i=0
\Longrightarrow
z_{ij}=0\quad\forall j.
```

This is stronger than the usual pathology statement that a positive slide
contains at least one positive-looking patch.

## 2. Contamination

If a negative slide can contain positive-looking patches due to label noise,

```math
P(z_{ij}=1\mid y_i=0)>0,
```

then max and sparse attention can overreact. If a positive slide contains no
sampled positive patch,

```math
P(\exists j\in B_i^{\mathrm{sample}}:z_{ij}=1\mid y_i=1)<1,
```

the model cannot recover the latent evidence from that bag.

## 3. Weak Label Likelihood

The observed bag likelihood marginalizes latent instance states:

```math
p(y_i\mid B_i)
=\sum_{z_i}p(y_i\mid z_i)p(z_i\mid B_i).
```

Pooling is an architectural approximation to this latent marginal, not a
neutral replacement for it.

## 4. Design Implication

The correct readout depends on whether the label is existential, burden-based,
relational, or noisy. A model can achieve high slide AUC while learning a
different latent rule from the intended pathology semantics.

