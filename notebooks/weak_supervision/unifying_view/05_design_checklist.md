# Design Checklist

Every weak-supervision method should state the following mathematical choices.

## 1. Latent Variables

Define the hidden object:

```math
U_i
=
(Y_i,Z_i,B_i,A_i,R_i).
```

Which components are actually needed for the task?

## 2. Observed Signal

Define the supervision:

```math
S_i^{\mathrm{obs}}
\sim
Q(S\mid U_i,H_i,G_i).
```

Is
```math
S_i^{\mathrm{obs}}
```
a bag label, noisy label, partial annotation,
contrastive pair, report-derived label, or mixture?

Do not put model-generated pseudo-labels here unless they are genuinely
observed from outside the training algorithm.

## 3. Bag Map

If using bag labels, state:

```math
Y_i
=
\Gamma(Z_i).
```

Is
```math
\Gamma
```
max, noisy-or, additive burden, threshold, distributional, or
something else?

## 4. Noise Model

If labels are noisy, state whether:

```math
P(\widetilde Y\mid Y,H,G)
=
P(\widetilde Y\mid Y)
```

or whether noise is instance-dependent.

## 5. Missingness Model

For partial labels, state the mask:

```math
M_i
```

and the missingness assumption:

```math
P(M\mid U,H,G).
```

## 6. Pseudo-Label Rule

For pseudo-labels, define:

```math
\widehat U_i
=
\Psi_t(H_i,G_i,S_i^{\mathrm{obs}},\theta_t,\mathcal{D}).
```

Is
```math
\Psi
```
thresholding, top-k attention, teacher prediction, clustering, or
pseudo-bag construction?

State whether gradients flow through
```math
\Psi_t
```
or whether the generated target
is treated as a stopped, fixed target during the update.

## 7. Contrastive Relation

For contrastive learning, define:

```math
\mathcal{P}
=
\{(a,b):a\sim b\}.
```

What latent equivalence relation does
```math
a\sim b
```
approximate?

## 8. Identifiability Claim

State what is identifiable:

```math
S^{\mathrm{obs}}
\Rightarrow
Y?
\quad
Z?
\quad
B?
\quad
A?
```

Do not interpret an unidentifiable variable as if it were supervised.

## 9. Failure Counterexample

Every method should include one counterexample:

```text
same observed supervision S^{obs}
different latent truth U
same training loss
different desired interpretation
```

For bag labels:

```math
Z_A=(1,0,\ldots,0),
\qquad
Z_B=(0,1,0,\ldots,0),
\qquad
\Gamma(Z_A)=\Gamma(Z_B)=1.
```

If the method claims to localize the witness, it must explain why it chooses
the correct one.

## 10. C/R/G/S Placement

Finally, place the method:

```text
C:
    how supervision shapes instance context

R:
    how supervision shapes aggregation

G:
    how supervision shapes geometry

S:
    what signal is observed and which targets are generated from it
```

## Dense Summary

A weak-supervision method is mathematically legible when the reader can fill:

```math
U
\xrightarrow{Q}
S^{\mathrm{obs}}
\xrightarrow{\Psi}
\widehat U
\xrightarrow{\ell}
\theta
\xrightarrow{F_\theta}
\widehat U\text{ or }\widehat Y.
```

That chain is the whole game.
