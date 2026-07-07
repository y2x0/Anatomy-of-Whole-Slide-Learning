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
S_i
\sim
Q(S\mid U_i,H_i,G_i).
```

Is $S_i$ a bag label, noisy label, partial annotation, pseudo-label, contrastive
pair, report-derived label, or mixture?

## 3. Bag Map

If using bag labels, state:

```math
Y_i
=
\Gamma(Z_i).
```

Is $\Gamma$ max, noisy-or, additive burden, threshold, distributional, or
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
\Psi_\theta(H_i,G_i,S_i).
```

Is $\Psi$ thresholding, top-k attention, teacher prediction, clustering, or
pseudo-bag construction?

## 7. Contrastive Relation

For contrastive learning, define:

```math
\mathcal{P}
=
\{(a,b):a\sim b\}.
```

What latent equivalence relation does $a\sim b$ approximate?

## 8. Identifiability Claim

State what is identifiable:

```math
S
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
same observed supervision S
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
    what signal is observed and how it is generated
```

## Dense Summary

A weak-supervision method is mathematically legible when the reader can fill:

```math
U
\xrightarrow{Q}
S
\xrightarrow{\ell}
\theta
\xrightarrow{F_\theta}
\widehat U\text{ or }\widehat Y.
```

That chain is the whole game.
