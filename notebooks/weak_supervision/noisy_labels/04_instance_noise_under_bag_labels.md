# Instance Noise Under Bag Labels

Slide-level label noise becomes complicated at the instance level.

Assume standard MIL:

```math
Y_i
=
\max_j Z_{ij}.
```

But observe:

```math
\widetilde Y_i.
```

## False Positive Bag

If:

```math
Y_i=0,
\qquad
\widetilde Y_i=1,
```

then every instance is truly negative:

```math
Z_{ij}=0
\quad
\forall j,
```

but the training objective searches for a positive witness.

A max MIL model will create:

```math
\widehat j_i
=
\arg\max_j s_\theta(h_{ij})
```

even though no true positive witness exists.

## False Negative Bag

If:

```math
Y_i=1,
\qquad
\widetilde Y_i=0,
```

then at least one instance is positive:

```math
\exists j:Z_{ij}=1,
```

but the negative-bag loss pushes every instance down.

Under noisy-or:

```math
\mathcal{L}_i^{-}
=
-
\sum_j\log(1-p_{ij}),
```

so all true positives in the mislabeled slide receive negative pressure.

## Attention Noise Amplification

For attention MIL:

```math
z_i
=
\sum_j a_{ij}h_{ij}.
```

A false positive bag can force attention to concentrate on the most
label-correlated negative morphology:

```math
a_{ij}
\uparrow
\quad
\text{for shortcut patch }j.
```

Pseudo-label methods can then turn this mistaken attention into explicit
instance supervision.

## Noise And Posterior Ambiguity

With clean positive bag labels, the posterior over witnesses is ambiguous.
With noisy labels, the model must also infer whether the bag event itself is
real:

```math
P(Z_i,Y_i\mid \widetilde Y_i,H_i).
```

This posterior includes:

```text
which instances are positive
whether the bag label is corrupted
how the corruption channel behaves
```

That is much harder than ordinary MIL.

## C/R/G/S Placement

```text
G:
    geometry may help identify implausible witnesses

C:
    contextualization can either correct or amplify noisy witnesses

R:
    max and attention can select false witnesses

S:
    corrupted slide labels generate corrupted latent-instance pressure
```

## Dense Summary

Slide noise is not just label noise at the output. It changes the latent
instance problem:

```math
\widetilde Y
\to
\text{wrong constraints on }Z.
```

False positive bags invent witnesses. False negative bags suppress true
witnesses.
