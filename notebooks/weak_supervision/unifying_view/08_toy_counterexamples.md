# Toy Counterexamples

Weak supervision should always be tested against tiny worlds. A good
counterexample has the same observed signal but different latent truth:

```math
S^{\mathrm{obs}}(U)
=
S^{\mathrm{obs}}(U'),
\qquad
T(U)
\ne
T(U').
```

If the loss only sees
```math
S^{\mathrm{obs}}
```
, it cannot identify
```math
T(U)
```
without an
extra assumption.

## Bag Label Witness Ambiguity

Two positive slides can have different witnesses:

```math
Z
=
(1,0,0),
\qquad
Z'
=
(0,1,0).
```

Under standard OR-MIL:

```math
Y
=
\max_j Z_j
=
1,
\qquad
Y'
=
\max_j Z'_j
=
1.
```

The observed supervision is identical:

```math
S^{\mathrm{obs}}
=
Y
=
1.
```

But the witness index differs:

```math
\arg\max_j Z_j
\ne
\arg\max_j Z'_j.
```

A slide loss can identify a positive bag predictor without identifying the
true positive patch.

## Noisy Label Non-Invertibility

Suppose two clean classes produce the same noisy-label distribution:

```math
P(\widetilde Y=0\mid Y=0)
=
P(\widetilde Y=0\mid Y=1)
=
\frac{1}{2},
```

```math
P(\widetilde Y=1\mid Y=0)
=
P(\widetilde Y=1\mid Y=1)
=
\frac{1}{2}.
```

Then:

```math
P(\widetilde Y\mid Y=0)
=
P(\widetilde Y\mid Y=1).
```

No classifier trained only from
```math
\widetilde Y
```
 can recover
```math
Y
```
from the label
channel alone. This is why transition-matrix invertibility is not optional.

## Partial Label Mask Bias

Let positive patches have two subtypes:

```math
Z=1,
\qquad
T\in\{A,B\}.
```

Assume annotators only mark obvious subtype
```math
A
```
:

```math
P(M=1\mid Z=1,T=A)=1,
\qquad
P(M=1\mid Z=1,T=B)=0.
```

The observed positives estimate:

```math
P(H\mid Z=1,M=1)
=
P(H\mid Z=1,T=A),
```

not:

```math
P(H\mid Z=1).
```

A model can become excellent at annotated positives while missing the unmarked
positive subtype.

## Pseudo-Label Confirmation Bias

Suppose the true witness is patch
```math
1
```
:

```math
Z
=
(1,0,0).
```

Early attention selects patch
```math
2
```
:

```math
\mathcal{T}^{+}
=
\{2\}.
```

The pseudo-label target becomes:

```math
\widehat Z_2
=
1.
```

Training then increases:

```math
P_\theta(Z_2=1\mid h_2),
```

even though:

```math
Z_2=0.
```

The generated target is self-reinforcing because the selection rule depends on
the model.

## Contrastive Class Collision

Let an anchor and a sampled negative share latent morphology:

```math
U_i
\sim
U_k,
```

but have different observed labels:

```math
Y_i
\ne
Y_k.
```

A supervised contrastive denominator treats
```math
k
```
as negative:

```math
k
\in
\mathcal{N}(i).
```

The loss pushes:

```math
z_i^\top z_k
\downarrow,
```

even though the latent morphology says they should be close.

## Report-Derived Unit Mismatch

A case contains two slides:

```math
\mathcal{S}_{\mathrm{case}}
=
\{s_1,s_2\}.
```

The report label is positive:

```math
\widetilde Y_{\mathrm{case}}
=
1.
```

Only slide
```math
s_1
```
contains diagnostic tissue:

```math
Y_{s_1}=1,
\qquad
Y_{s_2}=0.
```

If the case label is copied to both slides:

```math
\widetilde Y_{s_1}
=
\widetilde Y_{s_2}
=
1,
```

then slide
```math
s_2
```
becomes a false-positive training example. The supervision
channel identifies a case-level fact, not a slide-local fact.

## Dense Summary

Every weak-supervision claim should survive the question:

```text
Can I construct U and U' with the same observed signal but different desired
target?
```

If yes, the method needs an inductive bias, a validation experiment, or an
explicit assumption. The loss alone does not prove the latent interpretation.
