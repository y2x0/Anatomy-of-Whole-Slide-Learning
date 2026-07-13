# CAV Linear Separation Geometry

## 1. Construction

For a user-defined concept set `P_C` and random counterexample set `N`, collect
layer activations:

```math
Z_C^{(l)}=\{\phi_l(x):x\in P_C\},
\qquad
Z_N^{(l)}=\{\phi_l(x):x\in N\}.
```

Fit a linear separator

```math
q(z)=w_C^{(l)\top}z+b_C.
```

Orient the normal from the random side toward the concept side and normalize:

```math
v_C^{(l)}
=\frac{w_C^{(l)}}{\|w_C^{(l)}\|_2}.
```

The CAV is a property of the trained representation, layer, positive examples,
negative reference distribution, linear learner, and regularization.

## 2. CAV Is a Contrast

Changing the random set changes the hyperplane. In a Gaussian idealization
with common covariance,

```math
Z_C\sim\mathcal N(\mu_C,\Sigma),
\qquad
Z_N\sim\mathcal N(\mu_N,\Sigma),
```

the population linear discriminant direction is proportional to

```math
v_C\propto\Sigma^{-1}(\mu_C-\mu_N).
```

Thus the direction is not an intrinsic vector for concept `C`; it is the
whitened difference between `C` and the selected reference population.

## 3. Separability Is Not Semantics

High held-out separator accuracy establishes that the two sampled activation
distributions differ. It does not establish which attribute caused the
difference. If concept patches also differ in stain, scanner, or institution,
then

```math
v_C=v_{\mathrm{morphology}}+v_{\mathrm{nuisance}}
```

is observationally possible. Balanced counterexamples and nuisance-stratified
tests are required to isolate the intended concept.

## 4. Reparameterization

Under an invertible activation change

```math
z'=Az,
```

the covector preserving the same separating score transforms as

```math
w'=A^{-\top}w.
```

Euclidean normalization and directional derivatives depend on the coordinate
metric. CAV explanations are therefore representation-relative even when the
network function is unchanged.

## 5. Relative CAVs

To compare concepts `C` and `D`, train directly between their activation sets:

```math
v_{C,D}^{(l)}
\propto
\text{normal separating }Z_C^{(l)}\text{ from }Z_D^{(l)}.
```

This asks whether movement toward `C` rather than `D` increases a score. It is
not generally equal to subtracting two independently trained CAVs because their
negative reference distributions and separator scales differ.

