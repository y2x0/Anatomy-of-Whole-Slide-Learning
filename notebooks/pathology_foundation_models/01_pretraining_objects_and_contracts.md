# Pretraining Objects and Contracts

## 1. Representation Map

Let an image patch, text description, or slide be mapped by an encoder:

```math
h=f_\phi(x),
\qquad
z=F_\phi(\{x_j,c_j\}_{j=1}^{n}).
```

A foundation model is a representation-learning procedure that chooses `phi`
without the final downstream label being the only training signal.

## 2. Pretraining Contract

Every objective specifies four objects:

```math
\mathcal P
=\left(\mathcal V,\mathcal A,\mathcal L,\mathcal G\right),
```

where:

```text
V = views or modalities;
A = positive, negative, masked, or teacher relations;
L = objective being optimized;
G = geometry in which representation similarity is measured.
```

The learned embedding preserves the distinctions rewarded by this contract.

## 3. What “Foundation” Does Not Mean

Large scale does not imply universal pathology sufficiency. A representation
can be strong for morphology but weak for molecular state, spatial layout, or
rare disease. Transfer performance depends on whether the downstream signal
lies in the retained statistic:

```math
Y\not\!\perp\!\!\!\perp H
\quad\text{is not enough;}
\quad
Y\text{ must be predictable in the allowed readout class}.
```

## 4. C/R/G/S Entry Point

```math
H=\Phi_{\phi}(X),
\qquad
z=\mathcal R(H;G),
\qquad
\widehat Y=\mathcal H(z).
```

Pretraining changes `Phi`; adaptation and downstream aggregation determine how
task information enters afterward.

