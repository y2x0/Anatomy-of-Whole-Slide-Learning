# CLAM Class-Specific Attention and Clustering

## 1. Class-Conditioned Readout

For class `c`, CLAM learns an attention score and normalized map:

```math
a_{ijc}
=\frac{\exp(s_{ijc})}{\sum_\ell\exp(s_{i\ell c})},
\qquad
z_{ic}=\sum_j a_{ijc}h_{ij}.
```

Different classes can select different patch distributions.

## 2. Instance Clustering Constraint

An instance classifier produces patch logits `q_ij`. The clustering objective
uses selected instances to create class-discriminative supervision:

```math
\mathcal L_{\mathrm{inst}}
=\mathcal L_{\mathrm{BCE}}
(q_{ij},y_{ij}^{\mathrm{pseudo}}).
```

The pseudo-label is induced from the bag label and top/bottom attention
selection; it is not a ground-truth patch annotation.

## 3. Interpretation

The class-specific map explains which patches route into class `c`'s summary. It
does not automatically explain the margin between classes. A margin credit
requires

```math
\Delta\gamma_{ij}
=\gamma_{ijc}-\gamma_{ijc'}.
```

## 4. Failure

Top-instance clustering can reinforce an early shortcut. Class-specific maps may
also disagree by design, so “the” slide heatmap is undefined without naming a
target class.

