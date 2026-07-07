# Regularized Representation Drift

Fine-tuning changes the pretrained model:

```math
\phi
=
\phi_0+\Delta.
```

A weight-space drift penalty is:

```math
D_{\mathrm{weight}}
=
\|\phi-\phi_0\|_2^2.
```

The regularized loss is:

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda
\|\phi-\phi_0\|_2^2.
```

## Function-Space Drift

Weight distance is not the same as representation change. A function-space drift
penalty measures:

```math
D_{\mathrm{func}}
=
\mathbb{E}_{x\sim P_{\mathrm{ref}}}
\|f_\phi(x)-f_{\phi_0}(x)\|_2^2.
```

This directly controls how much the embedding changes on reference data.

## Fisher-Weighted Drift

A Fisher-style penalty weights important pretrained directions:

```math
D_{\mathrm{Fisher}}
=
\sum_k
F_k
(\phi_k-\phi_{0k})^2.
```

Large $F_k$ discourages movement in parameters important to the pretrained
objective.

## Dense Summary

Regularization encodes a prior:

```math
\phi
\approx
\phi_0.
```

It says the downstream task should adapt the model without destroying the
pretrained morphology geometry.
