# C/R/G/S Adaptation Matrix

Foundation adaptation changes where task supervision enters C/R/G/S.

| Adaptation | C | R | G | S | What Moves |
|---|---|---|---|---|---|
| Linear probe | frozen | frozen or fixed | inherited | labels train head | omega |
| Frozen feature MIL | post-encoder context may train | readout trains | optional downstream geometry | labels train C/R/H | psi, omega |
| Prompt tuning | prompt-conditioned context | prompt score or head | inherited visual-text geometry | labels/words train prompts | pi, omega |
| LoRA/adapters | low-rank or residual context update | depends on layer placement | mostly inherited | labels train small modules | Delta_eta, omega |
| Full fine-tuning | encoder context moves | readout moves | geometry can move | labels train all selected weights | phi, psi, omega |
| Retrieval-memory | retrieved cases condition context | vote/prototype/retrieval head | memory graph | labels live in memory | memory M and metric |

## Readout Versus Encoder Adaptation

Frozen features plus trainable readout:

```math
h_j
=
f_{\phi_0}(x_j),
\qquad
z
=
\mathcal{R}_\psi(\{h_j\}).
```

Fine-tuning:

```math
h_j
=
f_{\phi_0+\Delta}(x_j),
\qquad
z
=
\mathcal{R}_\psi(\{h_j\}).
```

These are not equivalent. The first changes the statistic of fixed features.
The second changes the features themselves.

## Prompt Versus Head

Prompt classifier:

```math
s_c
=
f_I(X)^\top f_T(p_c).
```

Linear head:

```math
s_c
=
w_c^\top f_I(X)+b_c.
```

Both are linear in the image embedding, but prompt classifiers constrain the
weight vector to lie in the text-encoder image of a prompt:

```math
w_c
=
f_T(p_c).
```

## Dense Summary

The same foundation model can be many different mathematical methods depending
on which part of C/R/G/S is allowed to move.
