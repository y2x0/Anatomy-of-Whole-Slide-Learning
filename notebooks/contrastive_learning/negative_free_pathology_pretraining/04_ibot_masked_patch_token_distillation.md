# iBOT Masked Patch-Token Distillation

Primary anchor:

- Zhou et al. "iBOT: Image BERT Pre-Training with an Online Tokenizer." ICLR
  2022. https://arxiv.org/abs/2111.07832

## Masked Student And Visible Teacher

Tokenize image view `u` into `N` patch tokens. Let:

```math
m_i
\in
\left\{0,1\right\}
```

indicate that token `i` is masked. The student input is:

```math
\widehat u_i
=
\left(
1-m_i
\right)u_i
+
m_i e_{\mathrm{MASK}}.
```

The teacher sees unmasked `u`, while the student sees `\widehat u`.

## Online Token Distributions

Teacher patch target:

```math
p_{t,i}
=
P_{\theta_t}^{\mathrm{patch}}(u_i).
```

Student patch prediction:

```math
p_{s,i}
=
P_{\theta_s}^{\mathrm{patch}}(\widehat u_i).
```

The teacher backbone and patch projection head form an online tokenizer. There
is no fixed offline codebook.

## Exact Masked Objective

```math
\mathcal{L}_{\mathrm{MIM}}
=
-\sum_{i=1}^{N}
m_i
p_{t,i}^{\top}
\log p_{s,i}.
```

iBOT symmetrizes across two augmented views. It also retains a class-token
self-distillation term:

```math
\mathcal{L}_{\mathrm{iBOT}}
=
\mathcal{L}_{\mathrm{CLS}}
+
\lambda_{\mathrm{patch}}
\mathcal{L}_{\mathrm{MIM}}.
```

The patch loss is evaluated only at masked positions.

## Conditional Prediction Interpretation

For masked index set `M` and visible set `O`, the student approximates:

```math
p
\left(
Z_i^{t}
\mid
X_O,
i\in M
\right),
```

where `Z_i^t` is the teacher's categorical token target. It does not reconstruct
raw RGB pixels.

## Exact Student Gradient

For student pre-softmax logits `a_{s,i}`:

```math
\frac{\partial\mathcal{L}_{\mathrm{MIM}}}
{\partial a_{s,i}^{(k)}}
=
\frac{m_i}{\tau_s}
\left(
p_{s,i}^{(k)}
-
p_{t,i}^{(k)}
\right).
```

Visible positions receive no direct patch-token loss, though they influence
masked predictions through self-attention.

## Mask Ratio Tradeoff

Let mask probability be `\rho`. The expected direct supervised positions are:

```math
\mathbb{E}[|M|]
=
\rho N.
```

Low `\rho` gives many visible clues but few prediction targets. High `\rho`
gives many targets but weak conditional evidence. The optimum depends on
spatial redundancy and tissue scale.

## Histology Ambiguity

Two masked tissue patches can have identical visible context but distinct
microstructure:

```math
X_O^{(a)}
=
X_O^{(b)},
\qquad
Z_i^{t,(a)}
\ne
Z_i^{t,(b)}.
```

The Bayes predictor averages the conditional target distribution. Masked
distillation can therefore favor context-predictable morphology over genuinely
unpredictable rare cellular detail.
