# ProtoMIL Sparse Concepts and Visual Prototypes

ProtoMIL uses sparse autoencoder coordinates as concepts and calls highly
activating patches their visual prototypes. This differs fundamentally from a
ProtoPNet metric layer.

## 1. Sparse Autoencoder

For a frozen pathology foundation-model embedding `x`,

```math
h=\mathrm{ReLU}(W_{\mathrm{enc}}x+b)
\in\mathbb R^{d_{\mathrm{hid}}},
\qquad d_{\mathrm{hid}}>d_{\mathrm{in}},
```

```math
\widehat x=W_{\mathrm{dec}}h
=\sum_{k=1}^{d_{\mathrm{hid}}}h_k f_k,
```

where `f_k` is column `k` of the decoder. Training minimizes

```math
\mathcal L_{\mathrm{SAE}}
=\|\widehat x-x\|_2^2+\lambda\|h\|_1.
```

The reconstruction term encourages retention of information needed to recover
the embedding; it does not prove information preservation. The sparsity term
encourages a small active coordinate set. Neither term guarantees that one
coordinate is monosemantic or named by a human-recognizable pathology concept.

## 2. What Is the Prototype?

For concept coordinate `k`, ProtoMIL retrieves top-activating patches

```math
\mathcal V_k^{(m)}
=\mathrm{TopM}_{x\in\mathcal D}\ h_k(x).
```

These patches are visual representatives of a learned direction. They are not
the parameter `f_k`, and they are not nearest neighbors under a learned
prototype distance. A displayed patch can contain multiple correlated tissue
features even when only one coordinate drove retrieval.

## 3. Basis Ambiguity

Without sparsity and nonnegativity, a linear autoencoder admits basis changes

```math
W_{\mathrm{dec}}h
=(W_{\mathrm{dec}}A^{-1})(Ah)
```

for invertible `A`. The ReLU and L1 penalty reduce this invariance but
do not establish unique semantic coordinates. Seed stability and top-patch
consistency are therefore part of concept validation.

## 4. Required Semantic Claim

The mathematically supported statement is:

```text
These patches strongly activate sparse coordinate k under this frozen encoder
and learned sparse dictionary.
```

Naming the coordinate requires expert inspection or external concept labels.
Even after naming, duplicated coordinates can encode similar visual patterns,
and one coordinate can remain polysemantic. Matching coordinates and their
top-patch sets across seeds is therefore necessary before treating a concept
index as stable.
