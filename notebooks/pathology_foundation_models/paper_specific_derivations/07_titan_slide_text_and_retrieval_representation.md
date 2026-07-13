# TITAN: Slide-Text and Retrieval Representation

## 1. Slide-Level Object

TITAN is a slide-level pathology foundation model. Its representation is not
just a patch vector:

```math
z_{\mathrm{slide}}
=F_{\psi}(\{h_j,c_j\}_{j=1}^{n}).
```

Text or report information provides an aligned semantic space for slide-level
retrieval and downstream tasks.

## 2. Retrieval Similarity

For slide and text embeddings,

```math
s_{ir}=\frac{z_i^{\top}t_r}
{\|z_i\|_2\|t_r\|_2}.
```

The nearest reports or slides are selected by this learned geometry. Retrieval
is an explicit readout, not merely a visualization.

## 3. Patch-to-Slide Bottleneck

If the slide encoder first selects or pools salient patches,

```math
z_i=F_{\psi}(\{h_{ij}\}_{j\in I_i}),
```

then omitted patches cannot be recovered by text alignment. The retrieval
quality of a slide model inherits its patch coverage and context operator.

## 4. Interpretive Claim

A retrieved report or slide supports semantic or morphological neighborhood
similarity under the model. It does not establish that the retrieved case shares
the same biological mechanism or outcome.

