# Transformer Context Hierarchical Readout And Long Range

## 1. TransMIL's hybrid structure

TransMIL first constructs instance tokens and then applies transformer context
before a slide-level prediction head. Let H_i be patch embeddings. A generic
self-attention block is

`math
\widetilde H_i
=
\mathrm{MHSA}
\left(
H_i+E_i
\right)
+
H_i+E_i,
`

where E_i is a positional or spatial encoding. A learned readout then maps the
contextual token set to a slide representation.

The model is not just a transformer classifier. It is a context operator plus
a bag readout.

## 2. Nystrom approximation as an operator approximation

Full self-attention forms an n by n kernel-like score matrix. TransMIL uses a
Nystrom-style approximation with m landmark tokens:

`math
\widetilde K
=
K_{nm}K_{mm}^{\dagger}K_{mn}.
`

The approximation is a low-rank surrogate for the pairwise interaction matrix,
where K_{nm} contains instance-to-landmark interactions and K_{mm} contains
landmark-to-landmark interactions.

The rank is at most m, so the context operator cannot represent arbitrary
pairwise kernels when m is small. This is a computational inductive bias in
addition to the transformer's token mixing.

## 3. Positional encoding changes the object

If E_i is built from patch coordinates, then the contextual states are

`math
\widetilde h_{ij}
=
\Phi
\left(
\{h_{ik}+e(x_{ik})\}_{k=1}^{n_i}
\right)_j.
`

Permuting H without permuting E changes the output. Simultaneously permuting
tokens and their coordinate encodings preserves the slide object.

PPEG-style local positional encoding can also inject neighborhood structure
through a grid or convolution-like operator. It is not equivalent to a
coordinate-free set transformer.

## 4. Hierarchical transformer composition

For a nested hierarchy, let H_i^{(\ell)} be tokens at level ell. A generic
bottom-up transformer is

`math
H_i^{(\ell+1)}
=
\mathcal R_{\ell}
\left(
\mathrm{Transformer}_{\ell}
\left(
H_i^{(\ell)},E_i^{(\ell)}
\right),
P_i^{(\ell)}
\right).
`

The final slide token is

`math
z_i
=
\mathcal R_{\mathrm{slide}}
\left(
H_i^{(L)}
\right).
`

HIPT instantiates this pattern with 256-token local aggregation stages and
learned [CLS] summaries, followed by a region-level slide representation. The
critical distinction from TransMIL is that HIPT's parent windows are explicit
before aggregation, whereas TransMIL typically constructs one slide-level
token sequence with positional encoding.

## 5. Long-range context after local compression

Let two fine configurations differ only inside one parent window. If their local
transformer produces the same [CLS] token, all later region-level attention sees
the same input and must produce the same output. Long-range attention cannot
restore distinctions removed by local readout.

Conversely, a flat transformer can preserve fine tokens longer but has cost that
scales with the slide sequence. Hybrid design is a choice between early
compression and expensive long-range support.

## 6. Readout choices

For contextual tokens T_i, possible readouts include

`math
z_i^{\mathrm{mean}}
=
\frac{1}{n_i}\sum_jT_{ij},
\qquad
z_i^{\mathrm{max}}
=
\max_jT_{ij},
\qquad
z_i^{\mathrm{cls}}
=
T_{i,\mathrm{CLS}},
`

or attention

`math
z_i^{\mathrm{att}}
=
\sum_j\alpha_{ij}T_{ij}.
`

The same transformer context can therefore feed different surviving statistics.
Calling all such systems TransMIL-like does not identify the final
representation.

## 7. C/R/G/S placement

`text
C: dense or Nystrom-approximated self-attention plus feed-forward blocks.

R: [CLS], mean, max, or attention readout after contextualization.

G: token order, positional encodings, PPEG/local grids, or explicit nested
   parent maps.

S: slide labels for weak supervision; self-supervised pretraining can initialize
   the token and context operators.

The hybrid signal is long-range context over non-independent tokens followed by a
task-specific statistic.
`
