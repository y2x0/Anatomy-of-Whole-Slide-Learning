# Cross-Attention Failure Modes

Cross-attention is powerful because a query can choose the source evidence it
needs. It fails when the query is wrong, vague, or shortcut-aligned.

## Imported Query Geometry

If a text prompt defines the query:

```math
q_c
=
f_T(p_c),
```

then the attention geometry inherits the language model geometry. Similar
prompts may be close even when pathology distinctions are not.

## Modality Dominance

If a clinical or genomic token strongly predicts the label, cross-attention may
use pathology only weakly:

```math
z
=
F(\tilde M,P)
\approx
F(\tilde M).
```

The architecture appears multimodal, but the learned statistic may be mostly
non-pathology.

For a chosen output functional `F`, a model-level dominance diagnostic is:

```math
\Delta_P
=
\left|
F(P,M)-F(P_0,M)
\right|,
\qquad
\Delta_M
=
\left|
F(P,M)-F(P,M_0)
\right|,
```

where `P_0` and `M_0` are specified replacement or ablation inputs. These
quantities should be evaluated with a distribution-preserving intervention
when possible. They measure dependence of the trained predictor, not causal
dependence of the patient outcome.

## Many-To-One Ambiguity

Several tissue regions can satisfy a query:

```math
q^\top k_{j_1}
\approx
q^\top k_{j_2}
\approx
q^\top k_{j_3}.
```

Softmax distributes mass across them, but the final value average can blur
distinct mechanisms.

The relevant quantity is the value dispersion under the query measure:

```math
\mathrm{Var}_{a(q)}(v)
=
\sum_j
a_j(q)
\left(v_j-z(q)\right)
\left(v_j-z(q)\right)^\top.
```

A large dispersion indicates that the conditional mean may hide multiple
mechanisms even when the attention map is sharply localized in separate modes.

## Query Collapse

With multiple learned queries:

```math
q_1,\ldots,q_M,
```

different queries can learn similar measures:

```math
a_{1:}
\approx
a_{2:}
\approx
\cdots
\approx
a_{M:}.
```

The model pays for multiple slots but preserves fewer distinct statistics.

## Explanation Mismatch

Cross-attention maps are often appealing:

```text
pathway token -> tissue regions
text token -> image regions
class token -> patches
```

But the map is a construction path for a representation. It is not enough to
claim a pathway, word, or class causally depends on highlighted tissue.

## Dense Summary

Cross-attention fails when:

```text
queries encode the wrong task geometry
one modality dominates the prediction
multiple queries collapse
alignment is mistaken for evidence
averaging blurs distinct source mechanisms
```
