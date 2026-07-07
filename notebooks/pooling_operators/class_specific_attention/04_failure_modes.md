# Class-Specific Attention Failure Modes

Class-specific attention adds expressive power, but it also creates new failure
modes.

## 1. Pseudo-Label Error

CLAM treats high-attention instances as positive pseudo examples:

```math
j\in T_i^{+}(c)
\quad\Rightarrow\quad
\widetilde y_{ij}^{(c)}=1.
```

This implication is not guaranteed. If attention initially focuses on artifact,
stain, scanner effect, or tissue amount, the clustering loss can reinforce that
shortcut.

## 2. Bottom-K Is Not Negative Pathology

Low attention means:

```math
a_{ij}^{(c)}\ \text{is small}.
```

It does not necessarily mean:

```math
h_{ij}\ \text{is biologically negative for class}\ c.
```

A patch can be low-attention because the model already has enough evidence
elsewhere, because the patch is redundant, or because it is hard to classify.

## 3. Class Heads Can Learn Shortcuts

Class-specific attention allows:

```math
a_i^{(1)},\ldots,a_i^{(C)}
```

to specialize. That specialization can be biological or spurious.

For example:

```text
class A head:
    tumor morphology

class B head:
    stain intensity shortcut
```

Both may separate slides in training, but only the first is robust.

## 4. Competing Explanations

A multiclass prediction can depend on relative logits:

```math
p_i^{(c)}
=
\frac{\exp(o_i^{(c)})}
{\sum_r\exp(o_i^{(r)})}.
```

A class may win because its own evidence is strong or because competing classes
are weak. A class-specific heatmap shows only one side of this comparison.

## 5. Top-K Instability

If attention rankings are close:

```math
a_{ij}^{(c)}
\approx
a_{ik}^{(c)},
```

then small parameter changes can swap top-k membership. The auxiliary loss then
trains on different pseudo instances across epochs.

## 6. Interpretability Overreach

The class-specific attention map is:

```math
j
\mapsto
a_{ij}^{(c)}.
```

It is a readout weight map, not a pathology annotation. CLAM makes attention
more class-aligned than generic ABMIL, but the attention extremes are still
learned from slide labels.

## Dense Summary

Class-specific attention is most defensible when:

```text
1. each class has distinct morphology
2. high-attention patches are stable across training
3. top-k patches pass pathologist or perturbation sanity checks
4. slide labels are not dominated by cohort shortcuts
```

Its failure mode is self-reinforcing pseudo supervision: the model may learn to
make its own early attention mistakes more separable.

