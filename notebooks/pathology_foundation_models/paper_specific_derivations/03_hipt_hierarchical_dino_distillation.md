# HIPT: Hierarchical DINO-Style Distillation

## 1. Three Scales

HIPT builds representations at patch, region, and slide scales:

```math
h_j=f_{\phi_1}(x_j),
\qquad
h_r=f_{\phi_2}(\{h_j:j\in r\}),
\qquad
z=f_{\phi_3}(\{h_r\}_r).
```

The hierarchy makes local-to-global composition part of the representation
contract.

## 2. Teacher-Student Target

For teacher center `c`, teacher temperature `tau_t`, and student temperature
`tau_s`, a DINO-style cross-view loss is

```math
\mathcal L
=-\sum_k p_{t,k}\log p_{s,k},
```

```math
p_t=\mathrm{softmax}\left(\frac{g_t(h_t)-c}{\tau_t}\right),
\qquad
p_s=\mathrm{softmax}\left(\frac{g_s(h_s)}{\tau_s}\right).
```

The teacher follows an exponential moving average of the student.

## 3. Hierarchical Inductive Bias

The region partition determines which patch interactions are summarized before
slide-level processing. A region bottleneck can preserve long-range tissue
organization while discarding fine spatial arrangement inside a region.

## 4. Evaluation Consequence

Patch linear probing and slide-level transfer measure different objects. A
strong HIPT patch representation does not certify that its hierarchical slide
readout is optimal for WSI survival or rare lesion detection.

