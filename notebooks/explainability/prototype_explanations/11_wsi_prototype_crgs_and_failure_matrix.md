# WSI Prototype C/R/G/S and Failure Matrix

## 1. Unified Forward Map

For patch embeddings `H_i`, prototypes `P`, optional geometry `G_i`, and
supervision `S`, write

```math
U_i=\mathcal C(H_i;P,G_i,S),
\qquad
z_i=\mathcal R(U_i;P),
\qquad
F_i=\mathcal H(z_i).
```

Prototype explanations may attach to any arrow:

```math
H_i\xrightarrow{\text{similarity or assignment}}U_i
\xrightarrow{\text{prototype readout}}z_i
\xrightarrow{\text{task head}}F_i.
```

Explaining an earlier arrow does not automatically explain a later one.

## 2. Placement

| Method | Context `C` | Readout `R` | Geometry `G` | Supervision `S` | Surviving object |
|---|---|---|---|---|---|
| ProtoPNet | patch-prototype distances | max per prototype | latent metric | class labels plus cluster/separation losses | strongest match per prototype |
| PANTHER | GMM responsibilities | EM parameter estimation and concatenation | latent probabilistic geometry | downstream slide labels | prevalence, mean, and variance per component |
| PAMIL | prototype-instance cross-attention | instance pooling and prototype max pooling | learned bipartite similarity | branch prediction, equivalence, clustering | two branch-specific summaries |
| ProtoMIL | sparse concept encoding and MIL attention | attention-weighted concept sum | foundation-model feature geometry | reconstruction, sparsity, slide labels | sparse slide concept vector |

## 3. Failure Matrix

| Failure | Mathematical cause | Diagnostic |
|---|---|---|
| impure prototype | one latent neighborhood mixes morphologies | expert-labeled neighbor purity |
| dead prototype | activation is nearly zero for all slides | utilization distribution |
| artifact prototype | nearest matches exploit nuisance features | counterfactual artifact removal |
| redundant dictionary | correlated prototype activation functions | activation correlation and pruning |
| unstable identity | permutation and retraining nonidentifiability | matched seed stability |
| assignment-credit confusion | responsibility is interpreted as class effect | head-aware decomposition or ablation |
| prevalence-credit confusion | large mixture weight is interpreted as importance | vary component block with head fixed |
| spatial blindness | bag likelihood is permutation invariant | same distribution, different layout test |
| over-splitting | `K` exceeds stable semantic modes | merge stability and occupancy |
| under-splitting | one component covers heterogeneous tissue | within-component multimodality |

## 4. Two-Slide Sanity Check

Construct slides with identical patch multisets but different coordinates:

```math
\{h_{1j}\}_{j=1}^n=\{h_{2j}\}_{j=1}^n,
\qquad
G_1\ne G_2.
```

Every prototype method in the table remains identical unless geometry enters

```math
\mathcal C(H_i;P,G_i,S).
```

Thus a spatially smooth prototype map does not prove spatial reasoning.

## 5. Reporting Rule

Every prototype explanation should identify:

```text
prototype object; reference cohort; encoder and metric; aggregation rule;
target score; sign of credit; intervention semantics; purity; coverage;
redundancy; seed stability; spatial dependence.
```

Without this list, the word prototype hides more mathematical choices than it
reveals.
