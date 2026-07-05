# Deep Mathematical Review: `slide_representations/`

**Branch reviewed:** `codex/slide-representations-review`  
**Scope:** All 18 notes in `notebooks/slide_representations/` (~2,000 lines)  
**Date:** July 5, 2026

**Note on papers:** There are no PDF papers checked into the repo. Anchor citations live in each folder's README. This review treats the notes as the mathematical object and checks them against the cited literature.

---

## Executive Assessment

**Verdict:** This is a strong, coherent first pass. The notes do real mathematical work: they define objects, state symmetries, identify surviving statistics, and derive failure modes from those symmetries rather than from implementation anecdotes.

**Strengths:**

- Clean separation of **representation** vs **aggregation/readout** (central and often blurred in the WSI literature)
- Empirical-measure framing for sets is the right abstraction
- Symmetry groups (permutation invariance / equivariance) are stated correctly
- "What survives?" is the right organizing question throughout
- Failure modes follow logically from underspecified symmetries

**Main gaps for "deep mathematical depth":**

- No formal **unification** across set / sequence / graph (they are related, not disjoint)
- Missing **expressivity / approximation** results (Deep Sets theorem, WL limits for GNNs, attention-as-complete-graph)
- Anchor papers are listed but not **placed** in the C/R/G/S decomposition (unlike `survival_modeling/unifying_view/04_paper_placement_matrix.md`)
- Several families mentioned in the top-level README (hierarchy, distribution, retrieval, foundation latents) are not yet written

---

## Framework Alignment with the Repo's Master Decomposition

The root README defines:

```math
\widetilde H = \mathcal{C}(H; G, S), \quad z = \mathcal{R}(\widetilde H), \quad \hat y = \mathcal{H}(z)
```

The slide-representation notes implicitly use this but don't name it consistently:

| Family | Representation object $\mathcal{X}_i$ | Where $\mathcal{C}$ lives | Where $\mathcal{R}$ lives |
|--------|------------------------------------------|-----------------------------|-----------------------------|
| **Set** | Empirical measure $\mu_i = \frac{1}{n}\sum_j \delta_{h_{ij}}$ | Often fused with readout (Deep Sets, attention, Set Transformer) | Mean / max / attention / pool |
| **Sequence** | Ordered tuple $(h_{\sigma(1)}, \ldots, h_{\sigma(n)})$ | RNN / Transformer / SSM scan | Final state / pool |
| **Graph** | $(V, E, H, A)$ | Message passing (GNN layers) | Mean / attention / hierarchical pool |

**Recommendation:** Add one unifying note (e.g. `01_crgs_placement_for_representations.md`) that maps every method to $(\mathcal{C}, \mathcal{R}, G, S)$. That would connect this folder to `survival_modeling/unifying_view/`.

---

## 1. Set Representations — Mathematical Review

### What is correct and well done

**Empirical measure view** (`01_slide_as_set.md`) is the right formalization:

```math
\mu_i = \frac{1}{n_i}\sum_{j=1}^{n_i}\delta_{h_{ij}}, \quad f(H_i) = F(\mu_i)
```

Mean pooling is the first moment $\mathbb{E}_{\mu_i}[h]$. Attention defines a reweighted measure $\nu_i \propto a_\theta(h)\,\mu_i$. Prototype prevalence is a soft histogram over learned morphologies. All correct.

**Deep Sets template** (`02_deep_sets_and_mil.md`):

```math
f(\{h_j\}) = \rho\left(\sum_j \phi(h_j)\right)
```

This matches Zaheer et al. (NeurIPS 2017). The decomposition instance-transform → sum → bag head is exactly the MIL architecture family.

**Permutation invariance of attention MIL** is correctly argued: scores are computed per-instance and normalized over the bag, so the statistic is invariant.

**Set Transformer** equivariance before invariant pooling is correctly stated:

```math
\widetilde H_\pi = P_\pi \widetilde H
```

### Subtleties worth adding (mathematical depth)

1. **Deep Sets is an approximation theorem, not an exact factorization.**  
   Zaheer et al. show that *continuous* permutation-invariant functions on compact sets can be *universally approximated* by $\rho(\sum \phi(x))$. The notes read as if all invariant maps "can be written as" this form — true up to approximation, not necessarily exact for finite architectures or discontinuous targets.

2. **Max-MIL is not Deep-Set decomposable.**  
   The notes correctly separate max from sum-decomposable readouts. Worth stating explicitly: $\max_j g(h_j)$ is invariant but not of the form $\rho(\sum \phi(h_j))$. It corresponds to an **extreme-value functional** of $\mu_i$, not a moment.

3. **Set attention = complete-graph message passing (one layer).**  
   For one self-attention layer over all pairs $(j,k)$:

   ```math
   \tilde h_j = \sum_k \text{softmax}_k(q_j^\top k_k)\, v_k
   ```

   This is message passing on the **complete graph** $K_n$ with learned edge weights. This unifies set and graph families and should appear in `03_statistics_that_survive.md`.

4. **MIL identifiability.**  
   Weak supervision $\ell(f(H_i), y_i)$ with $y_i \sim p(y \mid H_i)$ does not identify patch labels without extra assumptions (e.g. instance-space MIL: $\exists j: y_{ij}=1 \Rightarrow y_i=1$). The notes mention this qualitatively; a one-line formal statement would sharpen the math.

5. **Prototype prevalence notation.**  
   $\int q_m(h)\,d\mu_i(h) = \frac{1}{n_i}\sum_j q_m(h_{ij})$ for soft assignments $q_m$. Correct as written; could note connection to **mixture models** and **Wasserstein barycenters** if going deeper.

### Paper alignment (anchor list)

| Paper | Placement in notes | Assessment |
|-------|-------------------|------------|
| Deep Sets (Zaheer 2017) | `02_deep_sets_and_mil.md` | Correct template; add approximation caveat |
| Set Transformer (Lee 2019) | `02_deep_sets_and_mil.md` | Correct equivariance → pool; missing inducing-point / PMA detail |
| Order Matters (Vinyals 2016) | Cited in README only | Not used in set notes (belongs in sequence) |
| Classic MIL (Ilse et al. gated attention) | Implicit in attention MIL | Gated variant $a \odot \tanh(v) + (1-a)\odot \sigma(v)$ not mentioned |

---

## 2. Sequence Representations — Mathematical Review

### What is correct and well done

**Order as constructed structure** is the key insight. A WSI has no canonical 1D order; $\sigma$ is a **compression map** from 2D layout to 1D index. The notes state this clearly.

**Positional encoding distinction** (sequence index $p_j$ vs physical coordinate $\phi(c_{ij})$) is essential and correctly emphasized.

**Without positional encoding, self-attention is permutation equivariant** — correct and often overlooked.

**State-space complexity** $O(n)$ vs transformer $O(n^2)$ — correct motivation for WSI-scale bags.

**Ordering and geometry** (`03_ordering_and_geometry.md`):

- Raster order discontinuities at row boundaries — correct
- Space-filling curves preserve locality *approximately* — correctly hedged with "often small," not "always"
- Learned reordering makes $\sigma_i$ model-dependent — important and correctly flagged

### Subtleties worth adding

1. **No 1D order preserves 2D neighborhoods.**  
   For any surjection from 2D grid to 1D sequence, there exist adjacent grid cells with arbitrarily large sequence separation (a consequence of dimension reduction). This is a **topological/embedding impossibility** result worth one sentence.

2. **Sequence locality ≠ spatial locality — quantify it.**  
   Define a **locality distortion** metric, e.g.:

   ```math
   D(\sigma) = \mathbb{E}_{(u,v)\in E_{\text{spatial}}}\big[|\sigma(u)-\sigma(v)|\big]
   ```

   Hilbert/Morton orders minimize this heuristically; raster order does not.

3. **Bidirectional SSM/scan** reduces but does not remove order sensitivity.  
   Unless the model is invariant to $\sigma$ (e.g. by averaging over orders or using set-equivariant structure), order remains part of the hypothesis class.

4. **MambaMIL (Yang et al. 2024).**  
   README cites it; notes mention "MambaMIL-style motivation" but don't formalize **selective SSM** (input-dependent $A_j, B_j, C_j$). The `02_sequence_operators.md` selective SSM block is correct; tie it explicitly to MambaMIL's reordering step as $\sigma_i = f_\theta(H_i, C_i)$.

### Paper alignment

| Paper | Notes | Gap |
|-------|-------|-----|
| Order Matters (Vinyals 2016) | README only | Should appear in `01_slide_as_sequence.md` as "sort → process → pool" |
| MambaMIL (Yang 2024) | `03_ordering_and_geometry.md` | Needs explicit algorithmic placement |

---

## 3. Graph Representations — Mathematical Review

### What is correct and well done

**Graph object** $(V, E, H, A)$ with optional edge features — standard and correct.

**Permutation equivariance of message passing:**

```math
\text{GNN}(PH, PAP^\top) = P\,\text{GNN}(H, A)
```

Correct.

**Receptive field after $L$ layers = $L$-hop neighborhood** — correct and important for WSI (patch size × hops = effective context radius).

**Graph construction as inductive bias** (`03_graph_construction.md`) — excellent. kNN vs radius vs similarity vs learned vs heterogeneous graphs are the right taxonomy.

**Failure modes** — oversmoothing, oversquashing, topology error, degree bias, readout bottleneck — all standard and correctly tied to the math.

### Subtleties worth adding

1. **Expressivity: 1-WL test.**  
   Standard MPNNs are at most as powerful as the 1-WL graph isomorphism test. Many WSI graphs (patch kNN on near-planar layouts) may be **WL-indistinguishable** despite different biology. Worth one paragraph in `02_message_passing.md`.

2. **Patch-GCN (Chen et al. 2021) placement.**  
   Cited in README. Method = patch nodes + coordinate kNN + GCN + survival head. Maps cleanly to:
   - $G$: 2D coordinates → kNN edges
   - $\mathcal{C}$: GCN message passing
   - $\mathcal{R}$: graph readout (mean/attention)
   - $\mathcal{H}$: Cox scalar risk  
   This should be a worked example in a paper-placement note.

3. **HACT (Pati et al. 2022)** — hierarchical cell-to-tissue graph. Notes mention "HACT-style" briefly in heterogeneous graphs; deserve a subsection: two-level graph with cell nodes, tissue super-nodes, containment edges.

4. **WiKG / dynamic graph (Li et al. CVPR 2024)** — cited in README; notes mention "WiKG-style" in passing. Dynamic adjacency $A_{uv} = g_\theta(h_u, h_v, c_u, c_v)$ makes the representation **end-to-end learned**; failure mode "learned topology instability" correctly anticipates this.

5. **Oversquashing bound (informal).**  
   If information from a distant set $B$ must pass through a cut of size $|cut|$, messages of dimension $d$ create a bottleneck. Recent literature (Topping et al., Di Giovanni et al.) formalizes this; even a sentence would add depth.

6. **Readout discards topology.**  
   Mean readout after GNN is still $\frac{1}{|V|}\sum_v h_v^{(L)}$ — a **set statistic over contextualized nodes**. The graph shaped $h_v^{(L)}$ but the final object may be set-equivalent. The notes state this; could add: **two slides with identical multiset of final node features but different graph structure are indistinguishable under mean readout.**

### Paper alignment

| Paper | Notes | Assessment |
|-------|-------|------------|
| Patch-GCN (Chen 2021) | README | Not worked through in body text |
| HACT (Pati 2022) | One line in heterogeneous section | Needs expansion |
| WiKG (Li 2024) | One line | Needs expansion |

---

## Cross-Family Unification (Missing but Central)

The three families are **not mutually exclusive**. Add a note making these equivalences explicit:

```text
Set attention (full)      ≅  1-layer MPNN on complete graph K_n
Sequence + full attention ≅  MPNN on path graph (with PE on edges)
kNN graph + L-layer GNN   ≅  L-hop spatial smoothing then set readout
Set (no interaction)      ≅  Graph with E = ∅, L = 0
```

**Design implication:** Choosing set vs graph vs sequence is choosing **adjacency structure** for $\mathcal{C}$:

| Adjacency | Model class |
|-----------|-------------|
| None | Deep Sets / mean MIL |
| Complete | Set Transformer / full self-attention |
| kNN / radius | Patch-GCN-style GNN |
| Path / learned order | MambaMIL / sequence SSM |
| Hierarchical | HACT / region graphs |

This table would be the mathematical "payoff" of the whole folder.

---

## Mathematical Correctness Audit (Specific Items)

| Claim | Location | Status |
|-------|----------|--------|
| $f(H) = f(PH)$ for all permutations $P$ | Set notes | ✓ Correct |
| Mean = first moment of $\mu_i$ | Set notes | ✓ Correct |
| Attention MIL invariant | Set notes | ✓ Correct |
| Max not equivalent to mean | Set notes | ✓ Correct (implicit) |
| No PE → attention equivariant | Sequence notes | ✓ Correct |
| SSM recurrence $s_{j+1} = A_j s_j + B_j h_j$ | Sequence notes | ✓ Correct |
| GNN equivariance under node relabeling | Graph notes | ✓ Correct |
| L layers → L-hop receptive field | Graph notes | ✓ Correct |
| Deep Sets exact representation for all invariant maps | Set notes | ⚠ Imprecise — universal approximation on compact sets |
| Prototype $\int q_m(h)\,d\mu$ | Set notes | ✓ Correct for empirical measure |
| Fine-Gray / survival | N/A in this folder | — |

No **errors** found that would mislead a reader; only **imprecisions** and **missing depth**.

---

## Failure-Mode Logic (Quality Check)

The failure-mode notes are the strongest part mathematically. Each follows the pattern:

1. State symmetry / inductive bias
2. Identify information not representable under that symmetry
3. Give concrete WSI pathology examples
4. Provide diagnostic questions

**Set:** geometry loss, sparse positives diluted by mean, attention collapse, bag shortcuts — all valid consequences of permutation invariance without coordinate features.

**Sequence:** arbitrary order artifacts, raster discontinuities, long-range compression, direction bias — all valid consequences of treating constructed order as structure.

**Graph:** topology error, oversmoothing/squashing, degree bias, readout bottleneck — all valid consequences of wrong or over-smoothed adjacency.

This is exactly the "mathematically visible failure mode" standard promised in the root README.

---

## Comparison to Anchor Papers (No PDFs in Repo)

Since papers are not checked into the folders, here is how README citations map to what each paper **actually assumes mathematically**:

| Paper | Mathematical slide object | Key stat that survives | Notes match? |
|-------|---------------------------|------------------------|--------------|
| **Deep Sets** | Unordered set | $\rho(\sum \phi(h))$ | ✓ |
| **Set Transformer** | Set with pairwise attention | Pool of equivariant states | ✓ |
| **Order Matters** | Sorted set → RNN | Order-dependent hidden state | Partially (not in body) |
| **MambaMIL** | Reordered sequence | SSM-compressed trajectory | Partially |
| **Patch-GCN** | kNN patch graph | GNN-contextualized mean/attention | Cited, not worked |
| **HACT** | Hierarchical cell-tissue graph | Multi-scale graph readout | One line only |
| **WiKG** | Learned dynamic graph | Knowledge-aware attention states | One line only |

---

## Recommended Additions (Priority Order)

1. **`unifying_view/` for slide representations** — C/R/G/S placement + paper matrix (mirror survival folder)
2. **Set ↔ complete graph equivalence** — one of the highest-value missing insights
3. **Deep Sets approximation theorem** — precise statement with compactness / continuity conditions
4. **GNN 1-WL expressivity** — one paragraph + implication for patch graphs
5. **Worked examples** — Patch-GCN, CLAM, MambaMIL, HACT as full forward maps with surviving statistics
6. **Distribution / hierarchy chapters** — promised in README but absent
7. **Robustness diagnostics as math** — e.g. "report performance under random patch permutations" as a test of whether set symmetry matches the task

---

## Bottom Line

The `slide_representations/` notes are **mathematically sound, well-structured, and unusually clear** for computational pathology. They correctly center on symmetries, empirical measures, and surviving statistics. For the stated goal of "deep mathematical depth," the main work remaining is:

- **Unification** across the three families
- **Expressivity / approximation** results
- **Paper placement** with full forward-map derivations
- **Extension** to hierarchy and distributional representations already promised in the README

There are no PDFs in the folders — only bibliographic pointers.
