# Brutal Review — `slide_representations/` Notebooks

**Branch reviewed:** `codex/slide-representations-review`  
**Date:** July 7, 2026  
**Overall score: 7/10**

Ambitious scope. The unifying view is genuinely strong. New families add breadth but repeat set material in places, and a few math slips remain. Not bad — but not uniformly deep.

---

## What's Actually Good

**`unifying_view/` is the spine.** The adjacency-as-context framing and C/R/G/S placement finally connect set, sequence, and graph as choices of neighbor structure, not separate model catalogs. This is the highest-value addition in the branch.

**Set and graph updates are solid.** Deep Sets is now framed with approximation caveats (not exact factorization). Max pooling is correctly separated as an extreme-value statistic, with the log-sum-exp limit noted. Gated attention is included. Set attention is linked to complete-graph message passing. Graph notes add 1-WL expressivity limits and oversquashing — both were missing before and belong here.

**Failure-mode notes follow a consistent logic:** state the symmetry → identify what information cannot be represented → give WSI examples → list diagnostic questions. That matches the repo's stated goal.

**Paper placement matrix** is useful as an index for orienting readers across methods — as long as nobody treats every row as a verified derivation.

---

## Mistakes and Imprecisions

### Hierarchy parent notation (`hierarchy/01`, ~L63)

The note writes:

```math
|\{\pi_i^{(\ell)}(v)\}|=1
```

This treats π(v) as a set. It isn't. π maps each fine node v to a single parent u ∈ V^{(ℓ+1)}. The correct statement: each v has exactly one parent (tree/forest), or soft weights w_vu summing to 1 (DAG). Small error, but this is a math reference repo — notation should be exact.

### Deep Sets theorem wording (`set/02`, L13–15)

The note leads with "countable domains." WSI patch embeddings live in continuous ℝᵈ. The relevant statement for practitioners is universal approximation of continuous permutation-invariant functions on compact sets via ρ(Σ φ(x)). Countability is a special case, not the headline.

### Retrieval / neighbor notation (`distribution/03`, L124–128)

```math
\mathcal{N}_K(i) = \operatorname*{arg\,topK}_{k} - d(\mu_i,\mu_k)
```

This reads like malformed math. Should be `argmin_k d(μ_i, μ_k)` or `argmax_k sim(μ_i, μ_k)`. Notation matters when the doc is meant to be copy-pasteable for PhD students.

### Distribution vs set (`distribution/` folder)

μ_i = (1/n_i) Σ δ_{h_ij} is the same mathematical object as the set representation. The distribution folder mostly repackages set MIL with statistic T(μ). That is a valid *lens*, but the notes never clearly say: "This is not a new symmetry group — it is set invariance plus a richer readout functional." Without that sentence, readers may think distribution is a fundamentally different slide object.

### Hierarchy compute claim (`hierarchy/02`, L152–181)

O(Rm² + R²) beats O(n²) when R ≪ n and regions are balanced. The note doesn't warn that bad region splits (R ≈ n, m small) can make hierarchical attention *worse* than flat attention, or that coarsening-before-context discards fine evidence irreversibly.

### MMD and characteristic kernels (`distribution/02`, L67–75)

The characteristic-kernel claim (m_μ = m_ν ⇔ μ = ν) is correct in theory. Missing in practice: with n_i ~ 10⁴–10⁵ patches and d in the hundreds, empirical MMD is expensive, high-variance, and rarely used end-to-end in WSI pipelines. Worth one sentence so readers don't treat it as a default tool.

### Foundation slide map (`foundation/01`, L15–18)

Writing z_i = F_FM({x_ij, c_ij}) glosses over how slide foundation models actually work: variable-length token packing, LongNet-style sparse attention, slide-level pretraining objectives, coordinate-aware encoders. The "latent object" framing is right conceptually but too thin on architecture-specific forward maps for GigaPath, TITAN, etc.

---

## Thin or Redundant Sections

**`distribution_representations/`** — ~80% overlaps `set/` and prototype material already in `03_statistics_that_survive.md`. The genuinely new content is in `03_distances_and_transport.md` (KL, JS, OT, MMD). Consider merging the rest into set + one transport note, or adding a explicit "distribution = set + T(μ)" bridge page.

**`retrieval/` and `foundation/` failure modes** — largely the same stories: imported geometry ≠ downstream task; mean pooling kills rare morphology; cohort shift breaks latent neighborhoods; prompt mismatch for VL models. One cross-family failure note would cut repetition without losing content.

**`04_extended_family_placement.md`** — restates `01_crgs_placement` and the paper matrix in compressed form. Pick one level of detail and delete the duplicate.

**`03_paper_placement_matrix.md` (788 lines)** — depth is uneven. Deep Sets, ABMIL, CLAM, Patch-GCN, MambaMIL sections have real math. PRISM, TITAN, SISH, Yottixel entries read like catalog blurbs. Either keep the table only, or fully derive five anchor papers and link out for the rest.

---

## Still Missing (Mathematical Depth)

1. **Positive-instance MIL assumption** — never stated formally: e.g. ∃j with y_ij = 1 ⇒ y_i = 1 (standard instance-space MIL). Weak labels + set readout without this are underspecified.

2. **Equivalence map across families** — complete graph + one attention layer = set transformer; path graph = sequence scan; μ-only readout cannot distinguish layouts with identical patch multisets. One explicit "same slide, different representation, what changes" note would tie the families together.

3. **Estimation variance** — prototype prevalence p_im = (1/n_i) Σ q_m(h_ij) ignores that rare morphologies have high variance when n_i is huge and |support| is tiny. No discussion of M, n_i, or confidence in T(μ).

4. **Worked forward maps for hierarchy** — HIPT and HACT are named but not walked through as explicit tensor pipelines (levels, parent maps, readout tokens).

5. **Toy counterexamples** — zero concrete examples. E.g.: two slides with identical μ but different spatial clustering; which methods fail? One box per family would make the failure modes testable, not just plausible.

---

## Section Grades

| Section | Grade | Verdict |
|---------|-------|---------|
| `unifying_view/` | **8.5/10** | Keep and build from here. |
| Set / graph / sequence (updates) | **8/10** | Clear, mostly correct, good improvements. |
| `distribution/` | **6/10** | Transport note good; rest redundant with set. |
| `hierarchy/` | **6.5/10** | Right compositional ideas; notation slip; no worked HIPT/HACT map. |
| `retrieval/` | **6/10** | Conceptually fine; mathematically light. |
| `foundation/` | **6/10** | Failure modes good; architecture detail weak. |
| Paper matrix | **6.5/10** | Useful index; overconfident on long tail of methods. |

---

## Top 5 Fixes (Highest ROI)

1. **Fix hierarchy parent notation and Deep Sets continuous-domain wording** — quick credibility wins.

2. **Add one bridge page:** "Set vs distribution vs graph — when they are the same object and when they are not." Kills redundancy and clarifies the taxonomy.

3. **Shrink paper matrix** to table + five fully worked papers: CLAM, Patch-GCN, HIPT, PANTHER, MambaMIL. Link externally for the rest.

4. **Add one toy counterexample per family** — same μ different layout; wrong kNN graph; arbitrary scan order σ; collapsed prototypes; retrieval by stain not biology.

5. **Merge duplicate failure-mode bullets** across retrieval, foundation, and distribution into a single cross-cutting note.

---

## Bottom Line

The framework matured significantly on this branch. `unifying_view/` carries the repo. The new folders (distribution, hierarchy, retrieval, foundation) make the README checklist look complete, but they are not all equally deep — some are repackaging, some are catalog entries. Next pass should tighten and merge, not add more pages.

**Not on `main`.** This review file lives on `codex/slide-representations-review` when pushed.
