# Brutal Review — `slide_representations/` Notebooks

**Branch reviewed:** `codex/slide-representations-review`  
**Date:** July 7, 2026

**Overall: 7/10.** Ambitious scope, `unifying_view/` is genuinely good. New families add breadth but repeat set material, and a few math slips remain.

---

## What's Actually Good

- **`unifying_view/`** — best part. Adjacency-as-context and C/R/G/S placement fix the fragmentation problem.
- **Set/graph updates** — Deep Sets caveat, max-as-limit, gated attention, set-attention = complete graph, 1-WL + oversquashing: solid.
- **Paper matrix** — useful if treated as a draft index, not ground truth.
- **Failure-mode structure** — consistent and task-aware.

---

## Mistakes / Imprecisions

| Issue | Where | Problem |
|-------|--------|---------|
| **Tree parent notation** | `hierarchy/01` L63 | `\|\{\pi(v)\}\|=1` is wrong — π(v) is a single parent, not a set. Should say each v has exactly one parent in V^{(ℓ+1)}. |
| **Deep Sets theorem** | `set/02` L13–15 | Says "countable domains" first; WSI embeddings live in **continuous** ℝᵈ. Lead with compactness + universal approximation, not countability. |
| **Retrieval notation** | `distribution/03` L124–128 | `arg topK_k - d(μ_i,μ_k)` reads like broken math. Use `argmin_k d(...)` or `argmax_k sim(...)`. |
| **Distribution = set in disguise** | whole `distribution/` folder | μ_i is the same object as set MIL. You renamed set as measure without saying when it's a *different* hypothesis class vs just richer T(μ). |
| **Hierarchy cost claim** | `hierarchy/02` L152–181 | O(Rm²+R²) < O(n²) only when R ≪ n. No warning that bad region splits can be **worse** than flat attention. |
| **MMD / characteristic kernel** | `distribution/02` L67–75 | Correct in theory, but no note that **finite n_i** + high-d embeddings make MMD noisy and expensive at WSI scale. |
| **Foundation slide map** | `foundation/01` L15–18 | `F_FM({x_ij, c_ij})` glosses over how slide FMs actually work (token packing, LongNet, etc.) — too hand-wavy for a "latent object" claim. |

---

## Thin / Redundant Sections (Cut or Merge)

1. **`distribution_representations/`** — 80% overlaps `set/` + prototype material. Only `03_distances_and_transport.md` adds real new math. Rest is repackaging.
2. **`retrieval/` vs `foundation/` failure modes** — same story repeated: "imported geometry ≠ task," "mean pooling kills rare patterns," "cohort shift." Merge into one cross-cutting failure note.
3. **`04_extended_family_placement.md`** — mostly duplicates `01_crgs` + paper matrix in shorter form. Pick one.
4. **`03_paper_placement_matrix.md` (788 lines)** — too long for the depth per method. PRISM/TITAN/SISH/Yottixel entries read like catalog blurbs, not derivations. Either shorten to table-only or fully derive 5 anchor papers.

---

## Missing Math (Still)

- **Positive-instance MIL assumption** — never stated formally (`∃j: y_ij=1 ⇒ y_i=1`).
- **When set ≡ graph** — complete graph + 1 layer mentioned, but no statement that **distribution with only T(μ) has weaker symmetry than graph** when layout matters.
- **Estimation variance** — prototype/histogram/MMD all ignore n_i (often 10⁴–10⁵ patches) vs M (prototypes). No sample-complexity discussion.
- **Hierarchy readout** — `03_hierarchical_readout.md` not wrong, but no HIPT/HACT forward map with explicit tensor shapes.
- **No sanity checks** — zero toy examples (e.g. two slides, same μ, different layout → which methods fail).

---

## Section Grades

| Section | Grade | One-liner |
|---------|-------|-----------|
| `unifying_view/` | **8.5/10** | Keep this; it's the spine. |
| Set / graph / sequence (updates) | **8/10** | Clear, mostly correct. |
| `distribution/` | **6/10** | Transport note good; rest redundant. |
| `hierarchy/` | **6.5/10** | Right ideas; notation slip + no worked HIPT/HACT map. |
| `retrieval/` | **6/10** | Conceptually fine, mathematically light. |
| `foundation/` | **6/10** | Good failure modes; weak on actual FM architectures. |
| Paper matrix | **6.5/10** | Useful index, overconfident detail. |

---

## Top 5 Fixes (Highest ROI)

1. Fix hierarchy parent notation + Deep Sets continuous-domain wording.
2. Add one page: **"Set vs distribution vs graph — when they're the same object"** (kill redundancy).
3. Shrink paper matrix to table + **5 fully worked papers** (CLAM, Patch-GCN, HIPT, PANTHER, MambaMIL).
4. Add **one toy counterexample** per family (same μ, different layout; wrong kNN; bad σ).
5. Merge repeated failure-mode bullets across retrieval/foundation/distribution.

---

## Bottom Line

Framework matured a lot. New folders look complete on a README checklist but aren't all equally deep — `unifying_view` carries the repo; distribution/retrieval/foundation need tightening or merging, not more pages.
