# Explanation C/R/G/S Ledger

Use the repository-wide decomposition

```math
\widetilde H_i=\mathcal C(H_i;G_i,S_i),
\qquad
z_i=\mathcal R(\widetilde H_i),
\qquad
F_i=\mathcal H(z_i).
```

| Explanation family | Context target | Readout target | Geometry target | Supervision target |
|---|---|---|---|---|
| attention | normalized routing | weighted statistic | optional coordinates | task loss |
| gradient | local Jacobian | score functional | representation metric | chosen output |
| perturbation | recomputed computation | deletion/insertion effect | feasibility manifold | intervention target |
| prototype | similarity/assignment | presence or block credit | learned metric | exemplar or mixture objective |
| concept | directional or semantic relation | concept coordinate | activation or VLM space | concept examples or prompts |
| graph | message/path support | node/type pooling | topology and coordinates | graph label or survival target |
| survival | time/cause context | risk, hazard, curve, incidence | WSI geometry | censoring-aware loss |

The same word “importance” can occupy different columns. A paper placement is
incomplete until the column is named.

