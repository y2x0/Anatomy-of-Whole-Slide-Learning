# Objective Foundations

This folder derives the statistical and geometric core of contrastive and
negative-free joint-embedding objectives.

## Files

- `01_candidate_identification_and_density_ratio.md`: Bayes candidate posterior
  and the exact density-ratio critic.
- `02_infonce_mutual_information_bound.md`: rigorous finite-candidate MI lower
  bound, assumptions, saturation, and proposal mismatch.
- `03_temperature_and_hyperspherical_geometry.md`: cosine-distance equivalence,
  temperature limits, angular margins, and projector geometry.
- `04_exact_gradients_hessian_and_normalization.md`: query/key gradients,
  covariance Hessian, and the normalization Jacobian.
- `05_multi_positive_supervised_contrastive.md`: SupCon outside-log and
  inside-log objectives, Jensen relation, and positive weighting.
- `06_false_negatives_and_batch_bias.md`: collision probabilities, proposal
  shift, batch conditioning, and augmentation violations.
- `07_moco_dictionary_queue_and_staleness.md`: queue-based dictionaries,
  momentum keys, age distribution, and consistency tradeoffs.
- `08_byol_simsiam_stop_gradient_dynamics.md`: asymmetric prediction,
  stop-gradient, EMA targets, and collapse caveats.
- `09_barlow_twins_vicreg_moment_constraints.md`: cross-correlation,
  invariance, variance, covariance, and explicit collapse control.
- `10_crgs_surviving_geometry_and_failure_matrix.md`: objective comparison,
  C/R/G/S placement, surviving geometry, and design failures.

## Source Boundary

Only these primary papers from the private research map are used in this
portion:

- CPC: https://arxiv.org/abs/1807.03748
- SimCLR: https://arxiv.org/abs/2002.05709
- MoCo: https://arxiv.org/abs/1911.05722
- Supervised Contrastive Learning: https://arxiv.org/abs/2004.11362
- BYOL: https://arxiv.org/abs/2006.07733
- SimSiam: https://arxiv.org/abs/2011.10566
- Barlow Twins: https://arxiv.org/abs/2103.03230
- VICReg: https://arxiv.org/abs/2105.04906
