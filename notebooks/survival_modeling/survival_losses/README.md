# Survival Losses

A survival loss defines which observed facts about $(X_i,\delta_i)$ are used
and which risk object is trained.

The central split:

```text
partial likelihood:
    event ordering through risk sets

hazard likelihood:
    event/censoring likelihood under hazard model

discrete likelihood:
    masked binary sequence or PMF likelihood

ranking losses:
    pairwise ordering surrogate

IPCW losses:
    censoring-weighted supervised risk at horizons
```

## Files

- `01_loss_family_map.md`: common survival objectives.
- `02_likelihood_losses.md`: continuous, discrete, Cox, and PMF likelihoods.
- `03_ranking_losses.md`: pairwise/event-order losses and gradients.
- `04_ipcw_losses.md`: censoring-weighted risk estimation.
- `05_multi_objective_and_regularization.md`: how losses are combined in WSI.
