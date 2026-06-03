# Comparison Between Wang (2025) and the Current RHMVSP Model

| Aspect                | Wang et al. (2025)                                                                                                  | Current RHMVSP Model (Liu 2007 Uncertainty Theory + Reliability Measures)                                                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Problem Goal          | Reliable ship-fleet planning: decide ship numbers, routes, hub activation under uncertainty to minimize total cost. | Multi-modal hazmat vehicle scheduling: decide routes, hub activation, timing under demand/cost uncertainties to minimize total cost (deployment, transport, transfer, holding, shortage penalty).                  |
| Uncertainty Types     | 1. Demand uncertainty 2. Transport-cost uncertainty                                                                 | 1. Time-varying risk: phi(t) on accident probability, psi(t) on consequence. 2. Fuzzy travel time: trapezoidal fuzzy variables for transport/transfer. 3. Travel cost uncertainty: budget constraint with alpha_c. |
| Modeling Approach     | Two-stage robust optimization with budget uncertainty sets.                                                         | Two-stage fuzzy-reliability optimization with time-dependent reliability constraints and fuzzy expected cost.                                                                                                      |
| Algorithmic Framework | Column Generation + Branch-and-Price (BAP).                                                                         | Same BAP framework with fuzzy-expected cost and reliability constraints.                                                                                                                                           |
| Solution Technique    | Gurobi; robust constraints via dualization of budget set.                                                           | Gurobi/PuLP; closed-form fuzzy expectations and deterministic equivalents.                                                                                                                                         |
| Reference Sources     | Wang et al. (2025), Transportation Research Part E. Bertsimas & Sim (2004).                                         | Liu (2007). Fabiano et al. (2002), Abkowitz & Cheng (1988) for time-varying risk.                                                                                                                                  |

---

## Take-away

- **Wang model** emphasizes robustness via worst-case budgeting.
- **Our model** blends fuzzy expectation with reliability guarantees via time-varying risk multipliers and fuzzy travel time, offering a more practical solution while respecting service-level targets.
