# Sensitivity Experiments — Independent Variable Values

## 1. `sensitivity_alpha_max.py`

| Independent Variable | Fixed Parameters | Values                                                                         |
| -------------------- | ---------------- | ------------------------------------------------------------------------------ |
| **α_max**            | α_T=0.8, δ_T=0.8 | `[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.10]` (10 values) |

## 2. `sensitivity_alpha_T.py`

| Independent Variable | Fixed Parameters    | Values                                                           |
| -------------------- | ------------------- | ---------------------------------------------------------------- |
| **α_T**              | α_max=0.05, δ_T=0.8 | `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]` (10 values) |

## 3. `sensitivity_risk_time_coupled.py`

| Independent Variable | Fixed Parameters | Values                                                                         |
| -------------------- | ---------------- | ------------------------------------------------------------------------------ |
| **α_max**            | α_T=0.80         | `[0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.10]` (10 values) |
| **δ_T**              | (same)           | `[0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]` (10 values)             |

Grid: 10 × 10 = 100 combinations.

## 4. `sensitivity_value_of_fuzzy_time.py`

| Independent Variable | Fixed Parameters    | Values                                                             |
| -------------------- | ------------------- | ------------------------------------------------------------------ |
| **δ_T**              | α_max=0.05, α_T=0.8 | `[0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]` (10 values) |

## 5. `sensitivity_out_of_sample.py`

| Independent Variable | Fixed Parameters | Values                                             |
| -------------------- | ---------------- | -------------------------------------------------- |
| **δ_T**              | α_T=0.8          | `[0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8]`         |
| **α_max**            | (same)           | `[0.02, 0.05, 0.02, 0.05, 0.02, 0.05, 0.02, 0.05]` |

8 groups × 200 Monte Carlo scenarios × 2 models (BAP + LP).

## 6. `sensitivity_algorithm_performance.py`

| Independent Variable | Fixed Parameters             | Values                               |
| -------------------- | ---------------------------- | ------------------------------------ |
| **Instance Scale**   | α_max=0.05, α_T=0.8, δ_T=0.8 | `small` (medium/large commented out) |
| Instances per scale  | —                            | 2 (instance_1, instance_2)           |
