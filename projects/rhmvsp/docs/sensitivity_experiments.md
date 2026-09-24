# Sensitivity Experiments — Independent Variable Values

The main experiment uses `alpha_R = 0.90`, `alpha_T = 0.80`, and 50 vehicles at each provincial fleet centre.

| Experiment               | Independent-variable values                                          | Fixed reliability settings                    |
| ------------------------ | -------------------------------------------------------------------- | --------------------------------------------- |
| Risk reliability         | `alpha_R = [0.88, 0.90, 0.92, 0.94, 0.96, 0.98]`                     | `alpha_T = 0.80`                              |
| Time reliability         | `alpha_T = [0.50, 0.60, 0.70, 0.80, 0.90]`                           | `alpha_R = 0.90`                              |
| Fleet size               | `[10, 20, 30, 40, 50, 60, 70, 80]` vehicles per centre               | `alpha_R = 0.90`, `alpha_T = 0.80`            |
| Coupled reliability      | Cartesian product of the risk and time grids above (30 combinations) | —                                             |
| Value of reliability     | `(alpha_T, alpha_R) = [(0.60, 0.905), (0.85, 0.920), (0.95, 0.935)]` | Low, medium, and high reliability designs     |
| Algorithm performance    | `alpha_T = [0.70, 0.90]` × `alpha_R = [0.90, 0.94, 0.98]`            | 6 combinations per instance                   |
| Out-of-sample evaluation | `alpha_T = [0.75, 0.95]` × `alpha_R = [0.905, 0.920, 0.935]`         | 6 combinations, 200 scenarios per combination |

The reliability grids exclude the degenerate endpoints zero and one. The risk grid is concentrated around the expected plan-risk-reliability range produced by the synthetic component beliefs, while the time grid spans moderate to high uncertain-measure requirements.
