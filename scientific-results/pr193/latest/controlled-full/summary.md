# Causal4D self-hosted evaluation

- Profile: `full`
- Runner: `workstation2`
- Commit: `2d3382dda28c610e0ca290aac83290bc7ac4df08`

## Controlled counterfactual benchmark

| Scenario | World | Method | RMSE mm (95% bootstrap CI) | Coverage | NEES | Gross failure |
|---|---|---:|---:|---:|---:|---:|
| nominal | matched_contact | generative_only | 16.349 [16.346, 16.352] | 0.980 | 0.453 | 0.333 |
| nominal | matched_contact | hybrid | 2.152 [2.149, 2.155] | 0.946 | 0.866 | 0.000 |
| nominal | matched_contact | physics_only | 2.165 [2.162, 2.167] | 0.937 | 0.965 | 0.000 |
| nominal | shifted_contact | generative_only | 12.860 [12.858, 12.863] | 0.997 | 0.274 | 0.333 |
| nominal | shifted_contact | hybrid | 4.401 [4.396, 4.405] | 0.847 | 3.789 | 0.000 |
| nominal | shifted_contact | physics_only | 4.367 [4.364, 4.370] | 0.810 | 4.084 | 0.000 |
| matched_world | matched_contact | generative_only | 16.374 [16.370, 16.378] | 0.968 | 0.517 | 0.333 |
| matched_world | matched_contact | hybrid | 1.428 [1.421, 1.435] | 0.968 | 0.315 | 0.000 |
| matched_world | matched_contact | physics_only | 1.424 [1.418, 1.431] | 0.959 | 0.368 | 0.000 |
| matched_world | shifted_contact | generative_only | 12.877 [12.874, 12.881] | 0.997 | 0.291 | 0.333 |
| matched_world | shifted_contact | hybrid | 4.300 [4.295, 4.305] | 0.856 | 3.316 | 0.000 |
| matched_world | shifted_contact | physics_only | 4.301 [4.296, 4.306] | 0.842 | 3.922 | 0.000 |
| strong_world_mismatch | matched_contact | generative_only | 16.330 [16.327, 16.333] | 0.976 | 0.437 | 0.333 |
| strong_world_mismatch | matched_contact | hybrid | 3.520 [3.515, 3.524] | 0.866 | 2.264 | 0.000 |
| strong_world_mismatch | matched_contact | physics_only | 3.497 [3.495, 3.499] | 0.797 | 2.696 | 0.000 |
| strong_world_mismatch | shifted_contact | generative_only | 12.850 [12.847, 12.853] | 0.996 | 0.273 | 0.333 |
| strong_world_mismatch | shifted_contact | hybrid | 4.835 [4.825, 4.845] | 0.839 | 4.168 | 0.000 |
| strong_world_mismatch | shifted_contact | physics_only | 4.636 [4.632, 4.640] | 0.766 | 4.741 | 0.000 |
| high_inference_noise | matched_contact | generative_only | 16.352 [16.348, 16.355] | 0.979 | 0.452 | 0.333 |
| high_inference_noise | matched_contact | hybrid | 2.049 [2.046, 2.052] | 0.964 | 0.483 | 0.000 |
| high_inference_noise | matched_contact | physics_only | 2.063 [2.060, 2.066] | 0.964 | 0.556 | 0.000 |
| high_inference_noise | shifted_contact | generative_only | 12.863 [12.860, 12.867] | 0.998 | 0.274 | 0.333 |
| high_inference_noise | shifted_contact | hybrid | 4.475 [4.470, 4.481] | 0.884 | 2.265 | 0.000 |
| high_inference_noise | shifted_contact | physics_only | 4.438 [4.435, 4.442] | 0.860 | 2.477 | 0.000 |

## Paired hybrid comparisons

| Scenario | World | Comparison | Hybrid win fraction | Mean relative RMSE improvement (95% CI) |
|---|---|---|---:|---:|
| nominal | matched_contact | hybrid_vs_physics_only | 0.243 | 0.782% [0.712%, 0.849%] |
| nominal | matched_contact | hybrid_vs_generative_only | 1.000 | 86.816% [86.791%, 86.841%] |
| nominal | shifted_contact | hybrid_vs_physics_only | 0.000 | -1.116% [-1.214%, -1.016%] |
| nominal | shifted_contact | hybrid_vs_generative_only | 1.000 | 66.354% [66.307%, 66.402%] |
| matched_world | matched_contact | hybrid_vs_physics_only | 0.000 | -0.155% [-0.216%, -0.100%] |
| matched_world | matched_contact | hybrid_vs_generative_only | 1.000 | 91.430% [91.376%, 91.484%] |
| matched_world | shifted_contact | hybrid_vs_physics_only | 0.083 | 0.017% [0.010%, 0.025%] |
| matched_world | shifted_contact | hybrid_vs_generative_only | 1.000 | 67.367% [67.319%, 67.415%] |
| strong_world_mismatch | matched_contact | hybrid_vs_physics_only | 0.023 | -0.831% [-0.990%, -0.685%] |
| strong_world_mismatch | matched_contact | hybrid_vs_generative_only | 1.000 | 78.112% [78.069%, 78.152%] |
| strong_world_mismatch | shifted_contact | hybrid_vs_physics_only | 0.000 | -6.085% [-6.392%, -5.788%] |
| strong_world_mismatch | shifted_contact | hybrid_vs_generative_only | 1.000 | 62.325% [62.218%, 62.428%] |
| high_inference_noise | matched_contact | hybrid_vs_physics_only | 0.260 | 0.862% [0.762%, 0.957%] |
| high_inference_noise | matched_contact | hybrid_vs_generative_only | 1.000 | 87.461% [87.437%, 87.485%] |
| high_inference_noise | shifted_contact | hybrid_vs_physics_only | 0.000 | -1.210% [-1.353%, -1.067%] |
| high_inference_noise | shifted_contact | hybrid_vs_generative_only | 1.000 | 65.622% [65.569%, 65.673%] |

## Other diagnostics

- Latent-contact overall gate: `False`.
- Dynamic-contact cases: `2000`; all gates passed: `True`; prefix-only: `True`.
- Exact repeated-run determinism: `True`.

These runs are diagnostic and do not replace the registered same-object physical experiment.
