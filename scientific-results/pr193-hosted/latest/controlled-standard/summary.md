# Causal4D self-hosted evaluation

- Profile: `standard`
- Runner: `GitHub Actions 1000186506`
- Commit: `a2024c6d49141fb6d521089f031a50dccac0251f`

## Controlled counterfactual benchmark

| Scenario | World | Method | RMSE mm (95% bootstrap CI) | Coverage | NEES | Gross failure |
|---|---|---:|---:|---:|---:|---:|
| nominal | matched_contact | generative_only | 16.351 [16.346, 16.356] | 0.979 | 0.453 | 0.333 |
| nominal | matched_contact | hybrid | 2.154 [2.148, 2.160] | 0.945 | 0.874 | 0.000 |
| nominal | matched_contact | physics_only | 2.165 [2.160, 2.170] | 0.937 | 0.967 | 0.000 |
| nominal | shifted_contact | generative_only | 12.862 [12.857, 12.868] | 0.998 | 0.274 | 0.333 |
| nominal | shifted_contact | hybrid | 4.398 [4.388, 4.408] | 0.844 | 3.817 | 0.000 |
| nominal | shifted_contact | physics_only | 4.367 [4.361, 4.373] | 0.809 | 4.091 | 0.000 |
| matched_world | matched_contact | generative_only | 16.369 [16.362, 16.376] | 0.968 | 0.515 | 0.333 |
| matched_world | matched_contact | hybrid | 1.430 [1.418, 1.441] | 0.971 | 0.308 | 0.000 |
| matched_world | matched_contact | physics_only | 1.427 [1.417, 1.438] | 0.960 | 0.363 | 0.000 |
| matched_world | shifted_contact | generative_only | 12.873 [12.866, 12.880] | 0.996 | 0.290 | 0.333 |
| matched_world | shifted_contact | hybrid | 4.294 [4.285, 4.303] | 0.856 | 3.266 | 0.000 |
| matched_world | shifted_contact | physics_only | 4.295 [4.286, 4.304] | 0.842 | 3.895 | 0.000 |
| strong_world_mismatch | matched_contact | generative_only | 16.324 [16.317, 16.331] | 0.975 | 0.437 | 0.333 |
| strong_world_mismatch | matched_contact | hybrid | 3.517 [3.509, 3.526] | 0.865 | 2.274 | 0.000 |
| strong_world_mismatch | matched_contact | physics_only | 3.500 [3.496, 3.504] | 0.796 | 2.703 | 0.000 |
| strong_world_mismatch | shifted_contact | generative_only | 12.845 [12.838, 12.852] | 0.997 | 0.274 | 0.333 |
| strong_world_mismatch | shifted_contact | hybrid | 4.825 [4.808, 4.843] | 0.840 | 4.169 | 0.000 |
| strong_world_mismatch | shifted_contact | physics_only | 4.630 [4.623, 4.637] | 0.767 | 4.743 | 0.000 |
| high_inference_noise | matched_contact | generative_only | 16.346 [16.339, 16.353] | 0.979 | 0.452 | 0.333 |
| high_inference_noise | matched_contact | hybrid | 2.046 [2.041, 2.053] | 0.965 | 0.477 | 0.000 |
| high_inference_noise | matched_contact | physics_only | 2.062 [2.058, 2.068] | 0.965 | 0.556 | 0.000 |
| high_inference_noise | shifted_contact | generative_only | 12.859 [12.852, 12.865] | 0.998 | 0.274 | 0.333 |
| high_inference_noise | shifted_contact | hybrid | 4.478 [4.467, 4.488] | 0.887 | 2.245 | 0.000 |
| high_inference_noise | shifted_contact | physics_only | 4.437 [4.430, 4.443] | 0.859 | 2.478 | 0.000 |

## Paired hybrid comparisons

| Scenario | World | Comparison | Hybrid win fraction | Mean relative RMSE improvement (95% CI) |
|---|---|---|---:|---:|
| nominal | matched_contact | hybrid_vs_physics_only | 0.227 | 0.706% [0.566%, 0.842%] |
| nominal | matched_contact | hybrid_vs_generative_only | 1.000 | 86.807% [86.759%, 86.854%] |
| nominal | shifted_contact | hybrid_vs_physics_only | 0.000 | -1.027% [-1.231%, -0.820%] |
| nominal | shifted_contact | hybrid_vs_generative_only | 1.000 | 66.394% [66.307%, 66.485%] |
| matched_world | matched_contact | hybrid_vs_physics_only | 0.000 | -0.102% [-0.189%, -0.029%] |
| matched_world | matched_contact | hybrid_vs_generative_only | 1.000 | 91.375% [91.269%, 91.476%] |
| matched_world | shifted_contact | hybrid_vs_physics_only | 0.083 | 0.013% [0.003%, 0.026%] |
| matched_world | shifted_contact | hybrid_vs_generative_only | 1.000 | 67.421% [67.334%, 67.507%] |
| strong_world_mismatch | matched_contact | hybrid_vs_physics_only | 0.017 | -0.652% [-0.934%, -0.431%] |
| strong_world_mismatch | matched_contact | hybrid_vs_generative_only | 1.000 | 78.117% [78.032%, 78.192%] |
| strong_world_mismatch | shifted_contact | hybrid_vs_physics_only | 0.000 | -5.925% [-6.517%, -5.386%] |
| strong_world_mismatch | shifted_contact | hybrid_vs_generative_only | 1.000 | 62.420% [62.220%, 62.597%] |
| high_inference_noise | matched_contact | hybrid_vs_physics_only | 0.283 | 1.023% [0.809%, 1.203%] |
| high_inference_noise | matched_contact | hybrid_vs_generative_only | 1.000 | 87.464% [87.410%, 87.512%] |
| high_inference_noise | shifted_contact | hybrid_vs_physics_only | 0.000 | -1.358% [-1.658%, -1.053%] |
| high_inference_noise | shifted_contact | hybrid_vs_generative_only | 1.000 | 65.587% [65.471%, 65.710%] |

## Other diagnostics

- Latent-contact overall gate: `False`.
- Dynamic-contact cases: `400`; all gates passed: `True`; prefix-only: `True`.
- Exact repeated-run determinism: `True`.

These runs are diagnostic and do not replace the registered same-object physical experiment.
