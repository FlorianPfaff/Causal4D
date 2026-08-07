# Controlled reproducibility evidence — 2026-08-07

This directory records workflow-produced **controlled computational evidence**
for Causal4D. It does not contain confirmatory physical evidence and does not
change the frozen same-object physical protocol.

## Exact execution identity

- Pull request: `#223`
- Workflow run: `31156729576` (`Controlled reproducibility evidence`, success)
- Base revision: `8f23bc05e6f142f8d0ddccbf06ba641fe90875b9`
- Product branch head: `5a722b86902c2e6f890745aab682d4be9e12f787`
- Tested merge revision: `5621c4d32667d738180aadc9d96baaf99cbded2a`
- Canonical runtime: Python `3.12.13`

The ordinary merge gate, CI/release matrix, extended compatibility, and
security scanning also passed for this exact pull-request head.

## Evidence boundary

- `physical_evidence_increment = 0`
- `target_outcomes_used = false`
- Confirmatory physical status remains `0/36` acquired and `0/36` validated.
- These results refresh controlled reproducibility and stress evidence only.

## 64-seed controlled counterfactual replication

| Method | Contact condition | Cases | RMSE (mm) | Gross-failure rate | Coverage | NEES |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `generative_only` | `matched_contact` | 192 | 16.3492 | 0.3333 | 0.9792 | 0.4528 |
| `generative_only` | `shifted_contact` | 192 | 12.8604 | 0.3333 | 0.9975 | 0.2738 |
| `hybrid` | `matched_contact` | 192 | 2.1525 | 0.0000 | 0.9457 | 0.8691 |
| `hybrid` | `shifted_contact` | 192 | 4.3997 | 0.0000 | 0.8452 | 3.8040 |
| `physics_only` | `matched_contact` | 192 | 2.1646 | 0.0000 | 0.9370 | 0.9648 |
| `physics_only` | `shifted_contact` | 192 | 4.3679 | 0.0000 | 0.8095 | 4.0868 |

Relative to the generative-only baseline, the hybrid RMSE is lower by
**86.83%** for matched contact and **65.79%** for shifted contact. The
physics-only and hybrid errors are very close: hybrid is 0.0122 mm lower for
matched contact and 0.0319 mm higher for shifted contact. The controlled
evidence therefore supports the physics-based advantage over the generative
baseline, but not a blanket claim that the residual hybrid always improves the
physics-only predictor.

## 10-seed held-out-topology latent-contact replication

The workflow enforced `--require-gates`; all **13/13 registered gates passed**.

| Setting | Method | Contact condition | Cases | RMSE (mm) | Coverage | NEES |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `online_adaptation` | `latent_contact` | `matched_contact` | 30 | 1.9704 | 0.8917 | 2.3387 |
| `online_adaptation` | `latent_contact` | `shifted_contact` | 30 | 0.8161 | 0.9025 | 6.8564 |
| `online_adaptation` | `nominal_physics` | `matched_contact` | 30 | 2.4292 | 0.8889 | 1.3293 |
| `online_adaptation` | `nominal_physics` | `shifted_contact` | 30 | 4.1440 | 0.7800 | 4.7157 |
| `online_adaptation` | `oracle_contact` | `matched_contact` | 30 | 1.6526 | 0.9319 | 0.6105 |
| `online_adaptation` | `oracle_contact` | `shifted_contact` | 30 | 1.1650 | 0.9752 | 0.3579 |
| `online_adaptation` | `oracle_contact_theta` | `matched_contact` | 30 | 0.0450 | 1.0000 | 0.0012 |
| `online_adaptation` | `oracle_contact_theta` | `shifted_contact` | 30 | 0.0050 | 1.0000 | 0.0000 |

Under online prefix adaptation, latent-contact inference reduces nominal-physics
RMSE by **18.89%** on matched contact (2.4292 → 1.9704 mm) and **80.31%**
on shifted contact (4.1440 → 0.8161 mm).

### Registered gate outcomes

| Gate | Rule | Value | Threshold | Passed |
| --- | ---: | ---: | ---: | :---: |
| `shifted_oracle_gap_closure` | `>=` | 0.804046 | 0.500000 | yes |
| `matched_contact_relative_degradation` | `<=` | -0.188858 | 0.100000 | yes |
| `maximum_online_coverage_error` | `<=` | 0.008348 | 0.050000 | yes |
| `shifted_node_accuracy` | `>=` | 0.833333 | 0.800000 | yes |
| `shifted_node_credible_coverage` | `>=` | 0.900000 | 0.800000 | yes |
| `shifted_node_calibration_error` | `<=` | 0.116204 | 0.150000 | yes |
| `shifted_gain_mae` | `<=` | 0.087387 | 0.150000 | yes |
| `shifted_gain_coverage` | `>=` | 0.966667 | 0.800000 | yes |
| `shifted_delay_mae_steps` | `<=` | 0.337721 | 0.500000 | yes |
| `shifted_delay_map_accuracy` | `>=` | 0.933333 | 0.800000 | yes |
| `shifted_delay_coverage` | `>=` | 0.933333 | 0.800000 | yes |
| `held_out_topology_count` | `>=` | 3 | 3 | yes |
| `minimum_topology_oracle_gap_closure` | `>=` | 0.669181 | 0 | yes |

The three held-out topologies close 66.92%, 87.54%, and 70.95% of the
nominal-to-oracle shifted-contact gap. The aggregate shifted gap closure is
80.40%.

## Counterfactual stress replications

Each case uses 20 seeds. Positive values in the final column mean hybrid is
better than physics-only; negative values mean hybrid is worse.

| Stress case | Contact | Generative RMSE (mm) | Physics RMSE (mm) | Hybrid RMSE (mm) | Physics − hybrid (mm) |
| --- | --- | ---: | ---: | ---: | ---: |
| `observation-noise-3mm` | `matched_contact` | 16.4570 | 2.1651 | 2.1650 | 0.0001 |
| `observation-noise-3mm` | `shifted_contact` | 12.9910 | 4.3692 | 4.3743 | -0.0051 |
| `observation-noise-6mm` | `matched_contact` | 16.8410 | 2.1693 | 2.1724 | -0.0031 |
| `observation-noise-6mm` | `shifted_contact` | 13.4622 | 4.3783 | 4.3830 | -0.0047 |
| `rotation-15deg` | `matched_contact` | 16.3559 | 3.3241 | 3.3316 | -0.0074 |
| `rotation-15deg` | `shifted_contact` | 12.8665 | 4.6029 | 4.7589 | -0.1560 |
| `nonlinearity-0p30` | `matched_contact` | 16.3415 | 2.1652 | 2.1525 | 0.0127 |
| `nonlinearity-0p30` | `shifted_contact` | 12.8578 | 4.3545 | 4.3875 | -0.0330 |

Physics-only and hybrid retain zero gross failures in every stress case, while
the generative baseline retains a one-third gross-failure rate. The 15°
rotation case is the clearest limitation: under shifted contact, hybrid is
0.1560 mm worse than physics-only. This result is retained rather than hidden.

## Numerical portability

The 10-seed Python 3.10.20 and Python 3.14.6 runs produced byte-identical:

- `interventions.csv`
- `fit_diagnostics.csv`
- `protocol.json`

The only aggregate difference is damping CRPS at
`2.7755575615628914e-17`; the maximum row-level CRPS delta is
`2.7755575615628914e-16`. All other numerical fields agree exactly.

## Retained files

- `evidence.json`: compact execution, result, gate, stress, portability, and
  artifact record.

Full raw CSVs, environments, manifests, logs, and SHA-256 inventories are in
the GitHub Actions artifacts listed in `evidence.json`. The committed workflow
can regenerate them from the fixed seed inventories.

## Interpretation

This evidence strengthens reproducibility of the controlled result and exposes
its stress limitations. It does not advance the physical experiment count,
establish independent-execution calibration on physical data, or admit a fresh
real Prob4D provider.
