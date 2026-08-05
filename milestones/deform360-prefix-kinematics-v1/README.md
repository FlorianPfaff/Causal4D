# Deform360 source prefix-kinematics diagnostic V1

This milestone records the completed **source-only** test of whether missing
initial object velocity is a dominant repair for the sparse Deform360 official-
Warp backend. It uses five objects and all 30 already-opened source episodes.
It does not read a calibration outcome, target prefix, target future geometry,
or target tactile stream.

## Evidence identity

- Workflow run: `30972643551`.
- Exact Causal4D head: `77caf44dbd749e37b34dbecf47cba03799d4289f`.
- Artifact ID: `8917112270`.
- Artifact archive SHA-256:
  `77955b80f4ef5ff1b9d796d3d816a9dde1342f471120d4d9724d1d82454ba9f4`.
- Full result content SHA-256:
  `3f1eaa75800cd7bb24d3be82da112ec5c6ab93d2873508cb147a1e2de3a323b3`.
- `result.json` SHA-256:
  `78dc6c00a187451a76e8e2a811d626d9332f05f5abd34cd83426ab297711d93b`.
- Runtime sidecar SHA-256:
  `d6e218b64640dd2efb3c9ab9e56411e0bff058bbdf8a02ad25c9b0ad30f409ea`.
- Runtime-selection report SHA-256:
  `e3d1e48086ff6d16c549c74896715eef3ea25a61b87f063249b66380abfe15af`.

The run pins BayesianPhysTwin
`695f7d7b949988052d8bd83ac7ac91620d1c6bb1`, Deform360 code
`0fe36f0b7a7a917ba62b5f8cee707299a9a4a317`, official PhysTwin
`2b6630528141b9cba5a7677c8b88b2129b4a8390`, and Deform360 dataset
revision `7fea8e20231a47641d1d2bc8791920ec4e62ec5e`.

## Runtime boundary

The frozen source milestone records NumPy `2.5.1`. The available named
`bpt-gpu` interpreter uses NumPy `1.26.4` while matching Python `3.12.3`, SciPy
`1.13.1`, Torch `2.4.0+cu121`, CUDA `12.1`, and Warp `1.15.0`. The candidate
runtime is therefore admitted only through the separately checksummed
conditional reproduction contract. It does not rewrite or relabel the original
milestone environment.

The scientific comparison is admissible only because every one of the 30
zero-velocity reruns reproduced its archived Chamfer and p99-strain values
within the locked `0.5 mm` and `0.01` tolerances.

## Completed result

| Policy | Mean Chamfer | Relative change versus zero | Win fraction | Valid episodes |
| --- | ---: | ---: | ---: | ---: |
| `zero_v1` | 58.040 mm | 0 | n/a | 28/30 |
| `global_contact_translation_v1` | 58.307 mm | **0.459% worse** | 40% | 28/30 |
| `graph_harmonic_contact_v1` | 58.142 mm | **0.175% worse** | 30% | 28/30 |

Both velocity policies retain the same 28 strain-valid episodes as the zero
baseline, with no quality rescues and no quality regressions. The graph-harmonic
policy improves mean Chamfer only on scarf (`0.014%`) and spider (`0.451%`), and
neither object reaches the locked 60% episode-win requirement. Rope, stripe
rope, and blanket all worsen.

The primary source gate therefore fails:

```text
baseline_reproduction_passed=true
common_finite_episode_count=30
relative_improvement_vs_zero=-0.00175455
win_fraction_vs_zero=0.30
passed=false
```

## Interpretation

Missing initial object velocity is **not** a dominant, transferable repair for
the current sparse source backend. A more elaborate velocity estimator is not
justified on this opened cohort: the present graph-harmonic field already uses
causal contact-patch velocities, preserves the fixed physical candidates and
contact schedules, and produces neither aggregate nor object-consistent gain.

This result reinforces the earlier source-failure attribution. The next useful
work is representation, support/contact realization, and within-episode
dynamics competence, followed by genuinely independent public-object or
physical-execution evidence. A larger shared parameter grid or wider velocity
model should not be used to reopen the same source cohort.

## Claim boundary

- This is a completed negative source-only diagnostic.
- The registered Deform360 replication decision remains unchanged.
- The frozen 36-execution physical estimator and evidence count remain
  unchanged.
- `target_prefix_access_permitted=false`.
- `target_future_access_permitted=false`.
- No target-informed retuning or target access is authorized.

The compact machine-readable record is `summary.json`; the full row-level result
and runtime files remain in the checksummed GitHub Actions artifact.
