# Deform360 filament graph-support result

This milestone records the completed **source-only** structural test of a
deterministic component-bridge fallback for the exact eight filament reset
hulls that the registered extractor could not connect. The study used only the
36 already-opened reset hulls from `002-rope-silk` and `081-stripe-rope`. It did
not read mechanics scores, contact or tactile data, calibration outcomes, target
prefixes, or target futures.

## Evidence identity

- Workflow run: `30981252435`.
- Exact pull-request head: `249848c53378fb1f9a53e7ce774e39710b81a422`.
- Evaluated merge revision: `7aa5bf5d72a4965008bb8ebc8522bf2a7b01cacb`.
- Artifact ID: `8920243460`.
- Artifact name:
  `deform360-filament-support-7aa5bf5d72a4965008bb8ebc8522bf2a7b01cacb`.
- Artifact archive SHA-256:
  `7eb5a1b87e1d262705d17c37a9f51a106d2ce9e63763b1b06902f654675206a4`.
- Full result content SHA-256:
  `c1f84f2221639ef0a5cf982c1b63c933a1b4e269cbc4f5ceb08d0857057acddf`.
- `result.json` SHA-256:
  `8bfb4b70065ac9f6bbb71e51c5b0ab6c7e3cff0c60db230095694c463704a19c`.
- Runtime sidecar SHA-256:
  `fca638c4dd6abbfb6baa920c842f534a1171e0959e6f04cee8cb487eb83e562d`.
- Runtime-selection report SHA-256:
  `e3d1e48086ff6d16c549c74896715eef3ea25a61b87f063249b66380abfe15af`.
- Compact summary identity:
  `843d4ca3d1a8d1d6d7602cff20a57ead44368cab28a8c8598638803e20ba3f5b`.

The run used Deform360 dataset revision
`7fea8e20231a47641d1d2bc8791920ec4e62ec5e` with Python `3.12.3`, NumPy
`1.26.4`, and SciPy `1.13.1` under the already qualified conditional
reproduction-runtime boundary.

## Completed result

The registered failure boundary reproduced exactly:

```text
registered successes = 28
registered failures = 8
registered failure episodes = 7
```

The candidate completed all 36 reset constructions and preserved exact graph
content on all 28 resets already supported by the registered extractor. It did
not pass the preregistered admission boundary, however. Only two of the eight
repaired resets passed both structure and same-object geometry gates.

| Object / episode | Reset frame | Components | Max bridge / local scale | Length / object median | Structure | Geometry | Admit |
| --- | ---: | ---: | ---: | ---: | :---: | :---: | :---: |
| `002-rope-silk/episode_0000` | 176 | 2 | **12.348** | 1.138 | no | yes | no |
| `002-rope-silk/episode_0000` | 350 | 2 | 7.349 | **1.293** | yes | no | no |
| `002-rope-silk/episode_0005` | 380 | 2 | 9.633 | **1.562** | yes | no | no |
| `002-rope-silk/episode_0006` | 164 | 2 | 3.701 | 0.770 | yes | yes | **yes** |
| `002-rope-silk/episode_0007` | 290 | 2 | 5.333 | **1.271** | yes | no | no |
| `081-stripe-rope/episode_0001` | 104 | 3 | 9.473 | 0.865 | yes | yes | **yes** |
| `081-stripe-rope/episode_0007` | 224 | 2 | 3.527 | **0.668** | yes | no | no |
| `081-stripe-rope/episode_0009` | 86 | 2 | 6.481 | **0.673** | yes | no | no |

The frozen maximum bridge-to-local-scale ratio was `12.0`, and the frozen
same-object length interval was `[0.75, 1.25]`. Every repaired reset passed the
point-to-centerline p95, edge-uniformity, individual bridge-fraction, total
bridge-fraction, and component-count gates. One reset failed locality and five
failed the length envelope.

The authoritative decision is:

```text
classification = component_bridge_nonlocal_structure_failure
passed = false
primary_completed_reset_count = 36
exact_common_case_parity_count = 28
repaired_reset_count = 8
fully_admitted_repaired_reset_count = 2
mechanics_rescoring_permitted = false
```

## Interpretation

A deterministic minimum component bridge is technically complete, but it is not
an admissible general filament-support repair. The candidate sometimes joins
components across a bridge longer than the local graph permits, and more often
produces a centerline that is too long or too short relative to successful
same-object resets. Relaxing either threshold on these opened reset hulls would
convert a completed negative result into target-informed retuning and is not
permitted.

The next defensible source-side study would have to infer foreground filament
support **before** centerline extraction, for example from causal multi-frame
geometry or checksummed reconstruction provenance. Such a study needs a new
lock, must preserve exact common-case parity, and must not inherit changed
thresholds from this result. Mechanics rescoring is not authorized by this
milestone.

## Claim boundary

- This is a completed negative source-only structural diagnostic.
- The registered observed-reset result remains unchanged.
- The registered Deform360 replication decision remains unchanged.
- The frozen 36-execution estimator and physical evidence count remain
  unchanged at `0/36`.
- `target_prefix_access_permitted=false`.
- `target_future_access_permitted=false`.
- No mechanics, calibration, or target-informed retuning is authorized.

The compact machine-readable record is `summary.json`; the full 36-reset result,
runtime sidecar, run log, and checksum file remain in the checksummed GitHub
Actions artifact.
