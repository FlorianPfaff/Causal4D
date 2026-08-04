# Deform360 source-backend failure attribution V1

This milestone decomposes the terminal **source-only** failure of the locked
six-object Deform360 replication. It uses the already archived 30 source
candidate grids, four pooled fits, two recorded source-stage failures, and the
terminal source-backend decision. It does not open any calibration outcome,
target prefix, target future geometry, or target tactile stream.

## Locked inputs

- Replication protocol config SHA-256:
  `f0aab308345807b2183f653306a062d4ad0295584b6b283deb99d29b3c247934`.
- Source-backend decision result SHA-256:
  `3603d273f6263dfe682631b9d9c72ae73b11ec2707a649f6d065cb6855e343a2`.
- Frozen source milestone manifest SHA-256:
  `594107433cda1be386210df85e150796240255f91e25436e43d21c0911e9ffa2`.
- Full attribution result SHA-256:
  `8da1a6112a7afb959a6bf81c3f870a8135d8414b599db98d67656ba40e98c9eb`.

All 51 files listed by the frozen source milestone manifest were rehashed before
analysis. The diagnostic validated all 30 candidate grids and reproduced the
four archived pooled-fit identities and scores from those grids.

## Result

No object passes even the per-episode quality-constrained oracle gate. Across
the 28 source episodes for which at least one candidate satisfies the locked
strain constraint, the per-episode oracle beats persistence in only 6 cases
(`21.43%`). Therefore, choosing one shared parameter vector is not the primary
failure: most objects already lack sufficient within-episode backend
competence.

| Object | Stratum | First failed boundary | Oracle CD | Persistence | Oracle wins |
| --- | --- | --- | ---: | ---: | ---: |
| `002-rope-silk` | filament | episode-level backend competence | 49.58 mm | 39.69 mm | 3/6 |
| `081-stripe-rope` | filament | episode-level backend competence | 35.64 mm | 38.31 mm | 3/6 |
| `085-scarf-cloth` | sheet | episode-level backend competence | 65.78 mm | 37.00 mm | 0/6 |
| `083-blanket-cloth` | sheet | per-episode physical feasibility | 85.37 mm* | 45.04 mm* | 0/4* |
| `092-squirrel` | volumetric | source observation/geometry | n/a | n/a | n/a |
| `170-spider` | volumetric | episode-level backend competence | 59.23 mm | 24.30 mm | 0/6 |

`*` The blanket statistics cover the four of six source episodes with at least
one strain-valid candidate. No candidate is strain-valid on every blanket
source episode.

The stripe rope is the only object whose mean per-episode oracle improves over
persistence (`+6.97%`), but it wins only three of six episodes and therefore
fails the registered `60%` win requirement. The scarf, blanket, and spider
oracles are substantially worse than persistence even after choosing a separate
candidate for every source episode.

## Scientific interpretation

The completed decomposition supports this order for the next separately
versioned source-only study:

1. repair source geometry availability for volumetric objects;
2. test representation and material-feasibility changes on sheets without
   relaxing the registered strain limit;
3. ablate contact realization, support registration, and within-episode
   dynamics competence on filament, sheet, and volumetric objects;
4. revisit shared or hierarchical physical parameters only after a method
   passes an episode-level competence gate.

This result argues against spending the next iteration on a larger shared
parameter grid alone. It does not identify one unique cause among graph
representation, support/contact registration, and dynamics; those mechanisms
require a new source-only ablation with its own lock.

## Reproduction

From the repository root:

```bash
python -m pip install -e ".[dev]"
python scripts/ci/analyze_deform360_source_failure.py \
  --output /tmp/deform360-source-failure-attribution.json
```

The command verifies the frozen source milestone before computing the result.
The dedicated `Deform360 source failure attribution` workflow runs the analysis
twice and requires byte-identical attribution artifacts and summaries.

## Claim and information boundary

- This is a diagnostic decomposition of an already failed source gate.
- The registered method, thresholds, source-backend decision, and zero-target
  access state are unchanged.
- `target_prefix_access_permitted=false`.
- `target_future_access_permitted=false`.
- No target result or target-informed model selection is introduced.

The compact machine-readable record is `summary.json`; the full result is
reconstructed deterministically from the checked-in frozen source artifacts.
