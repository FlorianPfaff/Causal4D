# External sparse trajectory forecast import

Causal4D can ingest sparse 3-D trajectory forecasts produced outside the
MolmoMotion/PhysTwin environment. The importer normalizes producer-specific
arrays into a versioned, non-pickled, content-addressed
`ExternalForecastBundle` and preserves the physical posterior as an immutable
fallback.

The canonical artifact contains:

- exact physical node indices;
- finite metric anchor positions;
- one or more future sparse trajectories in the physical world frame;
- explicit physical-posterior frame indices;
- coordinate-level validity masks;
- source-model and source-revision provenance; and
- SHA-256 identities for every numerical payload and the producer input files.

It does not interpret a forecast as a physical-state update. The existing task
posterior scores only the selected readout `H_Q(X)` of complete physical
rollouts. A rejected or zero-weight semantic branch therefore leaves the
physical posterior unchanged.

## Import manifest v1

The producer writes a normal NumPy NPZ file with `allow_pickle=False`-compatible
arrays. A separate UTF-8 JSON manifest maps those arrays to the portable
contract:

```json
{
  "schema": "causal4d.external_forecast_import",
  "schema_version": 1,
  "case_id": "cloth_lift_001",
  "source": {
    "model": "MolmoMotion-adapted",
    "revision": "checkpoint-or-git-revision",
    "artifact_id": "optional-upstream-artifact-id"
  },
  "arrays": {
    "node_indices": "node_indices",
    "anchor_positions": "anchor_positions_world_m",
    "future_positions": "future_positions_world_m",
    "future_times_s": "future_times_s",
    "validity_mask": "validity_mask"
  },
  "layout": "KPFC",
  "coordinate_frame": "world",
  "position_unit": "m",
  "forecast_ids": ["instruction", "shuffled"],
  "anchor_physical_frame": 0,
  "physical_fps": 30.0,
  "forecast_metadata": {
    "instruction": {"caption": "Lift the cloth upward."},
    "shuffled": {"caption": "Push the cloth sideways."}
  },
  "metadata": {
    "producer_environment": "external-molmo-venv"
  }
}
```

Required producer arrays are `node_indices`, `anchor_positions`, and
`future_positions`. The supported position layouts are:

| Layout | Producer shape |
| --- | --- |
| `PFC` | `(point, future, xyz)` for one forecast |
| `FPC` | `(future, point, xyz)` for one forecast |
| `KPFC` | `(forecast, point, future, xyz)` |
| `KFPC` | `(forecast, future, point, xyz)` |

The validity mask may omit the final coordinate axis. Invalid coordinates are
canonicalized to NaN and excluded from scoring.

The importer accepts `m`, `cm`, or `mm`. For `coordinate_frame: "camera"`, the
manifest must additionally map `arrays.camera_to_world` to a finite homogeneous
`4 x 4` matrix. Its translation is expressed in metres; positions are converted
to metres before the transform is applied.

Frame alignment must be explicit. Supply either:

- `arrays.physical_frame_indices`; or
- `arrays.future_times_s` together with `physical_fps`.

When both are present, the importer verifies

```text
physical_frame = anchor_physical_frame + future_time_s * physical_fps
```

and fails closed on disagreement.

## Python API

```python
from causal4d.external_forecast import (
    import_external_forecast,
    save_external_forecast,
)

bundle = import_external_forecast(
    "producer_forecast.npz",
    "external_forecast_manifest.json",
)
save_external_forecast("canonical_external_forecast.npz", bundle)
```

The low-level module runner provides the same operation without adding another
installed console script:

```bash
python -m causal4d.cli.external_forecast_import \
  producer_forecast.npz \
  external_forecast_manifest.json \
  canonical_external_forecast.npz
```

The canonical artifact can then be passed to the existing grouped task-posterior
command in place of a legacy MolmoMotion NPZ:

```bash
causal4d experiment semantic build-task-posterior \
  physical_posterior.npz \
  canonical_external_forecast.npz \
  instruction \
  task_posterior.npz \
  --beta 0
```

`beta=0` remains byte-identical to the physical weights. A positive beta is
exploratory unless that external producer has its own source-only competence,
trust, and acceptance evidence. Importing an artifact does not make it eligible
under the MolmoMotion-specific acceptance result.

## Fail-closed checks

Import and reload reject:

- duplicate or non-string JSON keys;
- unexpected manifest or artifact fields;
- object arrays that require pickle;
- duplicate or negative physical node indices;
- shape/layout mismatches;
- non-finite valid coordinates or anchors;
- non-increasing future times or physical frames;
- frame/time inconsistencies;
- missing camera transforms for camera-frame inputs;
- payload mutation after publication; and
- source or manifest files that change while being imported.

The artifact identity is independent of local path names. It binds the exact
producer NPZ bytes, import-manifest bytes, normalized arrays, validity mask,
frame alignment, source identity, and finite JSON metadata.
