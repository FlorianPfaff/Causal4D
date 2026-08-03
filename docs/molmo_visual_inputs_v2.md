# Digest-bound PhysTwin visual inputs for MolmoMotion

The command `causal4d experiment semantic forecast` prepares MolmoMotion queries
through `bayesian_phystwin.causal4d_artifacts_v2`. It is additive; the historical
`causal4d archive semantic forecast-v1` command remains available for frozen runs.

## Manifest

The new command requires one JSON manifest with this exact schema:

```json
{
  "schema_name": "causal4d.phystwin_visual_inputs",
  "schema_version": 2,
  "provider_api": "bayesian_phystwin.causal4d_artifacts_v2",
  "final_data_sha256": "<sha256>",
  "metadata_sha256": "<sha256>",
  "pcd_sha256": "<sha256>",
  "calibration_sha256": "<sha256>",
  "cotracker_sha256": {
    "cotracker/0.npz": "<sha256>"
  },
  "image_sha256": {
    "color/0/120.png": "<sha256>",
    "color/0/122.png": "<sha256>",
    "color/0/124.png": "<sha256>"
  },
  "initial_match_tolerance_m": 0.000001
}
```

The CoTracker inventory must identify every released archive in the raw case.
The RGB inventory must identify exactly the history images implied by the
selected camera, `train_end_frame`, history size, and forecast cadence. Absolute
paths and parent-directory traversal are rejected.

## Execution boundary

Query preparation:

1. asks Bayesian-PhysTwin to verify and postflight `final_data.pkl`, metadata,
   PCD, calibration, and every CoTracker archive;
2. consumes only the immutable artifact returned by that provider;
3. chooses camera-local candidates without indexing one camera's archive using
   raw track IDs belonging to another camera;
4. verifies the selected RGB history images; and
5. records the provider artifact ID, manifest digest, and image digests in query
   metadata.

The V2 inference wrapper verifies the RGB files again immediately before and
after MolmoMotion execution. This detects changes between query preparation and
model consumption.

## Provider compatibility

The V2 command requires an installed Bayesian-PhysTwin revision that exports
`bayesian_phystwin.causal4d_artifacts_v2` with artifact schema version 2. The
provider PR must merge first; the Causal4D branch must then be validated against
that merged revision through the installed-wheel compatibility gate. A core-only
Causal4D installation remains independent of Bayesian-PhysTwin and can still
render every CLI help surface.

## Claim boundary

The manifest proves byte identity and correspondence provenance. It does not
establish tracker calibration, MolmoMotion accuracy, or physical-prediction
improvement. A SHA-256 digest must come from an independently trusted protocol
or data manifest; a digest supplied alongside untrusted bytes is not evidence.
