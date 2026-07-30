# Pre-acquisition readiness attestation

The registered v4 amendment defines the order of work before the 36-run
confirmatory experiment, but its Boolean collection gate is intentionally frozen
at `false`. It is a preregistration artifact, not a mutable checklist.

`causal4d protocol readiness` adds a separate, evidence-derived decision layer.
It never rewrites the v2, v3, or v4 protocol files. The first confirmatory
execution is permitted only when every prerequisite and operational gate below
validates from immutable, checksummed evidence.

## Decision contract

The readiness status distinguishes three states:

- **ready**: all evidence validates, all file hashes were checked, the registered
  chronology is respected, and no confirmatory manifest exists;
- **valid but incomplete**: the evidence tree is internally consistent, but one
  or more required artifacts are absent or still templates;
- **invalid**: present evidence is malformed or contradictory, an operational
  approval postdates the method freeze, or confirmatory collection already
  started before the gate opened.

With `--require-ready`, these states use exit codes `0`, `3`, and `2`,
respectively. This lets an acquisition launcher fail closed without confusing an
ordinary incomplete setup with corrupt evidence.

## Filesystem contract

Scaffolding creates five non-overwriting templates below the dataset root:

```text
preacquisition/
├── signature_panel.json
├── actuator_sync.json
├── support_registration.json
├── end_to_end_dry_run.json
├── software_environment.json
└── source_panel/executions/<execution-id>/manifest.template.json
```

Each gate record binds the locked protocol and v4 amendment, its underlying
files, completion and approval timestamps, the no-target-outcomes boundary, and
a canonical SHA-256 digest. The scaffold also writes one source-panel manifest
template per registered execution; operators promote completed copies to
`manifest.json` at the corresponding paths. `seal-gate` verifies all bound files
before replacing the completed gate template atomically. A sealed gate cannot be
resealed. Evidence descriptors use the same relative-path contract throughout:

```json
{"path": "relative/file", "sha256": "<64 lowercase hex>", "bytes": 123}
```

### Signature panel

The signature gate requires the exact 12 v2 source-panel execution IDs in their
registered order and 12 independent reset/grasp sessions. Every bound
`SourcePanelExecutionManifest` must state that it is complete, source-only,
outside all confirmatory folds, included without quality-gate failures, and did
not use target outcomes. Its artifact descriptors are hash-verified.

### Actuator synchronization

The actuator gate requires one checksummed
`ActuatorRealizationCalibration` artifact for every source-panel execution. It
checks the locked PyRecEst version, the source/dry-run information boundary,
hardware-timestamp authority, artifact identity, and the registered maximum
RGB-D/actuator synchronization error.

### Support and gravity registration

The support gate records the locked world frame, measured gravity vector,
support geometry, registration closure error, and the threshold chosen by the
pre-acquisition calibration procedure. The registration file must be included in
the gate's evidence descriptors.

### End-to-end dry run

The dry-run gate requires a nonconfirmatory execution ID and successful exercise
of all registered stages:

```text
synchronized_acquisition
observation_prefix_build
intervention_abduction
held_out_prediction
artifact_hash_validation
status_generation
```

The dry run may not reuse a confirmatory execution ID or use target outcomes.

### Software lineage

The software-environment gate is sealed after the method freeze. It binds:

- the exact method-freeze and independent-attestation file hashes;
- the frozen Causal4D and Bayesian-PhysTwin commits;
- package versions and SHA-256 descriptors for the installed wheel or equivalent
  immutable distribution artifact;
- an explicit Prob4D declaration. When Prob4D supplies claim-bearing
  observations, its commit, version, distribution, and observation-contract
  version are required. When it is not used, the record must say so and give a
  reason;
- the actual observation producer and Python runtime identity.

This makes the executed bytes and observation producer explicit without changing
the registered scientific method.

## Workflow

Create templates after scaffolding the registered dataset:

```bash
causal4d protocol readiness scaffold \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1
```

Complete each JSON template and its referenced evidence, then seal it. Operational
source-panel gates must be completed and approved before the method freeze. For
example:

```bash
causal4d protocol readiness seal-gate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  signature_panel_complete \
  --approved-by "<reviewer>"
```

Seal the software environment after `method_freeze.json` and its independent
attestation validate:

```bash
causal4d protocol readiness seal-gate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  software_environment_locked \
  --approved-by "<independent-verifier>"
```

Finally, require a hash-verified ready decision before execution 1:

```bash
causal4d protocol readiness status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes \
  --require-ready \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/preacquisition-readiness.json
```

The resulting status includes the derived collection flags, prerequisite and gate
validation details, confirmatory-manifest counts, blockers, and its own canonical
SHA-256 digest. Only `ready=true` and
`first_confirmatory_execution_allowed=true` authorize the first confirmatory
execution.
