# Changelog

## Unreleased

### Added

- Add a fail-closed source-panel status that validates the exact registered
  12-execution prefix, reports the next physical execution with its complete
  command profile, distinguishes invalid evidence from valid incompleteness, and
  requires hash verification before completion.
- Add exactly-once source-manifest publication. The publisher admits only the
  next registered execution, recursively rejects target-outcome fields, verifies
  every referenced artifact digest and byte count, validates the temporary
  manifest, and never overwrites a final evidence path.
- Add the physical source-panel operator runbook and adversarial coverage for
  out-of-order completion, modified templates, stale artifact hashes, incomplete
  status, and repeated publication.
- Add content-addressed rollout-bank archives with exact member inventories,
  strict finite-JSON manifests, atomic validation-before-replace publication,
  explicit no-overwrite support, and legacy archive loading.
- Add a stable rollout-bank identity over hypothesis metadata, priors, physical
  parameter support, trajectories, variance floor, and confidence level.
- Add explicit dense factual-abduction likelihood semantics. `legacy_v1` remains
  the registered identity-preserving default, while opt-in `normalized_v2`
  retains particle-specific scale normalization, includes the
  endpoint-to-first-response increment, and models adjacent-frame correlation.

This control plane advances pre-acquisition operations without creating physical
evidence. Source-panel executions remain source-only and cannot increment the
`0/36` confirmatory evidence count.

### Fixed

- Preserve the initial prior mass removed by contact-path pruning, including
  one-frame paths, and fail explicitly when the threshold removes every initial
  contact regime.
- Reject non-finite, zero, or negative fixed-contact posterior temperatures
  before simulation instead of allowing invalid posterior weights to propagate.
- Enforce strict prefix-only validation for fixed-contact and rollout-bank online
  updates, including zero-power calls, and reject invalid parameter-support limits.
- Normalize and recursively freeze rollout-hypothesis metadata so external or
  nested mutation cannot change intervention semantics after bank construction.
- Route rollout-producing and rollout-consuming commands through one strict
  archive implementation, while retaining exact support for legacy banks.
- Reject non-finite factual-abduction controls, correlation outside `(-1, 1)`,
  legacy requests that specify a correlation model, and contradictory requests
  for normalized dense and grouped full-covariance likelihoods.

These changes harden diagnostic accounting, artifact integrity, and validation.
The registered `legacy_v1` likelihood and posterior remain unchanged, and
`normalized_v2` is not admitted into the frozen estimator, protocol, thresholds,
evidence, or target identities.

## 0.5.0

### Breaking CLI consolidation

- Install exactly one executable: `causal4d`.
- Remove all 67 historical `causal4d-*` console scripts from current packages.
- Preserve every historical name as machine-readable migration metadata.
- Expose all retained functionality through typed grouped routes.
- Add lifecycle, optional-extra, provider, owner, and claim-boundary metadata.
- Require installed wheel and source-distribution tests to exercise `--help` for
  every grouped route and reject any residual historical wrapper.
- Update the registered real-analysis command strings to grouped invocations.
- Preserve the same fail-closed provenance, evidence, and posterior validation
  inside each handler; the migration changes invocation paths, not scientific
  semantics or admissibility checks.

Frozen tags, milestone files, and recorded environments are unchanged.
