# Changelog

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
