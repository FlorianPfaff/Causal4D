# Resumable PhysTwin Rollout Cache

The official PhysTwin rollout commands persist every completed
`replay_restart` call as an immutable, content-addressed NPZ record. A failed or
interrupted rollout-bank build can therefore resume from the completed physical
simulations instead of restarting the complete hypothesis-by-particle grid.

Caching is enabled by default for:

```text
causal4d experiment phystwin rollout-bank
causal4d experiment phystwin counterfactual
```

Without an explicit option, the cache is written beside the primary output as
`<output-stem>.rollout-cache/`. Use a shared location across compatible runs
with:

```bash
--rollout-cache-dir /path/to/phystwin-rollouts
```

Use `--no-rollout-cache` only when per-rollout persistence is intentionally
undesired.

## Cache identity

Each key binds all physical inputs that can affect one replay:

- Bayesian-PhysTwin provider version, API manifest, revision, and installed
  source fingerprint;
- official PhysTwin revision plus tracked, staged, and untracked source changes;
- SHA-256 digests of the case data, optimal parameters, checkpoint, released
  baseline trajectory, and Bayesian parameter profile;
- shifted spring-graph arrays and topology metadata;
- transformed controller trajectory;
- grouped spring log-scales;
- particle-specific endpoint position and velocity;
- replay frame interval;
- `dt`, substeps, collision mode, deterministic-force mode, parameterization,
  device, and numerical package versions.

Paths and hypothesis labels are deliberately excluded from the key. Identical
physical calls can therefore be reused across output directories and across
hypotheses that reduce to the same simulator inputs.

## Validation and publication

Records use non-pickled compressed NPZ files. Every read validates the schema,
content-addressed key, canonical descriptor, trajectory dimensions, finite
values, and an array checksum. A corrupt record is recomputed and atomically
replaced. New records are written to a temporary file, flushed, and published
without exposing a partial archive; concurrent writers reuse the validated
winner.

The final rollout-bank manifest contains one audit record per
`(hypothesis, parameter particle)` component, including the cache key, relative
record path, hit/miss/repair status, physical-input digests, trajectory digest,
and source/runtime identities.

For deterministic spring forces, a repeated compatible run reconstructs the
same rollout bank without constructing a real Warp provider when every record
is present. In nondeterministic mode, the cache intentionally freezes the
published sample for each content key; the manifest labels this behavior
explicitly.
