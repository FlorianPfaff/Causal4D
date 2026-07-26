# Content-addressed PhysTwin rollout cache

Causal4D can optionally cache each expensive official-PhysTwin restart rollout
before assembling a `JointRolloutBank`. The cache is disabled by default and
does not alter frozen commands or numerical semantics.

## Enabling the cache

The rollout-bank and counterfactual PhysTwin commands accept an explicit cache
directory:

```bash
causal4d-phystwin-rollout-bank \
  /path/to/PhysTwin /path/to/case profile.npz checkpoint.pt bank.npz \
  --rollout-cache-dir /scratch/causal4d-phystwin-cache

causal4d-counterfactual-phystwin \
  /path/to/PhysTwin /path/to/case profile.npz checkpoint.pt \
  twin_belief.npz factual_intervention.npz physical_posterior.npz \
  --rollout-cache-dir /scratch/causal4d-phystwin-cache
```

A cache hit skips one `replay_restart()` call. A cache miss runs the provider,
atomically writes the complete trajectory, reloads it, and verifies the stored
content before the rollout bank consumes it.

## Cache identity

One version-1 cache key covers:

- the replay-provider and graph-provider manifest IDs;
- the exact clean official PhysTwin Git revision;
- SHA-256 digests of `final_data.pkl`, `optimal_params.pkl`, the checkpoint,
  released baseline trajectory, and Bayesian-PhysTwin parameter profile;
- every spring-graph array and object-boundary count;
- the transformed controller trajectory;
- particle-specific endpoint position and velocity;
- grouped spring log-scales;
- the restart frame interval; and
- timestep, substeps, collision, force determinism, parameterization, and device.

Array digests bind dtype and shape as well as bytes. Cache records contain their
canonical descriptor, content address, trajectory dtype/shape, and trajectory
digest. Loading fails closed on missing fields, descriptor drift, digest drift,
shape drift, or nonfinite values.

## Reproducibility restrictions

Caching requires:

- deterministic spring forces;
- a clean official PhysTwin Git checkout with an exact 40-character revision;
- finite, validated rollout inputs; and
- the same provider manifests and runtime configuration.

A dirty or non-Git PhysTwin checkout is rejected rather than assigned an
ambiguous cache identity. Cache paths are execution infrastructure and are not
part of the scientific result identity; the rollout manifest records whether
the cache was enabled plus hit/miss/write counts and the content IDs used.

## Concurrency and recovery

Entries are written through a temporary file in the target shard and published
atomically only after the NPZ payload is complete. Existing entries are verified
before reuse, so interrupted jobs leave no partially readable target. Re-running
a failed bank assembly reuses all completed per-rollout records and computes
only the missing hypotheses and parameter particles.

The cache stores provider outputs, not posterior decisions. Reweighting,
intervention abduction, discrepancy handling, and final bank assembly remain
outside the cache and continue to bind their own artifacts and manifests.
