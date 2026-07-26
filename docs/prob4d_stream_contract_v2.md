# Prob4D causal observation stream contract v2

Causal4D validates Prob4D observation artifacts independently from both Prob4D
and Bayesian-PhysTwin. The neutral archive schema remains
`phys4d.observation_belief` version 1; the provider-specific interpretation is
identified by `prob4d_causal_stream_contract_version`.

## Supported interpretations

- **Version 1:** seven `gauge_latent_*` columns per independent window factor
  group. This is retained for frozen experiments.
- **Version 2:** canonical `joint_gauge_latent_####` columns in one shared factor
  group. The shared latent represents the joint cross-window `Sim(3)` covariance
  propagated from a fixed metric anchor through a causal sequential gauge tree.

Version 2 is admitted only when the descriptor and metadata agree on the factor
rank, `7K` full gauge dimension, retained covariance-trace threshold, parent
lineage, sequential gauge mode, and non-approximate boundary covariance.

Prob4D 0.2.0 emitted the joint representation before adding the explicit stream
version. Causal4D can recognize those canonical transitional artifacts and marks
`stream_contract_version_inferred` in the validation result. New artifacts must
carry the explicit version and the complete metric-anchor schema.

Approximate fixed-lag covariance is intentionally rejected from the strict
stream. Reconstruction controls may retain it, but they must not be admitted as
an uncertainty-preserving causal observation product.

The resolved contract version and covariance semantics are retained in the
validated observation lineage so later intervention artifacts remain auditable
without importing the upstream provider implementation.
