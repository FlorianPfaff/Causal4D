# Bayesian-PhysTwin provider boundary

Causal4D consumes Bayesian-PhysTwin through the public
`bayesian_phystwin.causal4d_provider_v1` module. The adapter no longer imports
experiment-specific or underscore-prefixed Bayesian-PhysTwin implementation
modules.

## Compatibility validation

Before constructing a `TwinBelief`, Causal4D converts the provider manifest to
its independent `PhysicalBeliefProviderManifest` representation and validates:

- provider schema version 1;
- content-addressed artifacts;
- particle-specific endpoint positions and velocities;
- physical parameter particles;
- readout-discrepancy moments;
- exact complete-belief fallback capability;
- `PhysicalBeliefV1` artifact schema version 1.

The manifest content address must be identical under both repositories'
canonicalization rules. A missing capability, unsupported schema, artifact
version mismatch, or canonicalization mismatch fails closed before inference.

## Execution boundary

The pinned provider exposes:

- `build_physical_belief_from_replays` for already available replay arrays;
- `replay_official_phystwin_particles` for the optional Warp execution path;
- public wrappers for released target-validity and collision conventions;
- non-pickled `PhysicalBeliefV1` artifacts with endpoint state, physical
  parameters, particle weights, and discrepancy moments.

Causal4D translates the validated provider artifact into its own `TwinBelief`
contract. Metadata records the exact provider revision, provider manifest ID,
provider physical-belief ID, required capabilities, and successful compatibility
validation.

## Reproducibility policy

The `phystwin` extra pins the exact provider commit used by this integration.
Frozen experiments should retain that exact revision. Normal development may
advance the pin only when the provider schema and capability checks pass and the
cross-repository integration tests remain green.

This boundary does not transfer ownership of causal inference to
Bayesian-PhysTwin. Bayesian-PhysTwin supplies the uncertain physical belief;
Causal4D owns realized-intervention abduction, intervention, and downstream
prediction.
