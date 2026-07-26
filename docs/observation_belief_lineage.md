# Observation lineage

Causal4D validates portable observation artifacts without importing Prob4D or
Bayesian-PhysTwin. Two related interfaces are supported:

1. `phys4d.observation_belief` is a marginalized, content-addressed observation
   belief suitable for grouped likelihoods and simple consumers.
2. `prob4d.observation-factor-bundle` schema v3 preserves the exact unfused
   points, conditional covariance, explicit `Sim(3)` gauge variables, separate
   reliability fields, and the exclusive causal cutoff consumed by the
   gauge-aware Bayesian-PhysTwin update.

The second interface is the stronger estimator-provenance binding. A compatible
marginalized belief does not prove that the estimator consumed the exact
factor-bundle manifest and payload pair.

## Observation-belief validation

```bash
causal4d-observation-lineage validate \
  observation_belief.npz \
  twin_belief.npz
```

Validation fails closed unless:

- the schema name, version, exact array set, and content digest are valid;
- every observation frame lies within the artifact's declared causal prefix;
- row identities, view/window references, group assignments, probabilities,
  covariance blocks, and low-rank factor shapes are valid;
- the observation and `TwinBelief` identify the same case;
- the observation interval is contained in the `TwinBelief` O-minus interval;
- an already bound `TwinBelief` names the same observation artifact.

Use `--require-bound` when a downstream command requires proof that the
`TwinBelief` was produced from the artifact rather than merely being compatible
with it.

## Strict Prob4D causal stream

When an artifact declares repository `FlorianPfaff/Prob4D` and stream
`prob4d:causal-overlap-window-points`, Causal4D performs an additional
provider-independent validation directly from the descriptor and numeric
arrays. It requires:

- an exact producer revision rather than `unknown`;
- the frozen seven-dimensional gauge-factor names and factor-group/window
  mapping;
- metric units, a nonempty world frame, and a fixed external metric anchor;
- binding of that anchor to the first selected source payload;
- the registered MotionCrafter windowing model and independently decoded
  overlap-window product;
- equality of descriptor and lineage cutoffs and source digests;
- zero reported future-payload access;
- source bounds for every selected window wholly before the cutoff; and
- containment of every observation row within its declared source window.

This check is implemented independently from Prob4D and Bayesian-PhysTwin. A
syntactically valid generic observation artifact cannot acquire a causal Prob4D
claim merely by using the provider's repository name. The completed validation
summary is retained when the observation belief is bound to a `TwinBelief`.

## Exact factor-bundle validation

```bash
causal4d-observation-lineage validate-factor-bundle \
  observation_factors.json \
  twin_belief.npz
```

Causal4D independently checks the schema-v3 manifest and its checksum-bound NPZ
payload. Validation includes:

- exact manifest and payload SHA-256 values and an artifact ID derived from both;
- a relative payload path that cannot escape the manifest directory;
- explicit case, stream, sequence, repository, and producer revision identities;
- one exclusive `causal_frame_stop` shared by every factor;
- unique factor and gauge identities and valid gauge references;
- finite `Sim(3)` means and positive-semidefinite gauge covariance;
- the exact payload array set, types, dimensions, and per-factor row identities;
- separate association probability and residual-independent prior reliability;
- fixed nominal-component probability and composite weight within every
  correlation group;
- finite active points, covariance, and optional rays;
- case equality and containment in the `TwinBelief` O-minus interval.

Schema-v2 factor bundles are deliberately not accepted here. Prob4D must first
load and rewrite them as schema v3 so that the exclusive cutoff and newly
separated reliability fields are explicit in the exact artifact being bound.

## Binding

The estimator that actually consumes an artifact should bind its content
address into the resulting `TwinBelief`. Administrative binding requires an
explicit acknowledgement.

For a marginalized observation belief:

```bash
causal4d-observation-lineage bind \
  observation_belief.npz \
  unbound_twin_belief.npz \
  bound_twin_belief.npz \
  --confirm-observation-was-consumed
```

For an exact Prob4D factor bundle:

```bash
causal4d-observation-lineage bind-factor-bundle \
  observation_factors.json \
  unbound_twin_belief.npz \
  bound_twin_belief.npz \
  --confirm-factor-bundle-was-consumed
```

Binding creates a new content-addressed `TwinBelief`; it never mutates the
source artifact. Marginalized-belief metadata records the artifact identity,
case/stream, causal cutoff, feeder repository/revision, source digest, and any
completed provider-specific validation summary. Factor-bundle metadata records:

- the exact combined artifact ID;
- manifest and payload SHA-256 values;
- schema and version;
- case, stream, and producer sequence identity;
- exclusive causal cutoff;
- source repository and revision.

When a binding already exists, validation checks every recorded lineage field,
not only the top-level artifact ID. A partial or internally inconsistent binding
fails closed.

The acknowledgement is not a substitute for estimator integration. Production
Bayesian-PhysTwin code should bind the exact artifact immediately after
constructing the belief from it. Inspection of a merely compatible artifact must
not create provenance.

## Cross-repository compatibility

Prob4D owns the observation producers. Bayesian-PhysTwin owns the state, gauge,
bias, and guarded-update inference. Causal4D independently checks the immutable
artifact identities and causal producer claims before using the resulting
physical belief. This preserves the dependency direction:

```text
Prob4D observation artifact
        -> Bayesian-PhysTwin guarded belief
        -> Causal4D lineage validation and counterfactual inference
```

The existing cross-repository golden `ObservationBeliefV1` fixture detects
changes to the marginalized contract. The factor-bundle validator adds an exact
byte-level lineage check for the richer estimator input.