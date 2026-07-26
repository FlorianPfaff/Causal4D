# Observation-belief lineage

Causal4D can validate the versioned `phys4d.observation_belief` artifact emitted
by Prob4D and consumed by Bayesian-PhysTwin without importing either provider.
This preserves the repository boundary: Causal4D consumes a content-addressed
belief and does not reinterpret the feeder's raw files.

## Validation

```bash
causal4d-observation-lineage validate \
  observation_belief.npz \
  twin_belief.npz
```

Validation fails closed unless all of the following hold:

- the schema name, version, exact array set, and content digest are valid;
- every observation frame lies within the artifact's declared causal prefix;
- row identities, view/window references, group assignments, probabilities,
  covariance blocks, and low-rank factor shapes are valid;
- the observation and `TwinBelief` identify the same case;
- the observation interval is contained in the `TwinBelief` O-minus interval;
- an already bound `TwinBelief` names the same observation artifact.

Use `--require-bound` when the downstream command requires proof that the
`TwinBelief` was produced from the validated observation rather than merely
being compatible with it.

## Binding

The estimator that actually consumes an observation artifact should bind its
content address into the resulting `TwinBelief` metadata. The administrative
CLI supports this operation only with an explicit acknowledgement:

```bash
causal4d-observation-lineage bind \
  observation_belief.npz \
  unbound_twin_belief.npz \
  bound_twin_belief.npz \
  --confirm-observation-was-consumed
```

Binding creates a new content-addressed `TwinBelief`; it never mutates the
source artifact. The metadata records the observation belief ID, schema,
case/stream identity, causal cutoff, feeder repository/revision, and source
manifest digest.

The acknowledgement is not a substitute for estimator integration. It prevents
accidental binding during inspection, while the intended production path is for
the Bayesian-PhysTwin estimator to call
`bind_twin_belief_observation_lineage` immediately after constructing the
belief from that exact artifact.

## Cross-repository compatibility

Prob4D, Bayesian-PhysTwin, and Causal4D carry the same golden fixture. Its fixed
artifact ID detects changes to descriptor canonicalization, array names, dtypes,
shapes, or hashing rules before incompatible artifacts are exchanged.
