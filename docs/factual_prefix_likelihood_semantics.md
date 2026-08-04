# Versioned factual prefix-likelihood semantics

Causal4D retains the registered dense factual-abduction likelihood as
`legacy_v1` and exposes a separate opt-in development contract,
`normalized_v2`. The default remains `legacy_v1`; no frozen result or existing
artifact is rewritten.

## Why the paths are versioned

The original single-execution factual-abduction path and the later hierarchical
abduction path used related but not identical Student-t composite likelihoods.
The distinction matters when physical particles carry different discrepancy
variances or when the endpoint-to-first-response increment contains intervention
information.

Silently changing the registered path would alter posterior weights and
content-addressed factual-intervention artifacts. Causal4D therefore makes the
choice explicit instead.

## `legacy_v1`

`legacy_v1` preserves the existing `JointRolloutBank.update_from_observations`
behavior exactly:

- the position block scores response frames after the endpoint;
- the dynamic block differences only the already-selected response frames;
- the dynamic scale reuses the declared observation scale;
- particle-specific discrepancy variance changes the standardized residual scale,
  but the historical score omits the corresponding scale-normalization term.

The default metadata retains the original four fields only:

```json
{
  "observation_scale_m": 0.01,
  "likelihood_power": 12.0,
  "dynamic_likelihood_weight": 0.25,
  "degrees_of_freedom": 4.0
}
```

This preserves legacy artifact identities and the registered physical protocol.

## `normalized_v2`

`normalized_v2` delegates to the shared correlation-aware prefix likelihood used
by hierarchical abduction. It:

- retains the Student-t `-log(scale)` term when component scales differ;
- differences the complete admitted prefix, including the
  endpoint-to-first-response increment;
- uses
  `observation_scale_m * sqrt(2 * (1 - difference_correlation))` for adjacent
  observation differences;
- keeps a time-persistent discrepancy mean out of the dynamic block because it
  cancels under differencing;
- uses static discrepancy variance only in the position block.

The selected semantics and adjacent-frame correlation are added to the factual
artifact metadata, so a normalized result cannot be mistaken for a legacy one.

## Command line

The factual-abduction command defaults to the registered path:

```bash
causal4d experiment phystwin abduct-intervention \
  rollout-bank.npz twin-belief.npz final_data.pkl \
  factual.npz evaluation.json
```

Run the development comparator explicitly with:

```bash
causal4d experiment phystwin abduct-intervention \
  rollout-bank.npz twin-belief.npz final_data.pkl \
  factual-normalized-v2.npz evaluation-normalized-v2.json \
  --likelihood-semantics normalized_v2 \
  --difference-correlation 0.25
```

`difference_correlation` is accepted only with `normalized_v2`. The normalized
dense path cannot be combined with the separate grouped full-covariance evidence
path; contradictory requests fail rather than silently ignoring one model.

## Promotion boundary

`normalized_v2` is implementation-ready but is not the registered Causal4D
estimator. It must be evaluated under a source-only selection rule and a fresh,
untouched contact-inference panel before any method-version decision. In
particular, it cannot revise the frozen seeds `0:5`, the independent `100:120`
panel, the `200:220` concentration diagnostic, or the locked 36-execution
physical experiment.

A future promotion requires a new method version, explicit evidence that a proper
score improves without materially degrading trajectory accuracy, contact
accuracy, or coverage, and another untouched evaluation panel. A negative result
remains a complete outcome.
