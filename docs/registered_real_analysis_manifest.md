# Registered real-analysis manifest

The confirmatory Causal4D analysis is now sealed as one self-contained,
content-addressed manifest after the method freeze and before target access. The
manifest does not change the estimator, protocol, split, calibration threshold,
or reporting rules. It makes the rules already distributed across the protocol,
method freeze, and reporting implementation independently inspectable in one
artifact.

## Seal the manifest

Run this from the exact clean checkout referenced by `method_freeze.json`:

```bash
causal4d protocol real analysis-manifest-seal \
  /opt/causal4d-frozen \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  /data/causal4d-sloth-multi-action-v1/registered-analysis.json \
  --registered-by "<independent-registrar>"
```

The command revalidates the protocol, every method-freeze file digest, the exact
Causal4D and BayesianPhysTwin revisions, and the frozen analysis and reporting
contracts. Publication is atomic and non-overwriting. The output records both:

- `analysis_id`: the canonical SHA-256 of the logical manifest excluding its own
  identity field; and
- the exact file SHA-256 and byte count used by readiness, evidence, and result
  artifacts.

A second publication to the same path fails. A changed method freeze, protocol,
bootstrap rule, comparison arm, calibration rule, endpoint inventory, or claim
boundary produces a different identity or fails validation.

## Revalidate retained bytes

```bash
causal4d protocol real analysis-manifest-validate \
  /opt/causal4d-frozen \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  /data/causal4d-sloth-multi-action-v1/registered-analysis.json
```

Validation reopens all three source files and verifies their exact identities.
The result is target-free and cannot authorize acquisition by itself.

## Readiness admission

The canonical `causal4d protocol readiness status` path requires
`registered-analysis.json` as a first-class prerequisite. It verifies the exact
method-freeze SHA-256, protocol and amendment identities, Causal4D and
BayesianPhysTwin revisions, manifest content identity, and registration
chronology. The collection gate exposes:

```text
primary_analysis_registered=true
```

A missing, malformed, consistently re-addressed policy change, pre-freeze
registration, or software-identity mismatch keeps
`first_confirmatory_execution_allowed=false`. The lower-level readiness evaluator
retains an explicit opt-in for isolated contract tests, but the canonical
repository-and-dataset builder always enables this prerequisite.

## Bound analysis contract

Schema version 2 closes and content-addresses:

- the protocol, v4 amendment, method freeze, Causal4D revision, and pinned
  BayesianPhysTwin revision;
- the six-frame causal prefix and zero-future-frame selection boundary;
- the exact primary and diagnostic command entrypoints;
- nominal PhysTwin, BayesianPhysTwin with nominal realized intervention, frozen
  Causal4D, and the diagnostic intervention oracle as distinct arms;
- complete factual, same-grasp, and new-contact endpoint inventories;
- equal-target-session effects with 20,000 deterministic session bootstrap
  replicates at 95% confidence;
- the 12-fold execution-block calibration contract with nine independent
  calibration units, rank 9 of 9, and no target threshold reselection;
- complete failure and preregistered-exclusion accounting;
- the obligation to report success or a well-powered negative result; and
- the same-object, non-SOTA, non-safety, and non-raw-covariance-calibration claim
  boundary.

The coarse nine-unit calibration design is therefore visible in the primary
analysis artifact rather than discoverable only from prose. Fragility diagnostics
remain mandatory and cannot select another threshold.

## Compatibility

The result-source verifier continues to accept historical schema-version-1
manifests for already frozen consumers. New confirmatory acquisition should use
the schema-version-2 sealing command. Schema 2 is strict: even a consistently
re-addressed policy change is rejected when any fixed analysis field differs.

## Scientific boundary

The manifest is preregistration infrastructure, not empirical evidence. It reads
no physical target outcome, does not increment the `0/36` acquisition count, and
cannot rescue a failed factual, transfer, new-contact, or calibration gate.
