# Cross-repository project-status contract

`ci/project_status_v1.json` is the machine-readable development status shared by
the Causal4D, Bayesian-PhysTwin, and Prob4D installed-wheel path. It records two
different kinds of information that must not be conflated:

1. package-version compatibility needed to execute the current development
   stack; and
2. the empirical claim boundary supported by completed experiments.

The current contract records:

- Causal4D `0.5.0`;
- Bayesian-PhysTwin compatibility `>=0.4,<0.5`;
- Prob4D development compatibility `>=0.3,<0.4`;
- a passed controlled counterfactual result;
- pending same-object multi-action validation and independent-execution
  calibration;
- a pending prospective Prob4D-to-Bayesian-PhysTwin experiment; and
- semantic reweighting as not admitted into the primary method.

## Automated checks

`ci/three_repository_status.py` fails closed when:

- Causal4D package metadata and `causal4d.__version__` disagree with the status
  contract;
- the recorded Bayesian-PhysTwin range differs from
  `BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE`;
- an installed Bayesian-PhysTwin or Prob4D wheel lies outside its recorded
  development range;
- the status record changes the decisive next milestone;
- a pending empirical result is promoted without versioning the contract; or
- explicit non-claims are removed.

The ordinary core test suite validates Causal4D and its provider range. The
private installed-wheel golden path additionally validates all three installed
distributions and writes the status ID, canonical SHA-256 digest, package
versions, next milestone, claim status, and empirical status into its JSON
summary.

The runner resolves the record from
`$GITHUB_WORKSPACE/causal4d/ci/project_status_v1.json` in GitHub Actions and from
`ci/project_status_v1.json` when launched at the repository root. No source
checkout is imported; the package-version checks run against the three built and
installed wheels.

## Update policy

A package release may update its version or compatibility range without changing
the empirical status. Conversely, a scientific claim may change only when the
corresponding locked experiment is complete and reported. Either change must be
made explicitly in the versioned status record and must pass the same installed
stack checks.

This contract does not replace exact commit pins used by frozen milestones or
method-freeze manifests. It is the ordinary-development compatibility and claim
status, while frozen experiments continue to bind exact revisions, artifacts,
and checksums.
