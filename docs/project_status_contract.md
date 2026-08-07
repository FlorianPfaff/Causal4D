# Cross-repository project-status contract

`ci/project_status_v2.json` is the authoritative machine-readable development
status shared by Causal4D, Bayesian-PhysTwin, and Prob4D. The historical
`ci/project_status_v1.json` remains in the source distribution so frozen
workflows can still validate the record they were built against.

The status contract deliberately separates:

1. package-version compatibility for the current development stack;
2. evidence established in controlled studies;
3. evidence still pending from a fresh real provider or physical experiment;
4. confirmatory-method admission; and
5. registered acquisition progress.

The human-readable
[`docs/current_project_status.md`](current_project_status.md) file is generated
from the v2 JSON record. It is not an independent source of truth.

## Version 2 status model

Version 2 records the following states independently:

- the controlled Causal4D counterfactual result and its immutable milestone
  revision;
- the registered 36-execution same-object physical experiment, including
  acquired and validated counts plus `claim_ready`;
- independent-execution calibration;
- the controlled synthetic Prob4D-to-Bayesian-PhysTwin result and its exact
  producer, consumer, protocol, report, and retained-evidence identities;
- a fresh real Prob4D provider gate and its confirmatory admission state; and
- semantic reweighting admission.

This prevents a controlled synthetic success from being presented as fresh
physical-provider evidence. It also prevents templates, dry runs, source-panel
controls, or workflow artifacts from incrementing the 36-execution
confirmatory registry.

When physical acquisition progress is nonzero, the status record must bind the
corresponding evidence-status artifact with a lowercase SHA-256 digest. The
explicit physical non-claim must contain the same acquired and validated counts,
so generated prose cannot silently drift from the machine-readable registry.

Version 2 is intentionally limited to the acquisition-stage claim boundary:
`status=pending`, `claim_ready=false`, and independent-execution calibration
pending. Completion, a failed registered endpoint, or a promoted claim requires
a new reviewed schema bound to the completed evidence rather than an in-place
reinterpretation of v2.

## Readiness controls are not empirical evidence

The repository may add or seal acquisition-readiness controls while the
empirical status remains unchanged. Examples include a registered analysis
manifest, method-freeze attestation, software-environment sealing, source-review
records, staged-manifest preflight, action-support admission, identifiability
checks, or baseline-relative fallback guards.

Those controls can make future collection and analysis safer, more reproducible,
or more fail closed. They do not by themselves establish a physical prediction
result, independent-execution calibration, a fresh real-provider benefit, or a
confirmatory method admission. In particular, they must not increment acquired
or validated execution counters and must not change `claim_ready=false`.

This omission is deliberate: version 2 summarizes empirical and admission states,
not every piece of prospective infrastructure present in the development tree.
A readiness control enters the status only when a reviewed schema defines its
claim-relevant state and binds the corresponding evidence identity.

## Automated checks

`ci/three_repository_status.py` fails closed when:

- JSON contains duplicate keys, non-finite values, or a symlinked status file;
- the schema, closed field sets, status identity, claim boundary, or decisive
  next milestone changes unexpectedly;
- Causal4D package metadata and `causal4d.__version__` disagree with the status
  contract;
- the recorded Bayesian-PhysTwin range differs from
  `BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE`;
- an installed Bayesian-PhysTwin or Prob4D wheel lies outside its recorded
  development range;
- either controlled evidence identity differs from its reviewed revision,
  protocol, or report binding;
- physical execution accounting violates
  `validated <= acquired <= specified`;
- nonzero physical progress lacks a bound evidence-status digest;
- physical progress and its explicit non-claim disagree;
- controlled synthetic Prob4D evidence is conflated with a fresh real-provider
  gate;
- Prob4D or semantic components are admitted into the confirmatory method under
  the acquisition-stage schema; or
- explicit non-claims are removed.

The same module validates the explicit v1-to-v2 evidence split and monotone v2
progress updates. Snapshot dates and execution counts may not move backwards,
and package roles may not change silently. V2 cannot promote the claim boundary;
the completed physical registry and independent-execution calibration must be
bound by the successor schema.

## Generated status

Regenerate the human-readable view with:

```bash
python ci/render_project_status.py
```

Check that the committed view is byte-current with:

```bash
python ci/render_project_status.py --check
```

The core regression test enforces the same byte equality. The renderer first
validates the JSON contract, so an invalid status cannot be published as
apparently authoritative Markdown.

## Installed-wheel path

The private installed-wheel golden path defaults to
`ci/project_status_v2.json`. It records the status schema, status ID, canonical
SHA-256 digest, package versions, next milestone, claim status, empirical
states, and physical acquisition counters in its JSON summary.

The runner resolves the record from
`$GITHUB_WORKSPACE/causal4d/ci/project_status_v2.json` in GitHub Actions and from
`ci/project_status_v2.json` when launched at the repository root. No source
checkout is imported; package-version checks run against the three built and
installed wheels.

## Update policy

A package release may update its version or compatibility range without changing
the empirical status. Acquisition counters may advance only from a
hash-verified evidence registry, must remain monotone, and must update the
matching non-claim. A scientific claim or confirmatory-method admission requires
a successor schema and the corresponding claim-bearing evidence.

The project-status contract does not replace exact commit pins used by frozen
milestones or method-freeze manifests. It is the ordinary-development
compatibility and evidence summary; frozen experiments continue to bind exact
revisions, artifacts, and checksums.
