# Continuous integration

Causal4D separates its lightweight contracts from public cross-repository and
hardware integrations. This keeps a base installation testable while still
detecting drift at the Prob4D and Bayesian-PhysTwin boundaries.

## Required pull-request checks

`CI and release` runs the following independent jobs:

- **Core-only** installs `.[dev]` without Bayesian-PhysTwin, OpenCV, PyTorch, or
  Warp and executes the default test suite on Python 3.10, 3.12, and 3.14.
- **Pinned Bayesian-PhysTwin integration** checks out the exact provider-API
  revision recorded in `requirements/ci/bayesian-phystwin-provider-v1.sha` and
  runs the BPT-facing Causal4D tests. The normal `phystwin` extra remains the
  supported `>=0.4,<0.5` development range.
- **Distributions and CLI help** build both wheel and source distribution,
  install each into a clean virtual environment, and require every declared
  console command to render `--help` without optional providers.
- **Quality** runs Ruff on the repository, Ruff formatting on changed Python
  files, and mypy on stable provider contracts and CI utilities.
- **Result bundles** produce small deterministic benchmark bundles, verify their
  embedded checksums, archive them, extract them into a clean directory, and
  verify the consumed copies again.
- **Frozen manifest** cross-checks the milestone path and checksum inventories,
  validates all recorded Git/SHA-256 identities, and hashes every frozen
  artifact stored in the checkout.

Formatting is an incremental ratchet: modified Python files must be
Ruff-formatted without forcing unrelated historical files into the same change.

## Public repository integration

Prob4D, BayesianPhysTwin, and Causal4D are public repositories. Required
provider checks therefore use ordinary read-only `actions/checkout` steps and do
not depend on repository secrets or deploy keys. Forked pull requests exercise
the same installed-wheel boundary as organization branches, and release tags
remain blocked unless the pinned public-provider job passes. Optional hardware
workflows retain their separately documented runtime requirements.

## Three-repository installed-wheel golden path

`.github/workflows/bayesian-phystwin-provider-compatibility.yml` is the terminal
Prob4D -> BayesianPhysTwin -> Causal4D compatibility check. It runs on relevant
pull requests and pushes, can be dispatched with explicit BPT and Prob4D
revisions, and runs weekly against both public repositories' `main` branches.
All events execute the full path; an unavailable checkout or contract failure is
a failing check rather than a credential-dependent skip.

The installed-wheel job records the exact clean revision of every checkout,
builds one wheel for each repository, records each wheel's SHA-256 identity, and
installs only those wheels plus runtime dependencies into a fresh virtual
environment. The runner modules are copied outside every source checkout,
`PYTHONPATH` is unset, and editable or checkout-resolved imports are rejected.

The job then executes two deliberately separate paths.

### Compatibility and frozen-reproduction path

The compatibility path:

1. reconstructs the deterministic Prob4D joint-gauge observation fixture and
   checks its fixed content address;
2. lets BayesianPhysTwin independently validate causal lineage, adapt the
   observation to a gauge-aware batch, and execute a deterministic update;
3. asserts that conditional local covariance is passed unchanged while the
   shared low-rank gauge root remains an explicit nuisance, preventing
   covariance double counting;
4. lets Causal4D independently recompute observation lineage, bind a
   content-addressed `TwinBelief`, validate separate scientific-provider-v1 and
   replay-provider-v2 manifests, and execute a CPU-only counterfactual rollout
   through a typed fake provider-v2 implementation;
5. verifies staged BPT truncation, Causal4D support reduction, composed posterior
   mass, rejection of inconsistent composition, and exclusion of exact-zero
   support cells from replay;
6. writes and verifies a clean `RunManifestV2` binding all three Git revisions,
   installed package versions, the observation input, `TwinBelief`, rollout
   bank, provider manifest, and method/protocol/split/baseline identifiers;
7. exercises expected rejections for payload-digest tampering, an opened future
   payload, a source window crossing the causal cutoff, metric-anchor lineage
   mismatch, insufficient retained gauge covariance, contradictory stream
   versions, fixed-lag covariance falsely labelled as strict v2, incomplete
   evidence manifests, and dirty promotable evidence; and
8. validates immutable replay request IDs, frame provenance, position and
   velocity histories, and complete resumable-cache reconstruction.

### Strict claim-bearing provider-v2 path

The prospective path constructs a fixture-only, calibrated provider-v2 artifact
and exercises every strict admission boundary before a physical-state update is
formed:

1. Prob4D's strict loader requires an explicit causal stream contract v2, the
   sequential joint gauge tree, canonical shared factors, both calibration IDs,
   and independently verified runtime provenance;
2. BayesianPhysTwin's claim-bearing adapter validates the producer statement,
   full cross-window covariance, calibration completeness, and fallback policy
   before constructing the innovation, then executes the deterministic guarded
   update twice and requires exact parity;
3. Causal4D independently recomputes the provider manifest and observation
   content address, requires the explicit stream-v2 contract and joint
   covariance, binds the same calibration identities, and refuses lineage with
   incomplete alignment calibration or any covariance fallback; and
4. all three consumers must reject provider-manifest tampering, calibration-ID
   drift, incomplete gauge calibration, recorded covariance fallback,
   permission for pointwise fallback, and a stream version that is merely
   inferred rather than explicitly declared.

The fixture uses synthetic calibration identities and is permanently labelled
`claim_evidence=false`. Passing this path establishes installed-wheel contract
compatibility and fail-closed admission, not empirical observation quality,
calibration, or physical-prediction benefit.

The workflow also runs the producer, provider, lineage, belief, manifest, cache,
and backend test files with `--import-mode=importlib` against the installed
wheels.

The uploaded `three-repository-installed-wheel-golden-path` artifact contains:

- `three-repository-summary.json` and the compatibility-path log;
- `three-repository-provider-v2-summary.json` and the strict-path log; and
- `three-repository-wheel-sha256.txt`, binding the three built wheel bytes.

The summaries record package origins and repository revisions, observation and
provider-manifest identities, the deterministic BPT update digest, the bound
`TwinBelief` ID, rollout digest and shape, staged posterior-mass accounting, the
evidence fingerprint, strict covariance-calibration semantics, and every
expected rejection.

## Scheduled and optional checks

`Optional integrations` keeps large or platform-specific dependencies out of the
core job:

- hosted CPU vision/OpenCV tests run weekly;
- BPT OpenCV collection and Warp CPU tests run separately when the private SSH
  credential is configured;
- the CUDA/Warp job is manual and targets a self-hosted runner carrying the
  labels `self-hosted`, `linux`, `x64`, and `gpu`.

The manual GPU job additionally requires `BPT_READ_SSH_KEY`.

## Reproducing the wheel boundary locally

The workflow is the canonical executable specification. A local equivalent
requires checkouts of all three public repositories:

```bash
python -m build --wheel --outdir /tmp/three-repo-wheels Prob4D
python -m build --wheel --outdir /tmp/three-repo-wheels BayesianPhysTwin
python -m build --wheel --outdir /tmp/three-repo-wheels Causal4D
python -m venv /tmp/three-repo-venv
/tmp/three-repo-venv/bin/python -m pip install /tmp/three-repo-wheels/*.whl
```

Copy all `Causal4D/ci/three_repository_*.py` modules to one directory outside
all three checkouts before running the main compatibility module and the strict
`three_repository_provider_v2_attestation.py` module. Unset `PYTHONPATH`, pass
the Prob4D fixture, all three exact Git revisions, and each checkout root. The
runners deliberately fail if they or an installed import resolve under any
checkout.
