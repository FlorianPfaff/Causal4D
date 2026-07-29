# Continuous integration

Causal4D separates its lightweight contracts from private-provider and
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

## Private repository access

Configure the least-privilege repository secret `BPT_READ_TOKEN` with read
access to both private repositories:

- `FlorianPfaff/Bayesian-PhysTwin`;
- `FlorianPfaff/Prob4D`.

The historical secret name is retained so existing repository configuration
does not need to change. Trusted same-repository pull requests, pushes,
scheduled runs, and manual dispatches now **fail** when the secret is absent.
The three-repository workflow is not allowed to report success after skipping
its checkouts, wheel builds, and compatibility tests.
Tag releases also fail in the main CI workflow when the credential is absent,
so a skipped pinned-provider job cannot authorize release publication.

GitHub does not expose repository secrets to pull requests from external forks.
Those events receive a dedicated `External PR cannot access private providers`
result explaining that private coverage is unavailable. A maintainer must
reproduce the proposed head on a trusted same-repository branch and obtain a
passing installed-wheel golden path before merge.

## Three-repository installed-wheel golden path

`.github/workflows/bayesian-phystwin-provider-compatibility.yml` is the terminal
Prob4D -> Bayesian-PhysTwin -> Causal4D compatibility check. It runs on relevant
same-repository pull requests and pushes, can be dispatched with explicit BPT
and Prob4D revisions, and runs weekly against both private repositories' `main`
branches.

The workflow begins with a separate credential gate. For trusted events, a
missing `BPT_READ_TOKEN` is a hard configuration failure and prevents a false
green result. Only an external-fork pull request can use the explicitly labelled
unavailable path, because GitHub withholds repository secrets by design.

The installed-wheel job then:

1. records the exact clean revision of every checkout;
2. builds one wheel for each repository and installs only those wheels plus
   runtime dependencies into a fresh virtual environment;
3. copies the runner modules outside every source checkout, unsets `PYTHONPATH`,
   and rejects editable or checkout-resolved imports;
4. reconstructs the deterministic Prob4D joint-gauge observation fixture and
   checks its fixed content address;
5. lets Bayesian-PhysTwin independently validate causal lineage, adapt the
   observation to a gauge-aware batch, and execute a deterministic update;
6. asserts that conditional local covariance is passed unchanged while the
   shared low-rank gauge root remains an explicit nuisance, preventing
   covariance double counting;
7. lets Causal4D independently recompute observation lineage, bind a
   content-addressed `TwinBelief`, validate separate scientific-provider-v1 and
   replay-provider-v2 manifests, and execute a CPU-only counterfactual rollout
   through a typed fake provider-v2 implementation;
8. verifies staged BPT truncation, Causal4D support reduction, composed posterior
   mass, rejection of inconsistent composition, and exclusion of exact-zero
   support cells from replay;
9. writes and verifies a clean `RunManifestV2` binding all three Git revisions,
   installed package versions, the observation input, `TwinBelief`, rollout
   bank, provider manifest, and method/protocol/split/baseline identifiers;
10. exercises expected rejections for payload-digest tampering, an opened future
    payload, a source window crossing the causal cutoff, metric-anchor lineage
    mismatch, insufficient retained gauge covariance, contradictory stream
    versions, fixed-lag covariance falsely labelled as strict v2, incomplete
    evidence manifests, and dirty promotable evidence;
11. validates immutable replay request IDs, frame provenance, position and
    velocity histories, and complete resumable-cache reconstruction; and
12. runs the existing producer, provider, lineage, belief, manifest, cache, and
    backend tests with `--import-mode=importlib` against the installed wheels.

The workflow uploads `three-repository-summary.json` and the complete runner log
as the `three-repository-installed-wheel-golden-path` artifact. The JSON summary
records package origins and repository revisions, the observation artifact ID,
the BPT update digest, the bound `TwinBelief` ID, rollout digest and shape,
provider compatibility, staged posterior-mass accounting, the evidence
fingerprint, and every expected rejection.

## Scheduled and optional checks

`Optional integrations` keeps large or platform-specific dependencies out of the
core job:

- hosted CPU vision/OpenCV tests run weekly;
- BPT OpenCV collection and Warp CPU tests run separately when the private token
  is configured;
- the CUDA/Warp job is manual and targets a self-hosted runner carrying the
  labels `self-hosted`, `linux`, `x64`, and `gpu`.

The manual GPU job additionally requires the same `BPT_READ_TOKEN` secret.

## Reproducing the wheel boundary locally

The workflow is the canonical executable specification because it can read the
two private repositories. A local equivalent requires checkouts of all three
repositories and a token is not needed once they are available:

```bash
python -m build --wheel --outdir /tmp/three-repo-wheels Prob4D
python -m build --wheel --outdir /tmp/three-repo-wheels Bayesian-PhysTwin
python -m build --wheel --outdir /tmp/three-repo-wheels Causal4D
python -m venv /tmp/three-repo-venv
/tmp/three-repo-venv/bin/python -m pip install /tmp/three-repo-wheels/*.whl
```

Copy all `Causal4D/ci/three_repository_*.py` modules to one directory outside
all three checkouts before running the main module, unset `PYTHONPATH`, and pass
the Prob4D fixture, all three exact Git revisions, and each checkout root. The
runner deliberately fails if it or an installed import resolves under any
checkout.
