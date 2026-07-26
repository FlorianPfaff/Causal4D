# Continuous integration

Causal4D separates its lightweight contracts from private-provider and
hardware integrations. This keeps a base installation testable while
still detecting drift at the Bayesian-PhysTwin boundary.

## Required pull-request checks

`CI and release` runs the following independent jobs:

- **Core-only** installs `.[dev]` without Bayesian-PhysTwin, OpenCV,
  PyTorch, or Warp and executes the default test suite on Python 3.10,
  3.12, and 3.14.
- **Pinned Bayesian-PhysTwin integration** checks out the exact
  provider-API revision recorded in
  `requirements/ci/bayesian-phystwin-provider-v1.sha` and runs the
  BPT-facing Causal4D tests. The normal `phystwin` extra remains the
  supported `>=0.4,<0.5` development range.
- **Distributions and CLI help** build both wheel and source
  distribution, install each into a clean virtual environment, and
  require every declared console command to render `--help` without
  optional providers.
- **Quality** runs Ruff on the repository, Ruff formatting on changed
  Python files, and mypy on stable provider contracts and CI utilities.
- **Result bundles** produce small deterministic benchmark bundles,
  verify their embedded checksums, archive them, extract them into a
  clean directory, and verify the consumed copies again.
- **Frozen manifest** cross-checks the milestone path and checksum
  inventories, validates all recorded Git/SHA-256 identities, and
  hashes every frozen artifact stored in the checkout.

Formatting is an incremental ratchet: modified Python files must be
Ruff-formatted without forcing unrelated historical files into the
same change.

## Private Bayesian-PhysTwin access

Bayesian-PhysTwin is private. Add a least-privilege repository secret
named `BPT_READ_TOKEN` with read access to
`FlorianPfaff/Bayesian-PhysTwin`. Pinned, scheduled-main, BPT-vision,
and Warp jobs report an explicit skipped-coverage warning when the
secret is absent; they do not pretend provider compatibility was
tested.

## Scheduled and optional checks

`Bayesian-PhysTwin provider compatibility` is the scheduled and
manually dispatchable cross-repository check against BPT `main`. The
ordinary CI job remains locked to the exact provider-API revision in
the dedicated CI pin.

`Optional integrations` keeps large or platform-specific dependencies
out of the core job:

- hosted CPU vision/OpenCV tests run weekly;
- BPT OpenCV collection and Warp CPU tests run separately when the
  private token is configured;
- the CUDA/Warp job is manual and targets a self-hosted runner carrying
  the labels `self-hosted`, `linux`, `x64`, and `gpu`.

The manual GPU job additionally requires the same `BPT_READ_TOKEN`
secret.
