# Exact pull-request head validation

GitHub App or automation-authored pull-request events do not always enqueue the
repository's ordinary `pull_request` workflows. Causal4D therefore has one
permanent, read-mostly validation workflow for an explicitly selected immutable
same-repository pull-request head:

```text
.github/workflows/exact-head-validation.yml
```

It is an engineering validation path. It does not access physical data, inspect
target outcomes, mutate registered evidence, or authorize acquisition.

## Requesting validation

A maintainer can dispatch **Exact pull-request head validation** with a pull
request number. For automation that cannot dispatch workflows, place this exact
marker once in the pull-request body:

```html
<!-- exact-head-validation: queued -->
```

The hourly scheduled run selects at most one queued pull request, oldest first.
Only open pull requests whose head and base are both in this repository and whose
base branch is `main` are admissible. Fork heads are rejected.

The marker is one-shot. The final job replaces it with a completion marker bound
to the exact head SHA, result, and Actions run URL. A new head requires a new
queue marker; an earlier successful result must not be interpreted as evidence
for later commits.

## Validation boundary

The selected head SHA and base SHA are resolved from GitHub before execution.
Every validation job checks out the exact head with persisted credentials
disabled, records both identities, and re-reads the pull request after testing.
A head movement during the run makes the validation fail.

The workflow runs:

- the complete default suite on Python 3.10, 3.12, and 3.14;
- syntax-warning rejection and byte compilation;
- Ruff lint and incremental formatting checks;
- MyPy over the stable contracts and CI utilities;
- wheel and source-distribution builds with Twine validation; and
- the exact pinned BayesianPhysTwin installed-wheel integration.

The final `Causal4DExactHeadValidation` JSON artifact records the pull request,
head SHA, base SHA, component job results, whether the head remained current,
and the workflow run URL. The pull-request comment is a human-readable pointer;
the uploaded JSON and Actions logs are the validation evidence.

## Security and claim boundary

The schedule uses the workflow from the default branch, never code-defined
workflow permissions from the pull-request head. It accepts only
same-repository heads, checks out without persisted credentials, and supplies
the write-capable token only to narrowly scoped GitHub API steps. Test processes
do not receive that token.

Passing establishes software validation for the named bytes. It is not physical
experiment evidence, empirical model evidence, independent scientific review, or
a substitute for the registered pre-acquisition readiness and method-freeze
gates.
