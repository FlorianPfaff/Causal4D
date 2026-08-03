# Operator identity registry

The confirmatory acquisition workflow uses a sealed operator roster so that approval independence is checked at the **person** level rather than by comparing free-form display strings.

This is an operational-provenance control. It does not change the estimator, the 36-execution schedule, the registered analysis, the exclusion policy, or any scientific threshold.

## Identity model

The canonical artifact is stored at:

```text
preacquisition/operator_registry.json
```

Each entry contains only:

- a stable project-local `operator_id`;
- an `active` flag;
- registered roles; and
- `person_identity_sha256`, a privacy-preserving stable person digest.

Raw email addresses, account names, personnel numbers, and the HMAC key must not be written to the dataset. The registered digest method is:

```text
hmac-sha256-domain-separated-v1
```

A suitable institutional procedure is:

```text
HMAC-SHA256(
  institution-held secret,
  b"causal4d-operator-v1\0" + stable_institutional_principal
)
```

The secret and the stable institutional principal remain outside the repository and acquisition dataset. The same person must always produce the same digest, even when that person uses a different display name or project-local alias. Two registry entries may never share a person digest.

## Roles

The supported roles are:

- `freezer`: may seal the method freeze;
- `independent_verifier`: may attest the method freeze and counts as an independent software-lock approver;
- `gate_approver`: may approve operational readiness gates; and
- `software_environment_approver`: may independently approve the software-environment lock.

Every gate approver must have `gate_approver`. The software-environment approver must additionally have either `independent_verifier` or `software_environment_approver`, and must have a different person digest from the method freezer.

The method freezer and independent verifier must resolve to distinct person digests. Different `operator_id` strings do not establish independence.

## Lifecycle

Scaffold the registered template after creating the dataset:

```bash
causal4d protocol readiness scaffold-operator-registry \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1
```

Edit only the `operators` array in:

```text
preacquisition/operator_registry.template.json
```

Keep the template artifact kind, `status: "template"`, registered protocol and amendment digests, `target_outcomes_used: false`, and all seal fields unchanged. Roles and operators are canonicalized during sealing.

Seal the roster exactly once:

```bash
causal4d protocol readiness seal-operator-registry \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/preacquisition/operator_registry.template.json \
  --sealed-by freezer.primary
```

The seal fails closed when:

- the canonical registry already exists;
- method-freeze or freeze-attestation evidence already exists;
- an operational gate has already been approved;
- confirmatory execution or session manifests already exist;
- a person digest or operator ID is duplicated;
- an operator is unknown, inactive, or missing a required role;
- an unsupported or noncanonical field is present; or
- target outcomes entered the identity artifact.

The canonical registry is published atomically and cannot be replaced or resealed.

## Governed approvals

Use registered operator IDs, not names:

```bash
causal4d protocol freeze seal \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  --frozen-by freezer.primary

causal4d protocol freeze attest \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  configs/causal4d/sloth_multi_action_v1.json \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1/method_freeze_validation.json \
  --verified-by verifier.independent

causal4d protocol readiness seal-gate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  support_registration_passed \
  --approved-by verifier.independent
```

The registry must predate every governed approval. A postdated registry, role change, alias collision, unknown identity, or person-level independence failure blocks readiness and claim status.

## Status-time revalidation

Approval checks are not limited to the commands that create an artifact. Every readiness and claim-status evaluation reopens and validates the complete governed chain:

- method freezer and independent freeze verifier;
- timebase-calibration approver;
- contact-registration approver; and
- every operational readiness-gate approver.

The derived `operator_identity_bindings` prerequisite records the canonical registry digest, the resolved person digests, approval-role bindings, and SHA-256 digests of all governed source artifacts. Missing or template evidence remains valid-but-incomplete. A completed artifact with an unknown, inactive, role-incompatible, postdated, aliased, or non-independent identity fails closed.

This status-time validation prevents manually edited artifacts or legacy free-form identifiers from bypassing the creation-time checks.

## Evidence identity

The registry carries a canonical `artifact_sha256`. Pre-acquisition readiness and real-evidence status include that digest as a prerequisite. Portable readiness identity removes workstation-local paths but retains the registry digest and the governed source digests, so relocating an unchanged dataset preserves evidence identity while changing the roster or any approval artifact does not.

Registry templates, dry runs, or scaffold output never count as physical executions and never change the valid evidence boundary from `0/36 acquired` until genuine acquisition begins.
