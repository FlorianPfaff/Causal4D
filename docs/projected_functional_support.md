# Task-projected functional-support certification

## Scope

`causal4d.projected_functional_support_v1` is an additive source-only
certificate. It does not change the frozen estimator, registered physical
protocol, target split, threshold, or evidence count.

The existing `functional_support_v1` certificate compares complete rollout-space
behavior. A reduction can nevertheless preserve aggregate trajectory metrics
while changing one task-relevant joint readout. The projected certificate adds
a frozen set of linear readouts over `(frame, node, coordinate)` and checks each
source action/projection pair.

## Quantities

For projection coefficients `a`, component mean trajectory `mu`, independent
conditional variance `d`, and low-rank conditional factors `F`, the component
projection has

```text
mean = a^T mu
variance = sum(a**2 * d) + sum_r (a^T F_r)**2
```

The certificate then applies the law of total variance across the weighted
support components. It reports and gates:

- projected predictive-variance relative error; and
- maximum projected Gaussian-mixture interval-endpoint error.

The latter also detects a projected mean shift because a mean translation moves
both interval endpoints.

## Fail-closed contract

Certification requires:

- a base certificate covering the same actions in the same order;
- base-certificate provenance for every action artifact;
- the complete action-by-projection Cartesian product;
- a source-frozen policy and confidence level;
- digest-shaped source-artifact identities;
- exact consistency between every metric and the aggregate decision; and
- no target outcome, target continuation, or target loss in source metadata.

Projection coefficients and optional low-rank factors are immutable and
content-addressed. A plain `FunctionalSupportActionV1` is equivalent to a
projected action with no low-rank conditional factors.

## Example

```python
from causal4d.projected_functional_support_v1 import (
    FunctionalSupportProjectionV1,
    ProjectedFunctionalSupportActionV1,
    ProjectedFunctionalSupportPolicyV1,
    certify_projected_functional_support_v1,
)

projected_action = ProjectedFunctionalSupportActionV1(
    action=source_action,
    full_component_low_rank_factors_m=full_modes,
    reduced_component_low_rank_factors_m=reduced_modes,
)
projection = FunctionalSupportProjectionV1(
    projection_id="late-endpoint-displacement",
    coefficients=late_endpoint_coefficients,
)
certificate = certify_projected_functional_support_v1(
    (projected_action,),
    (projection,),
    policy=ProjectedFunctionalSupportPolicyV1(
        maximum_projected_variance_relative_error=0.10,
        maximum_projected_interval_endpoint_error_m=0.002,
    ),
    base_certificate=base_functional_support_certificate,
    source_artifact_ids=(source_projection_freeze_sha256,),
)
```

A passing certificate establishes preservation only for the registered source
actions and projections. It is not target-domain accuracy, transfer,
calibration, or physical-experiment evidence.
