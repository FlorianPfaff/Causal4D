# Sensor-Factorized Realized-Intervention Abduction

## Status

This is an opt-in development extension. It does not modify the frozen
`v0.3.0-causal4d-aip` path or promote a new real-data result.

## Motivation

The ordinary factual-abduction path scores the permitted object-response prefix
and returns a posterior over physical particles and realized-intervention
hypotheses. Measured robot motion and force should not be folded into that same
object likelihood or reconstructed from the object response. They are separate
sensor factors:

```text
u_cmd -> measured actuator realization -> contact transmission -> object response
```

For component `k`, the update is

```text
p_k^+ proportional to p_k(object prefix)
                       L_actuator(measured end effector | component k)
                       L_wrench(measured force/wrench | component k).
```

No object observation is accepted by the factorized update API. This prevents a
second use of the object-response prefix after ordinary Causal4D abduction.

## Typed evidence

`ActuatorEvidence` stores synchronized measured end-effector positions, metric
variance, a validity mask, clock identity, provenance, and the last admissible
causal frame. `ContactWrenchEvidence` provides the analogous contract for force
or wrench quantities with explicit column names. Both artifacts:

- are immutable after construction;
- bind the protocol, case, and observed-action identities;
- have deterministic SHA-256 artifact identities;
- serialize to non-pickled NPZ files;
- reject evidence extending beyond the factual-abduction prefix;
- require provenance explaining the independently measured sensor stream.

When actuator and wrench evidence are combined, they must name the same trusted
clock. Evidence from another protocol, case, or observed action is rejected
before any likelihood is evaluated.

Use `save_independent_sensor_evidence` and
`load_independent_sensor_evidence` for checksummed round trips.

## Robust independent-sensor factors

`reweight_factual_intervention_with_independent_sensors` starts from an existing
`FactualIntervention`. Component predictions must be supplied on the evidence
sample grid. The update uses diagonal Student-t factors with:

- observation and optional component-prediction variance;
- separate actuator and wrench likelihood powers;
- capped effective scalar sample counts;
- the variance-normalization term, so broad predictions are not rewarded;
- explicit metadata that the object likelihood was not reused.

Absent evidence, all-invalid evidence, zero-powered evidence, or a
component-invariant likelihood returns the original `FactualIntervention`
object exactly. This preserves its artifact identity and weights bit-for-bit.

A minimal actuator update is:

```python
from causal4d import (
    ActuatorEvidence,
    predict_affine_actuator_realizations,
    reweight_factual_intervention_with_independent_sensors,
)

actuator = ActuatorEvidence(
    protocol_id=factual.context.protocol_id,
    case_id=factual.context.case_id,
    observed_action_id=factual.context.u_obs.action_id,
    stream_id="measured_end_effector",
    clock_id="robot_monotonic",
    provenance="robot encoder independent of RGB-D reconstruction",
    sample_times_s=times,
    positions_m=measured_positions,
    variance_m2=encoder_variance,
    evidence_frame_stop=factual.evidence_frame_stop,
)

predicted = predict_affine_actuator_realizations(
    commanded_positions,
    factual.phi_names,
    factual.phi,
    rotation_axis=(0.0, 1.0, 0.0),
)

updated = reweight_factual_intervention_with_independent_sensors(
    factual,
    actuator_evidence=actuator,
    predicted_actuator_positions_m=predicted,
)
```

`predict_affine_actuator_realizations` implements the currently typed gain,
integer delay, and controller-frame rotation variables. Gain and rotation act on
displacement from the first command pose. Nominal gain, zero delay, and zero
rotation reproduce the commanded trajectory exactly.

## Distributed contact traction and wrench

`graph_traction_field` lifts low-rank coefficient forces through a graph basis.
`integrate_contact_wrench` then computes resultant force and torque about a
specified origin. These utilities support a sparse distributed traction field
without requiring the inference API to assume one fixed contact node. A
single-node basis remains an exact special case.

The PhysTwin adapter remains responsible for generating component-wise actuator
and wrench predictions without target-future object observations. The
factorized module does not splice trajectories or inject corrections into the
simulator state.

## Causal-sufficiency falsification

`assess_command_residual_sufficiency` asks whether commanded-action identity
still predicts held-out residuals after conditioning on the supplied realized-
intervention features. It compares cross-fitted ridge predictors:

```text
residual ~ realized-intervention features
residual ~ realized-intervention features + command identity
```

The comparison is grouped by independent execution or session and uses a
permutation diagnostic. A significant command-identity gain indicates that the
chosen realization variables are incomplete. Failure to detect a gain is not a
proof of conditional independence.

## Integration boundary

A real Bayesian-PhysTwin integration should:

1. retain measured actuator and force streams on a trusted common clock;
2. generate component predictions from command, physical particle, contact path,
   and traction hypotheses without reading future object outcomes;
3. perform ordinary object-prefix abduction once;
4. apply the independent sensor factors;
5. carry the updated joint posterior into the existing counterfactual operator;
6. evaluate the command-residual sufficiency test on untouched executions.

The locked 36-execution protocol and frozen milestones remain unchanged. Any
promotion requires source-only factor settings, independent-session evaluation,
held-out action/contact prediction, and the existing calibration gates.
