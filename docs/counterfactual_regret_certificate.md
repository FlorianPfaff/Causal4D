# Baseline-Relative Counterfactual Regret Certificate

## Purpose

A Causal4D candidate can be numerically admissible, identifiable, and inside the
source action-support envelope while still predicting worse than its declared
physical baseline. The controlled benchmark contains such a regime: a hybrid
correction can improve average performance yet incur positive regret under
combined model mismatch and shifted contact.

`causal4d.counterfactual_regret` adds a prospective, source-calibrated decision
layer that answers a narrower deployment question:

> Given this target-safe diagnostic vector, is the candidate supported by
> independent source sessions whose candidate-versus-baseline regret is within
> the frozen policy?

The certificate is protocol- and endpoint-specific and binds the baseline role,
candidate role, metric, feature schema, source sessions, source losses, support
rule, local
neighborhood size, regret tolerances, and prerequisite gate inventory. Rejection
returns the caller-provided baseline object by identity.

This module does not alter the frozen estimator or the registered 36-execution
physical protocol.

## Information boundary

A `CounterfactualRegretSourceCase` deliberately contains held-out **source**
losses for the baseline and candidate. Those losses are used to calibrate the
selective policy.

A `CounterfactualRegretTarget` contains no target loss or target continuation. It
contains only:

- the protocol, endpoint, target case, and independent target-session identity;
- the declared baseline/candidate roles;
- immutable baseline and candidate artifact identities;
- a fixed finite feature vector computed without the target continuation; and
- independently content-addressed prerequisite decisions.

The resulting decision records:

```text
target_future_observations_read = 0
target_future_outcomes_used = false
```

Changing or withholding a target future therefore cannot change the decision,
because no target-future field exists at the target boundary. The target case and
session must both be disjoint from every calibration source case and session.

## Recommended target-safe features

`CounterfactualRegretFeatures` is intentionally provider-neutral. A protocol
must freeze the names and construction of its feature vector before target
access. Recommended Causal4D features include:

- intervention effective-rank fraction and minimum conditional-information
  eigenvalue;
- residualized intervention-response fraction;
- maximum intervention/nuisance subspace cosine;
- query null-response fraction;
- factual posterior entropy or effective sample-size fraction;
- nominal-contact posterior mass;
- prefix-only robust log score and nominal-component responsibilities;
- discrepancy-to-physical-response energy ratio;
- represented physical support mass;
- action-support distance and supported posterior mass; and
- upstream guarded-update magnitude relative to prior uncertainty.

Features must use the same names, units, sign conventions, and causal cutoff in
source and target cases. Arbitrary feature selection after target outcomes are
opened is inadmissible.

A deterministic mapping constructor is available:

```python
features = CounterfactualRegretFeatures.from_mapping(
    {
        "action_support_distance": action_decision.nearest_source_distance,
        "action_supported_mass": action_decision.supported_component_mass,
        "query_null_response_fraction": query_null_fraction,
        "represented_physical_mass": represented_mass,
    }
)
```

Mapping keys are sorted before content addressing, so caller dictionary order
does not change the feature identity.

## Session-aware source calibration

Source executions can share grasp, reset, registration, object state, or other
nuisance conditions. The certificate therefore does not treat source cases as
independent merely because they have different case identifiers.

Each source case declares a `session_id` and exact `protocol_id`. Calibration:

1. requires at least three independent source sessions;
2. computes candidate relative improvement for each source case;
3. averages improvement within each source session for global policy checks;
4. weights the resulting sessions equally; and
5. computes source-support distance with leave-one-session-out neighbors.

For a lower-is-better metric, relative improvement is

```text
r = (loss_baseline - loss_candidate) / max(abs(loss_baseline), 1e-12).
```

For a higher-is-better metric, the numerator is reversed. Positive values favor
the candidate; negative values are relative regret.

Global source admission records:

- equal-session mean relative improvement;
- equal-session win fraction;
- fraction of sessions exceeding the frozen harmful-regret tolerance;
- worst session-level relative regret; and
- a derived `candidate_enabled` decision.

All derived values are independently recomputed by the loader. Changing a
source loss or derived summary and merely recalculating the outer SHA-256 does
not produce an admissible certificate.

## Source-support geometry

Let `f_i` be the fixed feature vector for source case `i`. The certificate uses a
robust, component-wise source scale:

```text
center = median(f_i)
scale = max(1.4826 * MAD, half range, numerical magnitude floor)
```

Distances are root-mean-square standardized distances. For every source case,
the nearest case from a **different session** is found. The largest such
distance, multiplied by the frozen support margin, defines the target support
radius.

This establishes empirical source support only. It is not a guarantee outside
the source feature and mechanism distribution.

## Local selective-regret decision

For one target feature vector, the decision:

1. computes its distance to every source case;
2. retains only the nearest case from each source session;
3. selects the frozen number of nearest independent sessions;
4. reports their source relative improvements; and
5. applies the frozen local regret policy.

The candidate is rejected when any of the following occurs:

- an upstream prerequisite decision rejected;
- global source regret policy did not enable the candidate;
- target features lie outside source support;
- local mean source improvement is too small;
- local source win fraction is too small;
- local harmful-session fraction is too large; or
- local worst relative regret exceeds its bound.

The local neighborhood is selected using target-safe feature distance only.
Source outcomes do not influence which source case is chosen within a session.

## Prerequisite composition

The certificate stores an exact prerequisite-name inventory. A target must
provide exactly that inventory, with each decision bound by a SHA-256 identity.
A typical counterfactual protocol should require:

```text
prob4d_provider_acceptance          (when Prob4D is used)
bayesian_phystwin_acceptance
intervention_identifiability
action_support
```

A downstream regret pass cannot override an upstream rejection. Conversely, an
upstream pass does not imply that the Causal4D candidate has acceptable regret.
The gates answer different questions and remain conjunctive.

## Exact fallback

```python
selection = select_counterfactual_regret_candidate(
    certificate,
    target,
    baseline=nominal_z_physical_posterior,
    candidate=causal4d_candidate,
    baseline_artifact_id=nominal_z_physical_posterior.artifact_id,
    candidate_artifact_id=candidate_id,
)

if not selection.decision.accepted:
    assert selection.deployed is nominal_z_physical_posterior
```

The generic selector requires explicit baseline and candidate identities and
checks them against the target contract. Rejection returns the original baseline
object rather than a reconstructed approximation.

## Source fitting

```python
certificate = fit_counterfactual_regret_certificate(
    source_cases,
    required_prerequisite_names=(
        "bayesian_phystwin_acceptance",
        "intervention_identifiability",
        "action_support",
    ),
    local_session_count=3,
    harmful_relative_regret_threshold=0.02,
    minimum_global_mean_relative_improvement=0.0,
    minimum_global_win_fraction=0.5,
    maximum_global_harmful_fraction=1.0 / 3.0,
    maximum_global_worst_relative_regret=0.10,
    minimum_local_mean_relative_improvement=0.0,
    minimum_local_win_fraction=0.5,
    maximum_local_harmful_fraction=1.0 / 3.0,
    maximum_local_worst_relative_regret=0.05,
)

write_counterfactual_regret_certificate(
    "same-grasp-regret-certificate.json",
    certificate,
)
```

The numerical values above are API defaults for development, not promoted
scientific thresholds. Claim-bearing use must freeze endpoint-specific values on
an independently declared source split.

Load claim-bearing artifacts only with the independently frozen identity:

```python
certificate = load_claim_bearing_counterfactual_regret_certificate(
    "same-grasp-regret-certificate.json",
    expected_certificate_id=registered_certificate_id,
)
```

Separate certificates are required for factual continuation, same-grasp
transfer, and new-contact transfer. Evidence from one endpoint must not silently
authorize another.

## Required evaluation

A prospective evaluation should use disjoint source-calibration and held-out
target sessions and report:

- baseline loss, candidate loss, and deployed-policy loss;
- candidate acceptance rate;
- accepted-candidate relative regret;
- harmful accepted updates;
- exact-fallback rate;
- global and local source-neighborhood diagnostics;
- results by action, contact topology, support distance, and realization regime;
- predictive coverage and interval width after applying the complete policy; and
- all prerequisite rejection rates.

The calibration target is selective regret relative to the declared baseline,
not synthetic regime classification accuracy.

## Scientific boundary

This is prospective method-development infrastructure. Passing the certificate
shows only that source sessions near the target-safe diagnostic vector met the
frozen baseline-relative regret policy. It does not establish:

- real provider competence;
- causal identifiability by itself;
- calibrated counterfactual coverage;
- object-class generalization;
- individual counterfactual ground truth;
- deployment safety; or
- superiority over the registered physical baseline.

The certificate is not wired into the frozen controlled benchmark or the
registered physical estimator. Promotion requires a separately frozen source
split, endpoint-specific certificate identities, held-out evaluation of the
complete gate stack, and an explicit preacquisition amendment.
