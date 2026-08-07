# Prospective Latent-Contact V2

## Scope

`causal4d.latent_contact_v2` is an additive development path for the three
remaining prospective-v2 items:

1. normalized, covariance-aware contact likelihoods;
2. support-aware parameter/contact uncertainty with exact mixture intervals; and
3. contact-patch/effect inference rather than an exact-node-only state.

It does not change the registered latent-contact estimator, the frozen exact-node
gate, the 18-session/36-execution protocol, or any retained result. Promotion
requires source-frozen configuration and a fresh untouched evaluation panel.

## 1. Linear covariance-aware evidence

`LinearContactObservationGroup` represents a sparse linear observation operator
on a rollout prefix. This permits both positions and dynamic increments under one
closed contract:

```text
y_r = sum_k a_k x[t_k, n_k, c_k].
```

Each group carries the full covariance of the resulting vector and a robust
Student-t nominal/outlier mixture. The normalization term, including the
log-determinant of the declared covariance, is retained. Consequently a component
cannot improve its likelihood merely by reporting a larger covariance.

Endpoint frame zero may appear only in a zero-sum contrast. The standard dense
builder therefore includes the endpoint-to-first-response increment while still
forbidding direct reuse of the endpoint as new O-plus evidence.

The convenience constructor supports:

- frame-level position groups;
- endpoint-inclusive adjacent differences;
- cross-coordinate correlation;
- exchangeable within-frame node correlation;
- physical adjacent-frame correlation for difference covariance; and
- masks that are frozen before evaluating a component.

More detailed Prob4D feeders can construct groups directly and supply their full
metric covariance.

## 2. Topology and multiplicity normalization

`ContactObservationEvidenceV2` applies two independent power controls:

- contributor multiplicity: reusing the same observation contributor in several
  groups divides the affected group power; and
- dimension normalization: a group log score is divided by
  `coordinate_count ** dimension_normalization_power`.

The default power of one treats a group as an average proper log score. Adding
more observed nodes or coordinates therefore does not automatically create a
stronger posterior merely because a topology or sensor has more scalar entries.
The power is part of the evidence identity and must be frozen prospectively.

## 3. Sparse contact patches

`SparseContactPatch` is a channel-wise simplex over material nodes. Every command
channel retains unit mass, and expanding a patch into a simulator action is
checked to conserve the complete commanded force at every frame.

`GraphContactPatchModelV2` generates connected local patches by combining:

- a nominal or graph-local shifted center;
- a frozen spread value distributing force over one-hop neighbors;
- gain, delay, and rotation support inherited from the source-fitted contact
  prior.

The patch weights explicitly absorb the old scalar slip/spread mechanism. The v2
state therefore does not simultaneously apply `WorldCondition.contact_spread`,
which would count the same effect twice.

Patch support may be deterministically truncated. The retained prior mass is
recorded before renormalization and is subject to an explicit admission policy.

## 4. Parameter and patch support admission

`ContactV2SupportPolicy` uses the existing deterministic
`ParameterSupportReduction` contract. The default is a weighted coreset, which
assigns every source posterior cell to a selected medoid and can represent full
source probability mass rather than silently discarding the tail.

Admission can bind:

- minimum represented parameter probability mass;
- minimum retained patch prior mass;
- maximum parameter mean error; and
- maximum parameter covariance error.

The bank builder evaluates these conditions before simulator execution and fails
closed with `ContactV2SupportRejectedError`. Generic selection additionally
preserves the caller-provided baseline object by identity on rejection.

## 5. Exact finite-mixture uncertainty

The predictive mean and variance use the law of total variance:

```text
Var(Y) = E[Var(Y | patch, theta)] + Var(E[Y | patch, theta]).
```

Marginal interval endpoints are obtained by deterministic inversion of the
conditional Gaussian-mixture CDF:

```text
sum_k w_k Phi((q - mu_k) / sigma_k) = alpha.
```

This replaces empirical component quantiles plus an external Gaussian margin.
It remains valid for separated contact modes and does not place a Gaussian
interval around a low-density gap by construction.

## 6. Effect posterior

`ContactEffectPosteriorV2` marginalizes gain, delay, rotation, and physical
parameters and reports:

- posterior mass for each distinct sparse patch;
- expected node traction mass for every command channel;
- channel-wise covariance of the traction simplex; and
- expected patch support size.

Exact-node MAP remains available as a special zero-spread patch, but it is no
longer the sole contact representation. Evaluation should emphasize held-out log
score, trajectory regret, patch mass near truth, and force/effect-field scores.
The registered exact-node gate remains unchanged and cannot be rescued by these
prospective metrics.

## Intended promotion experiment

A clean v2 study should freeze all settings on source topologies and compare on
new seed ranges and untouched topologies:

1. registered summed dense likelihood;
2. normalized linear likelihood with diagonal covariance;
3. normalized full-covariance likelihood;
4. v2 likelihood plus weighted-coreset and retained-support admission; and
5. complete contact-patch/effect inference.

Primary outcomes should be held-out trajectory regret against the physical
baseline, proper posterior log/Brier scores, calibration, and exact fallback
rate. Exact-node accuracy is retained as a secondary compatibility diagnostic.
A negative result is admissible and must not change the frozen estimator.

## Deployment boundary

A v2 candidate is not deployable merely because this module admits its finite
support. A claim-bearing pipeline must still require, independently:

- Prob4D provider competence when Prob4D evidence is used;
- BayesianPhysTwin accept/fallback;
- intervention and query identifiability;
- source-frozen action-support admission;
- the baseline-relative Causal4D regret certificate; and
- held-out query calibration.
