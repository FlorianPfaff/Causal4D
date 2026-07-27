# Same-Patch Counterfactual Semantics

## Motivation

The frozen counterfactual operator treats `same_grasp` as exact reuse of the
complete factual contact variable `kappa`, including attachment shifts and slip.
That remains the default for milestone compatibility.

The Causal4D formulation also permits a narrower intervention: retain the
factual contact-patch posterior while allowing slip to change under the new
action. This is now available as an explicit opt-in query semantic.

## Query

Set the following metadata on a `CounterfactualQuery` whose `contact_policy` is
`same_grasp`:

```python
query = CounterfactualQuery(
    ...,
    contact_policy="same_grasp",
    metadata={"same_grasp_semantics": "evolve_slip"},
)
```

Available values are:

- `fixed_kappa`: preserve the complete factual `(theta, phi, patch, slip)` joint
  posterior. This is the unchanged default.
- `evolve_slip`: preserve `(theta, phi, patch)` and resample counterfactual slip
  from the query-bank prior conditional on `phi` and the retained patch.

A `new_contact` query continues to carry only `p(theta, phi)` and samples the
complete counterfactual contact event.

## Posterior transport

For the opt-in same-patch semantic, the operator first marginalizes factual slip:

```text
p(theta, phi, patch | O+) = sum_slip p(theta, phi, patch, slip | O+).
```

It then combines this mass with the query-bank conditional distribution:

```text
p(slip_cf | phi, patch, do(u_cf)).
```

The current finite-bank implementation obtains that conditional distribution by
normalizing the preregistered query-hypothesis prior within each `(phi, patch)`
group. Therefore the bank must contain the intended slip support and its weights
must be frozen without target-future outcomes.

The same conditioning rule applies to every transport policy. When multiple
query hypotheses share the retained semantic key—`phi` for `new_contact`,
`(phi, kappa)` for fixed `same_grasp`, or `(phi, patch)` for `evolve_slip`—their
relative mass follows the frozen query-hypothesis prior rather than an arbitrary
uniform split. Exact-zero prior hypotheses remain exact-zero. A semantic group
with zero total prior is unavailable; if no represented factual mass remains,
the operator fails closed instead of returning a malformed posterior.

## Metadata

The returned `PhysicalPosterior` records:

- `same_grasp_semantics`;
- whether the complete factual `kappa` was reused;
- whether the factual contact patch was reused;
- whether factual slip was reused;
- whether counterfactual slip was resampled.

## Claim boundary

This change corrects intervention semantics but does not model a time-varying
contact regime by itself. The dynamic contact-path module remains responsible
for activation, sticking, slipping, detachment, and reattachment over time, and
a PhysTwin/Warp backend must still generate continuous path-conditioned
trajectories before real-data promotion.
