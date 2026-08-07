# Full-posterior topology scores

The registered latent-contact result uses exact contact-node recovery as its
primary discrete gate. The existing contact-posterior diagnostic additionally
reports MAP graph distance, one-hop recovery, graph-symmetry proxies, and a
normalized graph-diffusion force-field cosine.

Those MAP diagnostics do not describe how the remaining posterior mass is
allocated. A posterior with 49% on the exact node and 51% on a mechanically
similar neighbor has a different evidential meaning from a posterior with 51%
on a remote node, even though both fail exact MAP recovery.

The additive topology-score layer therefore evaluates the complete categorical
node posterior. It does not modify the frozen estimator, posterior temperature,
registered exact-node threshold, success-gate decision, or physical protocol.

## Reported quantities

For each shifted-contact case, the diagnostic records:

- posterior expected mean and maximum assignment distance in graph hops;
- exact-node posterior probability;
- posterior mass within one-hop and two-hop contact patches;
- the smallest truth-centered graph radius containing the registered credible
  mass;
- the minimum graph distance between truth and any member of the registered node
  credible set;
- posterior expected cosine similarity in the declared graph-diffusion
  force-field proxy;
- posterior mass above force-field cosine thresholds 0.80, 0.90, and 0.95;
- expected force-field proxy distance;
- posterior pairwise force-field dispersion; and
- a force-field energy score.

The graph assignment distance minimizes over permutations of equal-cardinality
contact sets. This treats the material contact set as unlabeled while preserving
the registered cardinality.

## Proper force-field score

For a normalized graph-diffusion force-field feature `f(z)` and the realized
contact assignment `z*`, the reported energy score is

```text
E ||f(Z) - f(z*)|| - 0.5 E ||f(Z) - f(Z')||,
```

where `Z` and `Z'` are independent draws from the node posterior. Lower is
better, and a point mass on the realized force-field proxy has score zero. This
is a proper score for the declared Euclidean proxy representation. Assignments
with identical proxy fields are intentionally treated as equivalent by this
score; it is not a claim that they are physically identical under every action
or simulator.

## Integration

`scripts/ci/analyze_contact_posterior.py` augments the already admitted and
parity-checked diagnostic result before publication. The existing JSON and CSV
artifacts receive:

```text
posterior_topology_scores
posterior_expected_assignment_graph_distance_hops
posterior_one_hop_patch_mass
posterior_truth_centered_credible_radius_hops
posterior_expected_force_field_cosine
posterior_force_field_energy_score
```

plus the remaining threshold and credible-set fields. Aggregates are reported
overall and by held-out topology.

The source bundle must still pass the existing immutable admission boundary and
the recomputed posterior must still match the retained registered metrics. The
new score layer is downstream of both checks.

## Claim boundary

These quantities are controlled, analysis-only diagnostics. They may explain
why an exact-node gate fails or identify posterior mass concentrated on nearby
mechanically similar assignments. They cannot change, replace, or rescue a
failed registered exact-node gate, and they do not establish a real physical
interventional-prediction result.
