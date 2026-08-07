# Graph-valued contact posterior

`causal4d.graph_contact_measure.GraphContactMeasure` is an immutable,
diagnostic-only representation of posterior mass over one or more contact nodes on
an object graph. It complements exact-node accuracy with topology-aware uncertainty
without changing the frozen Causal4D estimator or any registered gate.

## Why use it

An exact contact label can be too brittle when neighboring mesh nodes induce nearly
equivalent physical responses. The measure keeps the complete discrete posterior and
adds:

- canonical, permutation-invariant multi-contact states;
- deterministic MAP and tie-closed highest-mass credible states;
- graph-Bayes contact estimates under minimum mean assignment distance;
- exact and one-hop credible-region coverage;
- posterior expected graph error and pairwise graph dispersion;
- node-marginal contact probabilities for visualization; and
- content identities for both the graph-distance matrix and posterior measure.

The one-hop metric is an additional diagnostic. It does not replace the frozen exact
contact criterion.

## Example

```python
import numpy as np

from causal4d.graph_contact_measure import (
    GraphContactMeasure,
    all_pairs_shortest_path_distances,
)

# Four-node chain: 0 -- 1 -- 2 -- 3
distances = all_pairs_shortest_path_distances(
    4,
    [(0, 1), (1, 2), (2, 3)],
)
measure = GraphContactMeasure.from_weighted_contacts(
    contacts=[(0,), (1,), (3,)],
    weights=np.asarray([0.5, 0.3, 0.2]),
    graph_distances=distances,
)
report = measure.as_record(0.8, truth=(2,))
```

Here the 80% credible states are nodes 0 and 1. Exact credible coverage of node 2 is
false, while one-hop credible coverage is true. The report still retains the exact
truth probability, MAP graph error, posterior expected graph error, and graph-Bayes
risk, so the relaxed diagnostic cannot hide exact-node performance.

## Distance units

`all_pairs_shortest_path_distances` produces unweighted shortest-path distances.
Consequently, radius `1.0` means one graph edge, not one metre or one mesh-cell width.
Comparisons across differently discretized meshes require a separately registered
mapping or a physically scaled distance matrix; the helper does not infer one.

## Interpretation

Treat topology-aware quantities as an attribution aid, not as a replacement success
criterion. A one-hop recovery can distinguish a localized contact miss from a remote
or diffuse posterior, but the registered exact-node result remains unchanged. Report
both quantities together whenever a relaxed radius is shown.

The graph-distance matrix is part of the measure identity. The same probabilities on
a different mesh topology are therefore a different diagnostic object, even when the
node labels happen to match. This prevents results from being compared across
incompatible graph registrations without an explicit transformation.

The graph-Bayes estimate minimizes posterior expected assignment distance over the
retained posterior support. It does not synthesize an unrepresented contact subset.
Expanding the admissible decision space requires an explicit method change and must
not be inferred from this diagnostic summary.

## Multi-contact states

Node order is not semantic. `(4, 0)` and `(0, 4)` are canonicalized and their masses
are aggregated. All support states must have the same cardinality. Distances between
multi-contact states use an optimal one-to-one assignment; both minimum mean and
minimum bottleneck graph distances are available.

## Boundary

The class summarizes an existing posterior only. It does not read future frames,
change contact hypotheses or weights, widen the intervention bank, tune thresholds,
or authorize confirmatory acquisition. `as_record()` marks output as
`diagnostic_only=true` and includes deterministic SHA-256 identities.
