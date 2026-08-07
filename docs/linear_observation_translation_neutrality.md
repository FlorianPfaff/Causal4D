# Translation-neutral linear observation contrasts

`LinearContactObservationGroup` may include frame zero only to form a registered
response contrast. Frame zero is not a fresh absolute endpoint observation.

For Cartesian trajectories, ordinary translation acts independently on each
coordinate. A row is therefore translation-neutral only when, for every
coordinate used by that row, its operator coefficients sum to zero:

```text
for every row r and coordinate c:
    sum(a_j for terms j in row r with coordinate c) = 0
```

A global coefficient sum is insufficient. For example,

```text
+x(frame 0) - y(frame 1)
```

has global sum zero but changes under translation because the unmatched `x` and
`y` masses cannot cancel each other. It would reuse the endpoint as an absolute
observation and is rejected.

Valid examples include same-coordinate temporal differences and weighted
multi-node contrasts such as:

```text
-x_node0(frame 0) + 0.5 x_node0(frame 1) + 0.5 x_node1(frame 1)
```

Rows without frame-zero terms retain their existing semantics. This validation
changes only malformed endpoint contrasts; it does not change the frozen
`v0.3.0-causal4d-aip` estimator, the registered 18-session/36-execution
protocol, evidence counts, or target-access rules.
