# Contact-posterior diagnostic admission

Evidence-bearing contact-posterior analysis enters through
`analyze_admitted_contact_posterior_bundle`. The topology-aware numerical analyzer
remains a reusable lower-level kernel, but the command-line workflow and published
diagnostic artifacts use the admission boundary.

The boundary performs two independent checks before recomputation:

1. `verify_embedded_result_bundle` verifies the flat result-manifest schema, exact
   payload inventory, byte counts, SHA-256 identities, and ordinary-file/symlink
   constraints.
2. `verify_contact_posterior_source_bundle` verifies the Causal4D-specific bundle
   schema, unique seeds and row identities, canonical scalar encodings, source
   topology exclusions, observation fractions, paired trajectory methods, and
   agreement between summary and gate artifacts.

Finite diagnostics declared on the unit interval are admitted within `1e-12` of
its endpoints to accommodate machine-scale probability summation roundoff. The
raw serialized value is not clipped or rewritten, the tolerance is recorded in
the integrity report, and a larger excursion remains inadmissible.

The two verifiers must report the same source-manifest SHA-256. The low-level
analyzer must then retain that exact manifest identity. After analysis, both
verifiers run again and their complete reports must equal the pre-analysis reports.
This closes the interval in which a mutable bundle could otherwise change after
admission but before interpretation. Any disagreement stops publication.

Published source provenance contains the portable bundle name, manifest digest,
verified artifact identities, and both integrity reports. Runner-local absolute
paths are removed. This admission step is diagnostic-only: it does not alter the
frozen estimator, posterior, thresholds, exact-node gate, registered experiment,
or result values.
