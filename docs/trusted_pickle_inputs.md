# Trusted pickle inputs

The official PhysTwin compatibility path consumes three historical Python
pickle files: `final_data.pkl`, `optimal_params.pkl`, and `inference.pkl`.
Pickle can execute code while loading, so these files are **not** treated as a
safe interchange format.

Causal4D now fails closed unless the caller passes `--allow-unsafe-pickle` to
the PhysTwin belief, rollout-bank, or counterfactual command. This flag is an
explicit trust decision, not a sanitizer. The backend binds each loaded byte
sequence to the SHA-256 identity recorded in its source-artifact manifest and
rejects symlinked or non-ordinary inputs.

Use the flag only for files produced by a trusted, pinned PhysTwin workflow and
retained under the same content-addressed evidence boundary. Never pass it for
a downloaded, user-supplied, or dataset-bundled pickle whose producer is not
trusted. New artifact formats should use non-executable JSON, NPZ, or another
data-only serialization instead.
