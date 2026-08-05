# Identity-bound held-out physical targets

The stable beta-zero physical evaluator consumes a versioned, non-pickled
`HeldOutPhysicalTarget` artifact. It no longer opens `final_data.pkl` directly.
This separates the unsafe migration boundary from claim-bearing evaluation and
binds every score to the exact causal query and target bytes.

## Why this boundary exists

A pickle can execute arbitrary code while it is loaded. A compatible array shape
also does not prove that a target belongs to the same protocol, case, action, or
counterfactual query as a physical posterior. The held-out target archive closes
both gaps:

- it is a normal NumPy NPZ readable with `allow_pickle=False`;
- its descriptor includes the complete `CausalContext` and exact
  `source_query_id`;
- its target interval begins at the factual-prefix endpoint and ends at the
  registered counterfactual-action stop;
- schema v1 records the existing dense zero-based physical-node ordering
  explicitly;
- target positions, validity, node indices, and source bytes have SHA-256
  identities; and
- saving is atomic and followed by an independent reload/identity check.

The artifact contains only the aligned target trajectory required by the
physical posterior. It does not carry unused future observations or semantic
inputs.

## Convert a legacy PhysTwin target once

The migration command is intentionally not an installed console route. Invoke it
as a module and provide both explicit consent and the expected digest of the
trusted pickle bytes. The experimental factual-abduction CLI requires the same
explicit consent and digest before it can inspect the legacy prefix source:

```bash
python -m causal4d.cli.import_physical_target \
  physical_posterior.npz \
  /path/to/final_data.pkl \
  held_out_target.npz \
  --allow-unsafe-pickle \
  --expected-sha256 <lowercase-sha256> \
  --source-revision <dataset-or-acquisition-revision>
```

For the existing factual-abduction path, pass the same byte identity as:

```bash
python -m causal4d.cli.abduct_phystwin_intervention \
  rollout_bank.npz twin_belief.npz /path/to/final_data.pkl \
  factual.npz factual_evaluation.json \
  --allow-unsafe-pickle \
  --expected-final-data-sha256 <lowercase-sha256>
```

`--allow-unsafe-pickle` acknowledges executable-code risk. The SHA-256 check
establishes byte identity but does not make an untrusted pickle safe. The
converter rejects symlinks, nonordinary files, digest mismatch, malformed target
arrays, incompatible frame coverage, and a posterior whose trajectory length
does not agree with its causal context.

For a new acquisition pipeline, construct `HeldOutPhysicalTarget` directly from
validated data-only arrays and a source manifest instead of creating a pickle.

## Claim-bearing evaluation

```bash
causal4d evidence physical-counterfactual evaluate \
  physical_posterior.npz \
  held_out_target.npz \
  physical_evaluation.json \
  --start-frame 7 \
  --confidence-level 0.90 \
  --no-overwrite
```

Before scoring, the evaluator requires:

- exact equality of the posterior and target causal contexts;
- equality of the target and posterior `source_query_id`;
- exact trajectory and dense-node shape agreement; and
- a valid target interval and payload identity.

The output JSON is finite, key-sorted, atomically published, and content
addressed. It records:

- `evaluation_id`;
- `physical_posterior_id`;
- `held_out_target_id` and its complete validated descriptor;
- `source_query_id`;
- protocol, case, and counterfactual-action identities;
- relative and absolute evaluation intervals;
- confidence level and target source identity; and
- the complete beta-zero accuracy and calibration metrics.

The evaluator continues to state explicitly that semantic beta is zero and no
semantic evidence was consumed.

## CI policy

The command registry already distinguishes claim-bearing routes. CI now scans
all such command modules and rejects direct `pickle.load`, `pickle.loads`, and
`numpy.load(..., allow_pickle=True)` calls. A separate regression requires the
physical evaluator to use the safe held-out-target loader and the independently
reloaded atomic evaluation publisher rather than direct path writes.

This is an evidence-integrity and software-safety improvement. It does not change
posterior weights, the registered estimator, the 36-execution protocol, or any
frozen scientific result.
