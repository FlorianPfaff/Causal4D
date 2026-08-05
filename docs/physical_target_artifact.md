# Physical evaluation target artifact

The stable physical-counterfactual evaluator consumes a non-pickled,
content-addressed `PhysicalTargetBundle`. It no longer opens `final_data.pkl`
directly.

## Why this boundary exists

Python pickle is executable code. A digest proves byte identity but does not make
untrusted pickle content safe. Legacy PhysTwin data is therefore opened only by
an explicit conversion command that requires both operator consent and an
independently obtained SHA-256. All subsequent evaluation uses a strict NPZ
artifact loaded with `allow_pickle=False`.

The target artifact binds:

- the complete `CausalContext`, including protocol, case, observation windows,
  and counterfactual action;
- the canonical float32 `object_points` bytes used by the PhysTwin backend, from
  the pre-intervention endpoint through the end of `O+`;
- the point-frame validity mask and its canonical provider semantics;
- the trusted legacy source SHA-256;
- payload hashes and one content-derived artifact ID.

The importer reproduces the backend's declared float32 observation normalization
before checking `O-` and `O+`. The stored hashes therefore match the bytes used to
build the Causal4D context, while the trusted pickle SHA-256 separately binds the
original source file.

## Convert one trusted legacy target

Obtain the expected digest through the trusted experiment manifest or release
channel, not from an untrusted sidecar supplied with the same file.

```bash
sha256sum /data/case/final_data.pkl

causal4d evidence physical-target import-legacy \
  physical-posterior.npz \
  /data/case/final_data.pkl \
  physical-target.npz \
  --allow-unsafe-pickle \
  --expected-sha256 <lowercase-sha256>
```

The command verifies the digest before unpickling, rejects symlinked pickle
paths through `load_trusted_pickle`, validates the observation and validity
arrays, checks the complete `O+` digest against the posterior context, and
publishes the NPZ atomically. Publication is exactly once by default. Use
`--overwrite` only for an explicitly non-frozen replacement.

## Evaluate without pickle

```bash
causal4d evidence physical-counterfactual evaluate \
  physical-posterior.npz \
  physical-target.npz \
  physical-evaluation.json \
  --start-frame 7
```

The evaluator independently checks context equality, frame alignment, and node
support. It creates an additive `EvaluationTarget` identity for the exact suffix
selected by `--start-frame`, binds that ID together with the physical posterior,
query, and target IDs, and publishes a content-addressed JSON result atomically.
The output is also exactly once by default.

## Scientific boundary

A valid target artifact establishes data identity, alignment, and safe transport.
It is not accuracy, calibration, transfer, or physical-experiment evidence. The
registered analysis manifest and evidence registry still decide whether an
execution may contribute to a claim.
