# Atomic result-bundle publication

`causal4d.result_bundle_publication.publish_result_bundle` publishes a flat
multi-file result directory through an exactly-once transaction:

1. acquire a same-parent publication lock;
2. let the producer write ordinary files into a hidden `*.incomplete` directory;
3. flush every artifact and create a content-addressed `manifest.json` last;
4. verify exact file inventory, byte counts, and SHA-256 identities;
5. rename the complete directory to its final name; and
6. verify the exposed directory again.

Existing destinations are never replaced. Writer, validation, and rename
failures remove the staging directory and leave no authoritative partial
result.

```python
from pathlib import Path

from causal4d.result_bundle_publication import publish_result_bundle


def write_artifacts(staging: Path) -> None:
    (staging / "metrics.json").write_text('{"rmse": 0.012}\n')
    (staging / "predictions.bin").write_bytes(b"...")


verification = publish_result_bundle(
    "evidence/deform360-run-001",
    benchmark="deform360-prefix-v1",
    writer=write_artifacts,
)
```

Bundle writers may create only flat ordinary files and may not create the
reserved manifest themselves. The existing fail-closed verifier remains the
consumer boundary.
