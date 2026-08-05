#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH_ORIGINAL="${PYTHONPATH:-}"
python_bin="${PREFIX_KINEMATICS_PYTHON:-python3}"

if [[ ! -x "${python_bin}" ]]; then
  echo "prefix-kinematics Python is not executable: ${python_bin}" >&2
  exit 1
fi

repository_root="${1:-$PWD}"
bpt_root="${2:-$PWD/_bpt}"
deform360_root="${3:-$PWD/_deform360}"
official_root="${4:-$PWD/_official_phystwin}"
output_dir="${5:-$PWD/runs/deform360-prefix-kinematics}"

repository_root="$(realpath "$repository_root")"
bpt_root="$(realpath "$bpt_root")"
deform360_root="$(realpath "$deform360_root")"
official_root="$(realpath "$official_root")"
mkdir -p "$output_dir"
output_dir="$(realpath "$output_dir")"

data_root="$(
  "$python_bin" \
    "$repository_root/scripts/remote/find_deform360_replication_root.py"
)"

export PYTHONPATH="${repository_root}/src:${bpt_root}/src"
export PYTHONPATH="${PYTHONPATH}:${deform360_root}:${PYTHONPATH_ORIGINAL:-}"
export PYTHONUNBUFFERED=1

"$python_bin" \
  - "$repository_root" "$bpt_root" "$deform360_root" "$official_root" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy
import torch
import warp as wp

repository_root, bpt_root, deform360_root, official_root = map(
    Path, sys.argv[1:]
)
expected = json.loads(
    (
        repository_root
        / "milestones"
        / "deform360-replication-source-backend-v1"
        / "verification"
        / "environment.json"
    ).read_text(encoding="utf-8")
)
protocol = json.loads(
    (
        repository_root
        / "configs"
        / "causal4d_public"
        / "deform360_replication_v1.json"
    ).read_text(encoding="utf-8")
)["config"]
observed = {
    "python": ".".join(map(str, sys.version_info[:3])),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "warp": wp.__version__,
}
for name, value in observed.items():
    if value != expected[name]:
        raise SystemExit(
            f"source-backend runtime changed for {name}: "
            f"expected {expected[name]!r}, observed {value!r}"
        )
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA")
wp.init()
if not wp.get_cuda_devices():
    raise SystemExit("Warp cannot see CUDA")
expected_revisions = {
    bpt_root: subprocess.check_output(
        [sys.executable, repository_root / "scripts/ci/read_bpt_pin.py"],
        text=True,
    ).strip(),
    deform360_root: protocol["deform360_code_commit"],
    official_root: protocol["official_phystwin_commit"],
}
for root, revision in expected_revisions.items():
    observed_revision = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if observed_revision != revision:
        raise SystemExit(
            f"repository revision changed for {root}: "
            f"expected {revision}, observed {observed_revision}"
        )
print(
    json.dumps(
        {
            "dataset_revision": expected["dataset_revision"],
            "repositories": {
                str(root): revision for root, revision in expected_revisions.items()
            },
            "runtime": observed,
        },
        sort_keys=True,
    )
)
PY

"$python_bin" -m pytest -q \
  "$repository_root/tests/test_deform360_prefix_kinematics.py" \
  "$repository_root/tests/test_deform360_prefix_kinematics_diagnostic.py" \
  "$repository_root/tests/test_deform360_replication_case.py" \
  "$repository_root/tests/test_deform360_replication_warp.py"

result="$output_dir/result.json"
"$python_bin" "$repository_root/scripts/remote/run_deform360_prefix_kinematics.py" \
  --repository-root "$repository_root" \
  --data-root "$data_root" \
  --bayesian-phystwin-repo "$bpt_root" \
  --deform360-repo "$deform360_root" \
  --official-phystwin-repo "$official_root" \
  --output "$result" \
  --device cuda:0

sha256sum "$result" "$output_dir/result.runtime.json" \
  > "$output_dir/SHA256SUMS"
cat "$output_dir/SHA256SUMS"
