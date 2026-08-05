#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH_ORIGINAL="${PYTHONPATH:-}"
python_bin="${FILAMENT_SUPPORT_PYTHON:-python3}"

if [[ ! -x "${python_bin}" ]]; then
  echo "filament-support Python is not executable: ${python_bin}" >&2
  exit 1
fi

repository_root="${1:-$PWD}"
output_dir="${2:-$PWD/runs/deform360-filament-support}"
repository_root="$(realpath "$repository_root")"
mkdir -p "$output_dir"
output_dir="$(realpath "$output_dir")"
runtime_selection="$output_dir/python-selection.json"

if [[ ! -f "$runtime_selection" ]]; then
  selected="$(
    "$python_bin" \
      "$repository_root/scripts/remote/select_deform360_prefix_kinematics_python.py" \
      --repository-root "$repository_root" \
      --candidate "$python_bin" \
      --report "$runtime_selection"
  )"
  if [[ "$selected" != "$python_bin" ]]; then
    echo "runtime selector chose another interpreter: $selected" >&2
    exit 1
  fi
fi

data_root="${DEFORM360_REPLICATION_ROOT:-}"
if [[ -z "$data_root" ]]; then
  data_root="$(
    "$python_bin" \
      "$repository_root/scripts/remote/find_deform360_replication_root.py"
  )"
fi

data_root="$(realpath "$data_root")"
export PYTHONPATH="${repository_root}/src:${PYTHONPATH_ORIGINAL:-}"
export PYTHONUNBUFFERED=1

"$python_bin" - <<'PY'
import numpy as np
import scipy

print({"numpy": np.__version__, "scipy": scipy.__version__})
PY

"$python_bin" -m pytest -q \
  "$repository_root/tests/test_deform360_filament_support.py" \
  "$repository_root/tests/test_deform360_rope_graph.py" \
  "$repository_root/tests/test_deform360_replication_graph.py" \
  "$repository_root/tests/test_deform360_reset_mechanics.py"

result="$output_dir/result.json"
"$python_bin" "$repository_root/scripts/remote/run_deform360_filament_support.py" \
  --repository-root "$repository_root" \
  --data-root "$data_root" \
  --runtime-selection "$runtime_selection" \
  --output "$result"

sha256sum \
  "$result" \
  "$output_dir/result.runtime.json" \
  "$runtime_selection" \
  > "$output_dir/SHA256SUMS"
cat "$output_dir/SHA256SUMS"
