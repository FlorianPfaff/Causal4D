#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  stage_software_environment.sh \
    <causal4d-root> <bayesianphystwin-root> <dataset-root> <deployment-venv> \
    <observation-producer-name> <observation-producer-version> \
    <observation-artifact-contract> <numpy_cpu|warp_cpu|cuda> \
    [causal4d-extras] [container-image-digest]

The optional extras argument is a comma-separated Causal4D extras list. Use '-'
for no extras. Defaults are empty for numpy_cpu, warp for warp_cpu, and
vision,warp for cuda.

Run this only after the method freeze and independent attestation exist. The
script refuses a pre-existing deployment environment, builds wheels from clean
Git archives, installs them into that environment, captures pip freeze, and
stages the unapproved software-environment gate. Independent approval and
`seal-gate software_environment_locked` remain separate steps.
EOF
}

if (( $# < 8 || $# > 10 )); then
  usage
  exit 2
fi

causal4d_root="$(cd "$1" && pwd -P)"
bayesian_phystwin_root="$(cd "$2" && pwd -P)"
dataset_root="$(cd "$3" && pwd -P)"
deployment_venv="$4"
producer_name="$5"
producer_version="$6"
producer_contract="$7"
backend="$8"
extras="${9:-}"
container_digest="${10:-}"
python_bin="${PYTHON:-python3}"

case "$backend" in
  numpy_cpu)
    default_extras=""
    ;;
  warp_cpu)
    default_extras="warp"
    ;;
  cuda)
    default_extras="vision,warp"
    ;;
  *)
    echo "Unsupported execution backend: $backend" >&2
    exit 2
    ;;
esac

if [[ -z "$extras" ]]; then
  extras="$default_extras"
elif [[ "$extras" == "-" ]]; then
  extras=""
fi

if [[ -e "$deployment_venv" ]]; then
  echo "Deployment environment already exists: $deployment_venv" >&2
  exit 2
fi

for checkout in "$causal4d_root" "$bayesian_phystwin_root"; do
  git -C "$checkout" rev-parse --show-toplevel >/dev/null
  test -z "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)"
done

if ! "$python_bin" -m build --version >/dev/null 2>&1; then
  echo "The selected Python environment needs the 'build' package." >&2
  exit 2
fi

work_root="$(mktemp -d "${TMPDIR:-/tmp}/causal4d-acquisition-capsule.XXXXXX")"
completed=false
cleanup() {
  rm -rf "$work_root"
  if [[ "$completed" != true ]]; then
    rm -rf "$deployment_venv"
  fi
}
trap cleanup EXIT

mkdir -p "$work_root/causal4d" "$work_root/bayesianphystwin" "$work_root/wheels"
git -C "$causal4d_root" archive --format=tar HEAD \
  | tar -xf - -C "$work_root/causal4d"
git -C "$bayesian_phystwin_root" archive --format=tar HEAD \
  | tar -xf - -C "$work_root/bayesianphystwin"

"$python_bin" -m build --wheel \
  --outdir "$work_root/wheels" \
  "$work_root/causal4d"
"$python_bin" -m build --wheel \
  --outdir "$work_root/wheels" \
  "$work_root/bayesianphystwin"

mapfile -t causal4d_wheels < <(
  find "$work_root/wheels" -maxdepth 1 -type f -name 'causal4d-*.whl' | sort
)
mapfile -t bayesian_phystwin_wheels < <(
  find "$work_root/wheels" -maxdepth 1 -type f \
    -name 'bayesian_phystwin-*.whl' | sort
)
if (( ${#causal4d_wheels[@]} != 1 || ${#bayesian_phystwin_wheels[@]} != 1 )); then
  echo "Expected exactly one Causal4D and one BayesianPhysTwin wheel." >&2
  exit 2
fi
causal4d_wheel="${causal4d_wheels[0]}"
bayesian_phystwin_wheel="${bayesian_phystwin_wheels[0]}"

"$python_bin" -m venv "$deployment_venv"
"$deployment_venv/bin/python" -m pip install --upgrade pip

causal4d_uri="$(
  "$python_bin" - "$causal4d_wheel" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve().as_uri())
PY
)"
bayesian_phystwin_uri="$(
  "$python_bin" - "$bayesian_phystwin_wheel" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).resolve().as_uri())
PY
)"
causal4d_requirement="causal4d"
if [[ -n "$extras" ]]; then
  causal4d_requirement+="[$extras]"
fi
causal4d_requirement+=" @ $causal4d_uri"
bayesian_phystwin_requirement="bayesian-phystwin @ $bayesian_phystwin_uri"

"$deployment_venv/bin/python" -m pip install \
  "$causal4d_requirement" \
  "$bayesian_phystwin_requirement"
"$deployment_venv/bin/python" -m pip check
"$deployment_venv/bin/python" -m pip freeze --all \
  > "$work_root/resolved-dependencies.txt"

command=(
  "$deployment_venv/bin/causal4d"
  protocol readiness software-environment-stage
  "$causal4d_root"
  "$bayesian_phystwin_root"
  "$dataset_root"
  "$causal4d_wheel"
  "$bayesian_phystwin_wheel"
  "$work_root/resolved-dependencies.txt"
  --observation-producer-name "$producer_name"
  --observation-producer-version "$producer_version"
  --observation-artifact-contract "$producer_contract"
  --execution-backend "$backend"
)
if [[ -n "$container_digest" ]]; then
  command+=(--container-image-digest "$container_digest")
fi
"${command[@]}"

completed=true
printf 'Deployment environment: %s\n' "$(cd "$deployment_venv" && pwd -P)"
printf 'Next independent step:\n'
printf '  %q protocol readiness seal-gate %q %q software_environment_locked --approved-by %q\n' \
  "$deployment_venv/bin/causal4d" \
  "$causal4d_root" \
  "$dataset_root" \
  '<independent-verifier>'
