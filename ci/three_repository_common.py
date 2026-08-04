"""Shared helpers for the installed-wheel three-repository golden path."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

CAUSAL4D_REPOSITORY = "IPS-Stuttgart/Causal4D"
BAYESIAN_PHYSTWIN_REPOSITORY = "IPS-Stuttgart/BayesianPhysTwin"
PROB4D_REPOSITORY = "IPS-Stuttgart/Prob4D"

EXPECTED_OBSERVATION_ARTIFACT_ID = (
    "2a1f24acd2dd741155eb5333c92a37f615e1f0578b7f55cb9df0d34c52af8640"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def installed_package_origins(checkout_roots: tuple[Path, ...]) -> dict[str, str]:
    packages = {
        "prob4d": "prob4d",
        "bayesian-phystwin": "bayesian_phystwin",
        "causal4d": "causal4d",
    }
    resolved_roots = tuple(root.resolve() for root in checkout_roots)
    result: dict[str, str] = {}
    for distribution_name, module_name in packages.items():
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        require(module_file is not None, f"{module_name} has no import origin")
        origin = Path(module_file).resolve()
        require(
            not any(_is_within(origin, root) for root in resolved_roots),
            f"{module_name} imported from a source checkout: {origin}",
        )

        distribution = importlib.metadata.distribution(distribution_name)
        distribution_root = Path(distribution.locate_file("")).resolve()
        require(
            _is_within(distribution_root, Path(sys.prefix).resolve()),
            f"{distribution_name} is outside the clean virtual environment",
        )
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            direct_url_payload = json.loads(direct_url)
            editable = bool(direct_url_payload.get("dir_info", {}).get("editable"))
            require(not editable, f"{distribution_name} was installed editable")
        result[distribution_name] = str(origin)
    return result
