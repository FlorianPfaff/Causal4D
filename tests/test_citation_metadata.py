from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _required_scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*([^\n]+)$", text)
    if match is None:
        raise AssertionError(f"CITATION.cff is missing {key!r}")
    return match.group(1).strip().strip("\"'")


def test_citation_metadata_matches_package_version_and_repository() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'(?m)^version = "([^"]+)"$', pyproject)
    assert version_match is not None

    assert _required_scalar(citation, "cff-version") == "1.2.0"
    assert _required_scalar(citation, "title") == "Causal4D"
    assert _required_scalar(citation, "type") == "software"
    assert _required_scalar(citation, "version") == version_match.group(1)
    assert _required_scalar(citation, "repository-code") == (
        "https://github.com/FlorianPfaff/Causal4D"
    )
    assert "family-names: Pfaff" in citation
    assert "given-names: Florian" in citation
    assert (
        'Citation = "https://github.com/FlorianPfaff/Causal4D/blob/main/'
        'CITATION.cff"'
    ) in pyproject
    assert "\t" not in citation
