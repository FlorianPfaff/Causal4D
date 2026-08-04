from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _required_scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*([^\n]+)$", text)
    if match is None:
        raise AssertionError(f"CITATION.cff is missing {key!r}")
    return match.group(1).strip().strip("\"'")


def test_citation_metadata_matches_package_version_repository_and_license() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    version_match = re.search(r'(?m)^version = "([^"]+)"$', pyproject)
    assert version_match is not None

    assert _required_scalar(citation, "cff-version") == "1.2.0"
    assert _required_scalar(citation, "title") == "Causal4D"
    assert _required_scalar(citation, "type") == "software"
    assert _required_scalar(citation, "version") == version_match.group(1)
    assert _required_scalar(citation, "license") == "MIT"
    assert _required_scalar(citation, "repository-code") == (
        "https://github.com/IPS-Stuttgart/Causal4D"
    )
    assert _required_scalar(citation, "url") == (
        "https://github.com/IPS-Stuttgart/Causal4D"
    )
    assert "family-names: Pfaff" in citation
    assert "given-names: Florian" in citation
    assert 'license = { file = "LICENSE" }' in pyproject
    assert '"License :: OSI Approved :: MIT License"' in pyproject
    assert (
        'Citation = "https://github.com/IPS-Stuttgart/Causal4D/blob/main/CITATION.cff"'
    ) in pyproject
    assert 'Repository = "https://github.com/IPS-Stuttgart/Causal4D"' in pyproject
    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 Florian Pfaff" in license_text
    assert "\t" not in citation


def test_current_documentation_explains_the_license_scope() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    licensing = (ROOT / "docs" / "licensing.md").read_text(encoding="utf-8")

    assert "## License" in readme
    assert "[MIT License](LICENSE)" in readme
    assert "## Licensing" in contributing
    assert "[MIT License](LICENSE)" in contributing
    assert "third-party material" in licensing
    assert "Frozen tags" in licensing
