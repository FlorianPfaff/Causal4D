#!/usr/bin/env python3
"""Print the exact Bayesian-PhysTwin commit pinned by the ``phystwin`` extra."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import tomllib


def read_bpt_pin(pyproject: Path) -> str:
    with pyproject.open("rb") as handle:
        optional = tomllib.load(handle)["project"]["optional-dependencies"]
    requirements = optional.get("phystwin", [])
    matches: list[str] = []
    for requirement in requirements:
        text = str(requirement)
        if text.lower().startswith("bayesian-phystwin"):
            match = re.search(r"@([0-9a-f]{40})$", text)
            if match is None:
                raise ValueError(
                    "the Bayesian-PhysTwin requirement must end in an exact 40-hex commit"
                )
            matches.append(match.group(1))
    if len(matches) != 1:
        raise ValueError(
            "the phystwin extra must declare exactly one Bayesian-PhysTwin requirement"
        )
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pyproject", nargs="?", type=Path, default=Path("pyproject.toml"))
    arguments = parser.parse_args(argv)
    print(read_bpt_pin(arguments.pyproject))
    return 0


if __name__ == "__main__":
    sys.exit(main())
