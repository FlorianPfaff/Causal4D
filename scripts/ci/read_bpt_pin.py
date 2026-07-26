#!/usr/bin/env python3
"""Print the exact Bayesian-PhysTwin revision used by pinned CI."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

DEFAULT_PIN = Path("requirements/ci/bayesian-phystwin-provider-v1.sha")
_SHA40 = re.compile(r"^[0-9a-f]{40}$")


def read_bpt_pin(path: Path = DEFAULT_PIN) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if _SHA40.fullmatch(value) is None:
        raise ValueError(f"{path} must contain one lowercase 40-hex commit")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pin_file", nargs="?", type=Path, default=DEFAULT_PIN)
    arguments = parser.parse_args(argv)
    print(read_bpt_pin(arguments.pin_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
