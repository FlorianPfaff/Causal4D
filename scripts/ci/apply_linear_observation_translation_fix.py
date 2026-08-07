#!/usr/bin/env python3
"""Apply the coordinate-wise endpoint-contrast validation change."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "src" / "causal4d" / "latent_contact_v2.py"

OLD = '''        for row in np.unique(rows[frames == 0]):
            row_terms = rows == row
            if not np.isclose(
                float(np.sum(coefficients[row_terms])),
                0.0,
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ValueError(
                    "endpoint frame zero may appear only in a zero-sum contrast"
                )
            if not np.any(frames[row_terms] > 0):
                raise ValueError("endpoint contrasts require a positive response frame")
'''

NEW = '''        for row in np.unique(rows[frames == 0]):
            row_terms = rows == row
            for coordinate in np.unique(coordinates[row_terms]):
                coordinate_terms = row_terms & (coordinates == coordinate)
                if not np.isclose(
                    float(np.sum(coefficients[coordinate_terms])),
                    0.0,
                    atol=1e-12,
                    rtol=1e-12,
                ):
                    raise ValueError(
                        "endpoint frame zero may appear only in a "
                        "coordinate-wise zero-sum contrast"
                    )
            if not np.any(frames[row_terms] > 0):
                raise ValueError("endpoint contrasts require a positive response frame")
'''


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(
            "expected exactly one reviewed endpoint validation block, "
            f"found {count}"
        )
    TARGET.write_text(text.replace(OLD, NEW), encoding="utf-8")


if __name__ == "__main__":
    main()
