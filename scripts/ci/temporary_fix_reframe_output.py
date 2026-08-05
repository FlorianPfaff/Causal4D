from __future__ import annotations

from pathlib import Path


path = Path("tests/test_deform360_prefix_kinematics_python_selector.py")
text = path.read_text(encoding="utf-8")
old = 'json.dumps(contract, indent=2, sort_keys=True) + "\n",'
new = 'json.dumps(contract, indent=2, sort_keys=True) + "\\n",'
if text.count(old) != 1:
    raise RuntimeError(
        "generated selector-test newline anchor changed: "
        f"found {text.count(old)} occurrences"
    )
path.write_text(text.replace(old, new, 1), encoding="utf-8")
