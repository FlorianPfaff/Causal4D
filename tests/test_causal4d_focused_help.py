from __future__ import annotations

import json

from causal4d.cli import root


def test_default_help_shows_stable_routes_only(capsys) -> None:
    assert root.main(["--help"]) == 0
    output = capsys.readouterr().out

    assert "stable routes:" in output
    assert "counterfactual" in output
    assert "protocol" in output
    assert "dynamic-contact" not in output
    assert "forecast-v1" not in output
    assert "Only stable routes are shown" in output
    assert "causal4d --help-all" in output


def test_help_all_shows_lifecycle_labels_without_importing_commands(capsys) -> None:
    assert root.main(["--help-all"]) == 0
    output = capsys.readouterr().out

    assert "all registered routes:" in output
    assert "dynamic-contact" in output
    assert "forecast-v1" in output
    assert "[experimental]" in output
    assert "[archive]" in output


def test_inventory_filters_by_lifecycle_and_claim_boundary(capsys) -> None:
    assert root.main(
        [
            "commands",
            "list",
            "--lifecycle",
            "stable",
            "--claim-bearing",
            "--json",
        ]
    ) == 0
    inventory = json.loads(capsys.readouterr().out)

    assert inventory
    assert all(item["lifecycle"] == "stable" for item in inventory)
    assert all(item["claim_bearing"] is True for item in inventory)
    assert any(item["route"] == ["protocol", "real"] for item in inventory)


def test_inventory_accepts_multiple_lifecycle_filters(capsys) -> None:
    assert root.main(
        [
            "commands",
            "list",
            "--lifecycle",
            "diagnostic",
            "--lifecycle",
            "experimental",
            "--json",
        ]
    ) == 0
    inventory = json.loads(capsys.readouterr().out)

    observed = {item["lifecycle"] for item in inventory}
    assert observed == {"diagnostic", "experimental"}
    assert all(item["lifecycle"] not in {"stable", "archive"} for item in inventory)


def test_unfiltered_json_inventory_remains_complete(capsys) -> None:
    assert root.main(["commands", "list", "--json"]) == 0
    inventory = json.loads(capsys.readouterr().out)

    lifecycles = {item["lifecycle"] for item in inventory}
    assert lifecycles == {
        "stable",
        "diagnostic",
        "experimental",
        "public-study",
        "archive",
    }
