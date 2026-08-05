from __future__ import annotations

import pytest

from causal4d.cli.counterfactual_phystwin import (
    build_parser as counterfactual_parser,
)
from causal4d.cli.export_bpt_belief import build_parser as belief_parser
from causal4d.cli.phystwin_rollout_bank import build_parser as rollout_parser


@pytest.mark.parametrize(
    ("parser_factory", "positionals"),
    [
        (belief_parser, ["repo", "case", "profile", "checkpoint", "out"]),
        (rollout_parser, ["repo", "case", "profile", "checkpoint", "out"]),
        (
            counterfactual_parser,
            [
                "repo",
                "case",
                "profile",
                "checkpoint",
                "belief",
                "factual",
                "out",
            ],
        ),
    ],
)
def test_phystwin_pickle_consent_is_explicit(parser_factory, positionals) -> None:
    parser = parser_factory()
    assert parser.parse_args(positionals).allow_unsafe_pickle is False
    assert (
        parser.parse_args([*positionals, "--allow-unsafe-pickle"]).allow_unsafe_pickle
        is True
    )
