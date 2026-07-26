from causal4d.cli.observation_lineage import build_parser


def test_factor_bundle_validation_command_parses() -> None:
    args = build_parser().parse_args(
        [
            "validate-factor-bundle",
            "observation_factors.json",
            "twin_belief.npz",
            "--require-bound",
        ]
    )

    assert args.command == "validate-factor-bundle"
    assert args.require_bound


def test_factor_bundle_binding_requires_explicit_flag_name() -> None:
    args = build_parser().parse_args(
        [
            "bind-factor-bundle",
            "observation_factors.json",
            "twin_belief.npz",
            "bound_twin_belief.npz",
            "--confirm-factor-bundle-was-consumed",
        ]
    )

    assert args.command == "bind-factor-bundle"
    assert args.confirm_factor_bundle_was_consumed
