from causal4d.cli.counterfactual_phystwin import build_parser as counterfactual_parser
from causal4d.cli.phystwin_rollout_bank import build_parser as rollout_bank_parser


def test_rollout_bank_accepts_explicit_cache_directory() -> None:
    args = rollout_bank_parser().parse_args(
        [
            "official",
            "case",
            "profile.npz",
            "checkpoint.pt",
            "bank.npz",
            "--rollout-cache-dir",
            "cache",
        ]
    )
    assert args.rollout_cache_dir == "cache"


def test_counterfactual_accepts_explicit_cache_directory() -> None:
    args = counterfactual_parser().parse_args(
        [
            "official",
            "case",
            "profile.npz",
            "checkpoint.pt",
            "belief.npz",
            "factual.npz",
            "posterior.npz",
            "--rollout-cache-dir",
            "cache",
        ]
    )
    assert args.rollout_cache_dir == "cache"
