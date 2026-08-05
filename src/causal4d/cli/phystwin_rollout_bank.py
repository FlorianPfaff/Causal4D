"""Build a real PhysTwin rollout bank for Causal4D inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _load_runtime_dependencies() -> None:
    """Load optional integrations only after argparse handles ``--help``."""
    global export_official_phystwin_twin_belief
    global TwinBelief
    global load_contract
    global save_contract
    global OfficialPhysTwinBackend
    global OfficialPhysTwinBackendConfig
    global PhysTwinActionProposal
    global PhysTwinHypothesisConfig
    global build_resumable_rollout_bank
    global hidden_action_proposals
    global known_action_proposal
    global save_rollout_bank

    from causal4d.bpt_belief import export_official_phystwin_twin_belief
    from causal4d.contracts import TwinBelief, load_contract, save_contract
    from causal4d.phystwin_backend import (
        OfficialPhysTwinBackend,
        OfficialPhysTwinBackendConfig,
        PhysTwinActionProposal,
        PhysTwinHypothesisConfig,
        hidden_action_proposals,
        known_action_proposal,
    )
    from causal4d.phystwin_resumable import build_resumable_rollout_bank
    from causal4d.rollout_bank_io import save_rollout_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Causal4D action/contact hypotheses through the official "
            "PhysTwin simulator under Bayesian-PhysTwin parameter particles."
        )
    )
    parser.add_argument("official_repo")
    parser.add_argument("case_dir")
    parser.add_argument("profile_path")
    parser.add_argument("checkpoint_path")
    parser.add_argument("output_npz")
    parser.add_argument(
        "--action-setting",
        choices=("known", "hidden", "ambiguous"),
        default="hidden",
    )
    parser.add_argument("--train-end-frame", type=int)
    parser.add_argument("--parameter-particles", type=int, default=4)
    parser.add_argument(
        "--parameter-support-method",
        choices=("top_mass", "weighted_coreset"),
        help="support reduction; inherited from --twin-belief when omitted",
    )
    parser.add_argument(
        "--twin-belief",
        help=(
            "Existing TwinBelief NPZ. If omitted, every theta particle is replayed "
            "through O- and a sibling .twin_belief.npz artifact is written."
        ),
    )
    parser.add_argument("--maximum-contact-states", type=int, default=12)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--num-substeps", type=int, default=667)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nondeterministic-spring-forces", action="store_true")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help=(
            "Explicitly trust the local PhysTwin .pkl inputs; loading "
            "pickle can execute arbitrary code."
        ),
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--rollout-cache-dir",
        help=(
            "Content-addressed per-rollout cache. Defaults to a sibling directory "
            "named after output_npz."
        ),
    )
    cache_group.add_argument(
        "--no-rollout-cache",
        action="store_true",
        help="Disable resumable per-rollout caching.",
    )
    return parser


def _train_end(case_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
    return int(split["train"][1])


def _action_proposals(
    setting: str,
    controller_points,
    *,
    train_end_frame: int,
) -> tuple[PhysTwinActionProposal, ...]:
    known = known_action_proposal(controller_points)
    hidden = hidden_action_proposals(
        controller_points,
        start_frame=train_end_frame,
    )
    if setting == "known":
        return (known,)
    if setting == "hidden":
        return hidden
    ambiguous_known = PhysTwinActionProposal(
        proposal_id=known.proposal_id,
        controller_points_m=known.controller_points_m,
        prior_weight=1.0,
        future_action_observed=True,
        provenance="candidate in an ambiguous finite action library",
    )
    return (ambiguous_known, *hidden)


def _cache_directory(args: argparse.Namespace, output_path: Path) -> Path | None:
    if args.no_rollout_cache:
        return None
    if args.rollout_cache_dir:
        return Path(args.rollout_cache_dir)
    return output_path.with_name(output_path.stem + ".rollout-cache")


def _cache_summary(manifest: dict) -> dict:
    cache = manifest["rollout_cache"]
    return {
        key: cache[key]
        for key in (
            "enabled",
            "root",
            "record_count",
            "hit_count",
            "miss_count",
            "repaired_count",
            "provider_instance_count",
        )
        if key in cache
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    case_dir = Path(args.case_dir)
    output_path = Path(args.output_npz)
    train_end = _train_end(case_dir, args.train_end_frame)
    loaded_belief = None
    if args.twin_belief:
        loaded = load_contract(args.twin_belief)
        if not isinstance(loaded, TwinBelief):
            raise TypeError("--twin-belief must contain a TwinBelief artifact")
        loaded_belief = loaded
    support_method = args.parameter_support_method or (
        str(loaded_belief.metadata.get("profile_support_method", "top_mass"))
        if loaded_belief is not None
        else "top_mass"
    )
    backend = OfficialPhysTwinBackend(
        official_repo=args.official_repo,
        final_data_path=case_dir / "final_data.pkl",
        optimal_params_path=case_dir / "optimal_params.pkl",
        checkpoint_path=args.checkpoint_path,
        baseline_trajectory_path=case_dir / "inference.pkl",
        profile_path=args.profile_path,
        train_end_frame=train_end,
        parameter_particle_count=args.parameter_particles,
        parameter_support_method=support_method,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        config=OfficialPhysTwinBackendConfig(
            dt=args.dt,
            num_substeps=args.num_substeps,
            deterministic_spring_forces=not args.nondeterministic_spring_forces,
            device=args.device,
        ),
    )
    proposals = _action_proposals(
        args.action_setting,
        backend.controller_points,
        train_end_frame=train_end,
    )
    context = backend.causal_context(proposals)
    if loaded_belief is not None:
        twin_belief = loaded_belief
        twin_belief_path = Path(args.twin_belief)
    else:
        twin_belief = export_official_phystwin_twin_belief(
            backend,
            context=context,
        )
        twin_belief_path = output_path.with_name(output_path.stem + ".twin_belief.npz")
        save_contract(twin_belief_path, twin_belief)
    bank, manifest = build_resumable_rollout_bank(
        backend,
        proposals,
        twin_belief=twin_belief,
        hypothesis_config=PhysTwinHypothesisConfig(
            maximum_contact_states=args.maximum_contact_states
        ),
        rollout_cache_dir=_cache_directory(args, output_path),
    )
    manifest["action_setting"] = args.action_setting
    save_rollout_bank(output_path, bank, manifest)
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "case": backend.case_name,
                "action_setting": args.action_setting,
                "rollout_shape": list(bank.trajectories.shape),
                "rollout_cache": _cache_summary(manifest),
                "bpt_retained_parameter_mass": (
                    backend.particles.bpt_retained_probability_mass
                ),
                "causal4d_retained_parameter_mass": (
                    backend.particles.causal4d_retained_probability_mass
                ),
                "retained_parameter_mass": backend.particles.retained_probability_mass,
                "represented_parameter_mass": (
                    backend.particles.represented_probability_mass
                ),
                "parameter_mass_accounting": (
                    backend.particles.probability_mass_accounting()
                ),
                "parameter_support_method": support_method,
                "twin_belief": str(twin_belief_path.resolve()),
                "twin_belief_id": twin_belief.artifact_id,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
