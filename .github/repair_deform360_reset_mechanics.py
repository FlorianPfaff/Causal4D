from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/causal4d_public/deform360_reset_mechanics.py"
TESTS = ROOT / "tests/test_deform360_reset_mechanics.py"
DOCS = ROOT / "docs/causal4d_deform360_reset_mechanics.md"


def replace_block(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement + text[end:]


def patch_source() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    constant = 'SOURCE_MILESTONE = Path("milestones/deform360-replication-source-backend-v1")\n'
    if "_RESET_TECHNICAL_EXCEPTIONS" not in text:
        text = text.replace(
            constant,
            constant
            + "_RESET_TECHNICAL_EXCEPTIONS = (\n"
            + "    ValueError,\n"
            + "    RuntimeError,\n"
            + "    FloatingPointError,\n"
            + "    np.linalg.LinAlgError,\n"
            + ")\n",
            1,
        )

    episode_block = '''def _episode_has_technical_failure(episode: Mapping[str, Any]) -> bool:
    resets = episode.get("resets")
    if not isinstance(resets, list):
        return False
    return any(
        isinstance(reset, Mapping) and reset.get("status") == "technical_failure"
        for reset in resets
    )


def _episode_horizon_record(
    episode: Mapping[str, Any],
    horizon_observations: int,
) -> dict[str, Any] | None:
    resets = episode.get("resets")
    if not isinstance(resets, list) or len(resets) == 0:
        return None
    key = _horizon_key(horizon_observations)
    candidate_scores: list[float] = []
    persistence_scores: list[float] = []
    quality_flags: list[bool] = []
    for raw_reset in resets:
        reset = _require_mapping(raw_reset, message="reset record is not a mapping")
        if reset.get("status") != "completed":
            return None
        horizons = _require_mapping(
            reset.get("horizons"),
            message="reset horizon records are missing",
        )
        row = _require_mapping(
            horizons.get(key),
            message=f"reset horizon {key} is missing",
        )
        if row.get("finite") is not True:
            return None
        candidate_scores.append(
            _finite_float(row["mean_chamfer_m"], name="mean_chamfer_m")
        )
        persistence_scores.append(
            _finite_float(
                row["persistence_mean_chamfer_m"],
                name="persistence_mean_chamfer_m",
            )
        )
        quality_flags.append(row.get("quality_valid") is True)
    candidate = float(np.mean(candidate_scores))
    persistence = float(np.mean(persistence_scores))
    _require(persistence > 0.0, "episode persistence score must be positive")
    return {
        "episode_id": str(episode["episode_id"]),
        "object_id": str(episode["object_id"]),
        "reset_count": len(resets),
        "mean_chamfer_m": candidate,
        "persistence_mean_chamfer_m": persistence,
        "relative_improvement_vs_persistence": (persistence - candidate)
        / persistence,
        "win_vs_persistence": candidate < persistence,
        "quality_valid_fraction": float(np.mean(quality_flags)),
    }


'''
    text = replace_block(
        text,
        "def _episode_horizon_record(",
        "def summarize_reset_horizon(",
        episode_block,
    )

    rows = '''    rows = [
        row
        for episode in episode_records
        if (row := _episode_horizon_record(episode, horizon)) is not None
    ]
'''
    text = text.replace(
        rows,
        rows
        + '''    technical_failure_episode_count = sum(
        _episode_has_technical_failure(episode) for episode in episode_records
    )
    excluded_episode_count = len(episode_records) - len(rows)
''',
        1,
    )
    text = text.replace(
        '''            "common_episode_count": 0,
            "mean_chamfer_m": None,''',
        '''            "common_episode_count": 0,
            "excluded_episode_count": excluded_episode_count,
            "technical_failure_episode_count": technical_failure_episode_count,
            "mean_chamfer_m": None,''',
        1,
    )
    text = text.replace(
        '''        "common_episode_count": len(rows),
        "mean_chamfer_m": candidate,''',
        '''        "common_episode_count": len(rows),
        "excluded_episode_count": excluded_episode_count,
        "technical_failure_episode_count": technical_failure_episode_count,
        "mean_chamfer_m": candidate,''',
        1,
    )

    decision_block = '''def build_reset_mechanics_decision(
    episode_records: Sequence[Mapping[str, Any]],
    *,
    config: ResetMechanicsConfig,
) -> dict[str, Any]:
    """Apply the predeclared reset-and-roll competence ladder."""

    reproduction = [
        bool(record.get("prefix_baseline_reproduction", {}).get("passed"))
        for record in episode_records
    ]
    summaries = {
        _horizon_key(horizon): summarize_reset_horizon(
            episode_records,
            horizon,
            config=config,
        )
        for horizon in config.horizon_observation_counts
    }
    first_failure = next(
        (
            horizon
            for horizon in config.horizon_observation_counts
            if summaries[_horizon_key(horizon)]["passed"] is not True
        ),
        None,
    )
    technical_failure_episode_count = sum(
        _episode_has_technical_failure(record) for record in episode_records
    )
    technical_failure_reset_count = sum(
        1
        for record in episode_records
        for reset in record.get("resets", [])
        if isinstance(reset, Mapping) and reset.get("status") == "technical_failure"
    )
    baseline_passed = bool(reproduction and all(reproduction))
    passed = bool(baseline_passed and first_failure is None)
    first_summary = (
        summaries[_horizon_key(first_failure)] if first_failure is not None else None
    )
    if not baseline_passed:
        classification = "baseline_reproduction_failure"
        interpretation = (
            "the reset diagnostic cannot be interpreted because the frozen "
            "prefix baseline did not reproduce"
        )
    elif (
        first_summary is not None
        and first_summary["common_episode_count"]
        < config.minimum_common_episode_count
    ):
        classification = "insufficient_common_episode_support"
        interpretation = (
            "retained technical or nonfinite reset failures leave fewer complete "
            "episodes than the registered gate requires; no mechanics conclusion "
            "is permitted"
        )
    elif first_failure == config.horizon_observation_counts[0]:
        classification = "instantaneous_mechanics_or_contact_realization_failure"
        interpretation = (
            "observed-state resets do not rescue the first registered forecast "
            "horizon; prioritize contact realization, support, mass, or force laws"
        )
    elif first_failure is not None:
        classification = "multi_step_dynamics_accumulation_failure"
        interpretation = (
            "observed-state resets pass shorter horizons but fail by "
            f"{first_failure} observations; prioritize damping, integration, and "
            "topology-specific dynamics"
        )
    else:
        classification = "observed_reset_mechanics_competence_supported"
        interpretation = (
            "the current backend passes the source-only observed-reset ladder; "
            "the remaining prefix failure is more consistent with initialization, "
            "state estimation, or contact-state inference"
        )
    return {
        "baseline_reproduction_passed": baseline_passed,
        "baseline_reproduction_episode_count": len(reproduction),
        "technical_failure_episode_count": technical_failure_episode_count,
        "technical_failure_reset_count": technical_failure_reset_count,
        "horizon_summaries": summaries,
        "first_failed_horizon_observations": first_failure,
        "classification": classification,
        "passed": passed,
        "interpretation": interpretation,
        "registered_method_changed": False,
        "target_prefix_access_permitted": False,
        "target_future_access_permitted": False,
    }


'''
    text = replace_block(
        text,
        "def build_reset_mechanics_decision(",
        "def _clear_optional_cuda_cache(",
        decision_block,
    )

    evaluator = '''def _evaluate_registered_reset(
    *,
    build_observation: Any,
    episode_dir: Path,
    episode_id: str,
    stratum: str,
    frames: np.ndarray,
    hulls: Sequence[np.ndarray],
    schedule: Any,
    reset_ordinal: int,
    reset_position: int,
    official_phystwin_repo: Path,
    simulation_config: Any,
    candidate: Any,
    device: str,
    horizons: Sequence[int],
) -> dict[str, Any]:
    base = {
        "reset_ordinal": reset_ordinal,
        "reset_hull_position": reset_position,
        "reset_raw_frame": int(frames[reset_position]),
        "available_future_observation_count": len(frames) - reset_position - 1,
    }
    stage = "build_observation"
    try:
        observation = build_observation(
            episode_dir,
            episode_id,
            stratum,
            frames[reset_position:],
            hulls[reset_position:],
            schedule,
        )
        stage = "rollout_and_score"
        evaluation = _run_reset(
            observation,
            official_phystwin_repo,
            simulation_config,
            candidate,
            device=device,
            horizons=horizons,
        )
    except _RESET_TECHNICAL_EXCEPTIONS as exc:
        _clear_optional_cuda_cache()
        return {
            **base,
            "status": "technical_failure",
            "technical_failure": {
                "stage": stage,
                "exception_type": type(exc).__name__,
                "message": str(exc) or repr(exc),
            },
        }
    return {
        **base,
        "status": "completed",
        "contact_associations": list(observation.contact_associations),
        **evaluation,
    }


'''
    text = text.replace("def _episode_record(", evaluator + "def _episode_record(", 1)

    loop_start = text.index("    resets = []\n", text.index("def _episode_record("))
    loop_end = text.index("    reproduction_passed = bool(", loop_start)
    loop = '''    resets = [
        _evaluate_registered_reset(
            build_observation=build_replication_warp_observation,
            episode_dir=episode_dir,
            episode_id=episode_id,
            stratum=str(grid["stratum"]),
            frames=frames,
            hulls=hulls,
            schedule=schedule,
            reset_ordinal=reset_ordinal,
            reset_position=reset_position,
            official_phystwin_repo=official_phystwin_repo,
            simulation_config=simulation_config,
            candidate=candidate,
            device=device,
            horizons=config.horizon_observation_counts,
        )
        for reset_ordinal, reset_position in enumerate(reset_positions)
    ]
    completed_reset_count = sum(reset["status"] == "completed" for reset in resets)
    technical_failure_reset_count = len(resets) - completed_reset_count
    prefix = (
        resets[0].get("full_remainder")
        if resets[0]["status"] == "completed"
        else None
    )
    mean_delta = (
        abs(float(prefix["mean_chamfer_m"]) - selected["archived_mean_chamfer_m"])
        if isinstance(prefix, Mapping) and prefix["mean_chamfer_m"] is not None
        else None
    )
    strain_delta = (
        abs(
            float(prefix["p99_relative_edge_strain"])
            - selected["archived_p99_relative_edge_strain"]
        )
        if isinstance(prefix, Mapping)
        and prefix["p99_relative_edge_strain"] is not None
        else None
    )
'''
    text = text[:loop_start] + loop + text[loop_end:]
    text = text.replace(
        '''        "reset_positions": list(reset_positions),
        "resets": resets,
        "prefix_baseline_reproduction": {''',
        '''        "reset_positions": list(reset_positions),
        "resets": resets,
        "completed_reset_count": completed_reset_count,
        "technical_failure_reset_count": technical_failure_reset_count,
        "technically_complete": technical_failure_reset_count == 0,
        "prefix_baseline_reproduction": {''',
        1,
    )

    validation_start = text.index(
        "        resets = _require_nonempty_list(",
        text.index("def validate_source_reset_mechanics_diagnostic("),
    )
    validation_end = text.index(
        "        episode_boundary = _require_mapping(", validation_start
    )
    validation = '''        resets = _require_nonempty_list(
            record.get("resets"),
            message="reset-mechanics episode has no resets",
        )
        _require(
            len(resets) == config.reset_count,
            "reset-mechanics episode reset count changed",
        )
        completed_reset_count = 0
        technical_failure_reset_count = 0
        for reset_ordinal, raw_reset in enumerate(resets):
            reset = _require_mapping(
                raw_reset,
                message="reset-mechanics reset is not a mapping",
            )
            _require(
                reset.get("reset_ordinal") == reset_ordinal
                and reset.get("reset_hull_position")
                == expected_positions[reset_ordinal],
                "reset-mechanics reset ordering changed",
            )
            status = reset.get("status")
            _require(
                status in {"completed", "technical_failure"},
                "reset-mechanics reset status is invalid",
            )
            if status == "completed":
                completed_reset_count += 1
                _require(
                    "technical_failure" not in reset,
                    "completed reset contains technical-failure metadata",
                )
                horizons = _require_mapping(
                    reset.get("horizons"),
                    message="reset-mechanics reset horizons are missing",
                )
                _require(
                    tuple(horizons)
                    == tuple(
                        _horizon_key(horizon)
                        for horizon in config.horizon_observation_counts
                    ),
                    "reset-mechanics horizon set or ordering changed",
                )
            else:
                technical_failure_reset_count += 1
                _require(
                    "horizons" not in reset and "full_remainder" not in reset,
                    "technical-failure reset contains scientific scores",
                )
                failure = _require_mapping(
                    reset.get("technical_failure"),
                    message="reset technical-failure metadata is missing",
                )
                _require(
                    set(failure) == {"stage", "exception_type", "message"}
                    and all(
                        type(failure[field]) is str and failure[field]
                        for field in ("stage", "exception_type", "message")
                    ),
                    "reset technical-failure metadata is invalid",
                )
        _require(
            record.get("completed_reset_count") == completed_reset_count
            and record.get("technical_failure_reset_count")
            == technical_failure_reset_count
            and record.get("technically_complete")
            is (technical_failure_reset_count == 0),
            "reset-mechanics episode technical-failure accounting changed",
        )
'''
    text = text[:validation_start] + validation + text[validation_end:]
    SOURCE.write_text(text, encoding="utf-8")


def patch_tests() -> None:
    text = TESTS.read_text(encoding="utf-8")
    text = text.replace(
        '''                "reset_hull_position": reset_index * 2,
                "horizons": reset_horizons,''',
        '''                "reset_hull_position": reset_index * 2,
                "status": "completed",
                "horizons": reset_horizons,''',
        1,
    )
    text = text.replace(
        '''                "reset_hull_position": position,
                "horizons": {''',
        '''                "reset_hull_position": position,
                "status": "completed",
                "horizons": {''',
        1,
    )
    text = text.replace(
        '''        "reset_positions": list(positions),
        "resets": resets,
        "information_boundary": {''',
        '''        "reset_positions": list(positions),
        "resets": resets,
        "completed_reset_count": 3,
        "technical_failure_reset_count": 0,
        "technically_complete": True,
        "information_boundary": {''',
        1,
    )

    scoring_test = '''def test_registered_reset_retains_graph_construction_failure() -> None:
    def fail_to_build(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ValueError("rope point cloud remains disconnected")

    frames = np.arange(7, dtype=np.int64)
    hulls = tuple(np.zeros((2, 3), dtype=np.float64) for _ in range(7))
    result = mechanics._evaluate_registered_reset(
        build_observation=fail_to_build,
        episode_dir=Path("/tmp/episode"),
        episode_id="rope/episode_0001",
        stratum="filament",
        frames=frames,
        hulls=hulls,
        schedule={},
        reset_ordinal=0,
        reset_position=0,
        official_phystwin_repo=Path("/tmp/phystwin"),
        simulation_config=object(),
        candidate=object(),
        device="cpu",
        horizons=(1, 3, 6),
    )
    assert result["status"] == "technical_failure"
    assert result["technical_failure"] == {
        "stage": "build_observation",
        "exception_type": "ValueError",
        "message": "rope point cloud remains disconnected",
    }
    assert "horizons" not in result


'''
    text = text.replace(
        "def test_reset_positions_depend_only_on_frame_availability() -> None:\n",
        scoring_test
        + "def test_reset_positions_depend_only_on_frame_availability() -> None:\n",
        1,
    )

    support_tests = '''def test_technical_failure_excludes_the_complete_episode_unit() -> None:
    config = ResetMechanicsConfig(minimum_common_episode_count=1)
    episode = _episode("rope/episode_0001")
    episode["resets"][1] = {
        "reset_ordinal": 1,
        "reset_hull_position": 2,
        "status": "technical_failure",
        "technical_failure": {
            "stage": "build_observation",
            "exception_type": "ValueError",
            "message": "disconnected graph",
        },
    }
    summary = summarize_reset_horizon([episode], 3, config=config)
    assert summary["common_episode_count"] == 0
    assert summary["excluded_episode_count"] == 1
    assert summary["technical_failure_episode_count"] == 1
    assert summary["passed"] is False


def test_decision_does_not_relabel_insufficient_support_as_mechanics_failure() -> None:
    config = ResetMechanicsConfig(
        minimum_common_episode_count=2,
        minimum_relative_improvement=0.0,
        minimum_win_fraction=0.0,
        minimum_quality_valid_fraction=0.0,
    )
    failed = _episode("cloth/episode_0002")
    failed["resets"][1] = {
        "reset_ordinal": 1,
        "reset_hull_position": 2,
        "status": "technical_failure",
        "technical_failure": {
            "stage": "build_observation",
            "exception_type": "ValueError",
            "message": "disconnected graph",
        },
    }
    decision = build_reset_mechanics_decision(
        [_episode("rope/episode_0001"), failed],
        config=config,
    )
    assert decision["baseline_reproduction_passed"] is True
    assert decision["technical_failure_episode_count"] == 1
    assert decision["technical_failure_reset_count"] == 1
    assert decision["classification"] == "insufficient_common_episode_support"
    assert decision["passed"] is False


'''
    text = text.replace(
        "def test_decision_identifies_the_first_failed_horizon() -> None:\n",
        support_tests + "def test_decision_identifies_the_first_failed_horizon() -> None:\n",
        1,
    )

    validation_test = '''def test_result_validation_retains_well_formed_technical_failure() -> None:
    payload = _validation_payload()
    episode = payload["episode_records"][0]
    episode["resets"][1] = {
        "reset_ordinal": 1,
        "reset_hull_position": 2,
        "status": "technical_failure",
        "technical_failure": {
            "stage": "build_observation",
            "exception_type": "ValueError",
            "message": "rope point cloud remains disconnected",
        },
    }
    episode["completed_reset_count"] = 2
    episode["technical_failure_reset_count"] = 1
    episode["technically_complete"] = False
    payload["result_sha256"] = mechanics._artifact_sha256(payload)
    validate_source_reset_mechanics_diagnostic(payload)

    episode["resets"][1]["horizons"] = {}
    payload["result_sha256"] = mechanics._artifact_sha256(payload)
    with pytest.raises(ValueError, match="contains scientific scores"):
        validate_source_reset_mechanics_diagnostic(payload)


'''
    text = text.replace(
        "def test_result_validation_detects_tampering_and_target_access() -> None:\n",
        validation_test
        + "def test_result_validation_detects_tampering_and_target_access() -> None:\n",
        1,
    )
    TESTS.write_text(text, encoding="utf-8")


def patch_docs() -> None:
    text = DOCS.read_text(encoding="utf-8")
    text = text.replace(
        '''Each reset also retains full-remainder Chamfer and strain diagnostics. For every
registered horizon, reset scores are first averaged inside one episode. The episode
is the statistical unit; three resets are never treated as three independent
samples.
''',
        '''Each reset also retains full-remainder Chamfer and strain diagnostics. Graph
construction, simulator initialization, rollout, or scoring failures are retained
as explicit per-reset technical-failure records with their stage and exception
type. They are never repaired, replaced, or silently omitted. An episode contributes
to a horizon only when all three registered resets complete with finite scores. For
every registered horizon, reset scores are first averaged inside one episode. The
episode is the statistical unit; three resets are never treated as three independent
samples.
''',
        1,
    )
    text = text.replace(
        '''The result is classified by the first failed boundary:

1. `baseline_reproduction_failure` — no interpretation is permitted;
2. `instantaneous_mechanics_or_contact_realization_failure` — the first horizon
   fails despite an observed reset;
3. `multi_step_dynamics_accumulation_failure` — shorter horizons pass but a later
   horizon fails; or
4. `observed_reset_mechanics_competence_supported` — all registered reset horizons
   pass, redirecting the next study toward initialization, state estimation, or
   contact-state inference.
''',
        '''The result is classified by the first failed boundary:

1. `baseline_reproduction_failure` — no interpretation is permitted;
2. `insufficient_common_episode_support` — retained technical or nonfinite reset
   failures leave fewer than 24 complete episodes, so no mechanics conclusion is
   permitted;
3. `instantaneous_mechanics_or_contact_realization_failure` — the first horizon
   fails despite an observed reset and sufficient complete-episode support;
4. `multi_step_dynamics_accumulation_failure` — shorter horizons pass but a later
   horizon fails with sufficient support; or
5. `observed_reset_mechanics_competence_supported` — all registered reset horizons
   pass, redirecting the next study toward initialization, state estimation, or
   contact-state inference.
''',
        1,
    )
    DOCS.write_text(text, encoding="utf-8")


patch_source()
patch_tests()
patch_docs()
