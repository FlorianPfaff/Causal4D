"""Typed command registry for the grouped ``causal4d`` executable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
from typing import Any, Literal

Lifecycle = Literal["stable", "diagnostic", "experimental", "legacy"]


@dataclass(frozen=True)
class CommandSpec:
    """One lazy command route and its compatibility metadata."""

    route: tuple[str, ...]
    target: str
    summary: str
    lifecycle: Lifecycle
    legacy_name: str | None = None
    extras: tuple[str, ...] = ()
    owner: str = "Causal4D"

    def __post_init__(self) -> None:
        if not self.route or any(
            not token or token.startswith("-") for token in self.route
        ):
            raise ValueError("command routes must contain non-option tokens")
        if ":" not in self.target:
            raise ValueError("command targets must use module:function syntax")
        if not self.summary or not self.owner:
            raise ValueError("command summary and owner must be nonempty")
        if self.legacy_name is not None and not self.legacy_name.startswith(
            "causal4d-"
        ):
            raise ValueError("legacy names must start with causal4d-")

    @property
    def route_name(self) -> str:
        return " ".join(self.route)

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["route"] = list(self.route)
        values["route_name"] = self.route_name
        values["extras"] = list(self.extras)
        return values


GROUPED_COMMANDS = (
    CommandSpec(
        route=("benchmark", "counterfactual"),
        target="causal4d.cli.counterfactual_benchmark:main",
        summary="Run the controlled counterfactual benchmark.",
        lifecycle="stable",
        legacy_name="causal4d-counterfactual-benchmark",
    ),
    CommandSpec(
        route=("benchmark", "latent-contact"),
        target="causal4d.cli.latent_contact_benchmark:main",
        summary="Run the controlled latent-contact benchmark.",
        lifecycle="stable",
        legacy_name="causal4d-latent-contact-benchmark",
    ),
    CommandSpec(
        route=("benchmark", "dynamic-contact"),
        target="causal4d.cli.dynamic_contact_benchmark:main",
        summary="Run the dynamic contact-path benchmark.",
        lifecycle="experimental",
        legacy_name="causal4d-dynamic-contact-benchmark",
    ),
    CommandSpec(
        route=("protocol", "real"),
        target="causal4d.cli.real_protocol:main",
        summary="Scaffold, validate, and inspect the locked real protocol.",
        lifecycle="stable",
        legacy_name="causal4d-real-protocol",
    ),
    CommandSpec(
        route=("protocol", "freeze"),
        target="causal4d.cli.real_experiment_freeze:main",
        summary="Seal or validate a confirmatory method freeze.",
        lifecycle="stable",
        legacy_name="causal4d-real-experiment-freeze",
    ),
    CommandSpec(
        route=("protocol", "preacquisition-v4"),
        target="causal4d.cli.preacquisition_protocol_v4:main",
        summary="Validate the version-4 pre-acquisition amendment.",
        lifecycle="stable",
        legacy_name="causal4d-preacquisition-protocol-v4",
    ),
    CommandSpec(
        route=("protocol", "readiness"),
        target="causal4d.cli.preacquisition_readiness:main",
        summary="Verify evidence-bound readiness before confirmatory collection.",
        lifecycle="stable",
    ),
    CommandSpec(
        route=("protocol", "acquisition"),
        target="causal4d.cli.acquisition_operations:main",
        summary="Run the acquisition doctor, watchdog, and append-only journal.",
        lifecycle="stable",
    ),
    CommandSpec(
        route=("evidence", "observation-lineage"),
        target="causal4d.cli.observation_lineage:main",
        summary="Validate or bind observation provenance.",
        lifecycle="stable",
        legacy_name="causal4d-observation-lineage",
    ),
    CommandSpec(
        route=("evidence", "interpret-real-result"),
        target="causal4d.cli.real_result_interpretation:main",
        summary="Apply the preregistered real-result interpretation tree.",
        lifecycle="stable",
    ),
    CommandSpec(
        route=("calibration", "real"),
        target="causal4d.cli.real_calibration:main",
        summary="Fit or evaluate real predictive calibration.",
        lifecycle="diagnostic",
        legacy_name="causal4d-real-calibration",
    ),
    CommandSpec(
        route=("calibration", "execution-block"),
        target="causal4d.cli.execution_block_calibration:main",
        summary="Fit or evaluate execution-block conformal calibration.",
        lifecycle="stable",
        legacy_name="causal4d-execution-block-calibration",
    ),
    CommandSpec(
        route=("diagnostic", "real-oracle-gap"),
        target="causal4d.cli.audit_real_oracle_gap:main",
        summary="Audit inference, proposal, and model-discrepancy headroom.",
        lifecycle="diagnostic",
        legacy_name="causal4d-audit-real-oracle-gap",
    ),
    CommandSpec(
        route=("diagnostic", "real-failure-attribution"),
        target="causal4d.cli.real_failure_attribution:main",
        summary="Aggregate execution-accounted real failure diagnostics.",
        lifecycle="diagnostic",
        legacy_name="causal4d-aggregate-real-failure-attribution",
    ),
)


def grouped_commands() -> tuple[CommandSpec, ...]:
    """Return the frozen grouped command surface after validating uniqueness."""

    routes = [command.route for command in GROUPED_COMMANDS]
    legacy_names = [
        command.legacy_name
        for command in GROUPED_COMMANDS
        if command.legacy_name is not None
    ]
    if len(routes) != len(set(routes)):
        raise RuntimeError("grouped command routes are not unique")
    if len(legacy_names) != len(set(legacy_names)):
        raise RuntimeError("grouped command legacy names are not unique")
    return GROUPED_COMMANDS


def _installed_console_scripts() -> dict[str, str] | None:
    try:
        distribution = importlib.metadata.distribution("causal4d")
    except importlib.metadata.PackageNotFoundError:
        return None
    scripts: dict[str, str] = {}
    duplicates: list[str] = []
    for entry_point in distribution.entry_points:
        if entry_point.group != "console_scripts":
            continue
        if entry_point.name in scripts:
            duplicates.append(entry_point.name)
        scripts[entry_point.name] = entry_point.value
    if duplicates:
        raise RuntimeError(
            "installed console script names are duplicated: "
            + ", ".join(sorted(set(duplicates)))
        )
    return scripts


def installed_legacy_commands() -> tuple[CommandSpec, ...]:
    """Discover legacy scripts without importing their target modules."""

    scripts = _installed_console_scripts()
    if scripts is None:
        return ()
    commands = []
    for name, target in scripts.items():
        if not name.startswith("causal4d-"):
            continue
        commands.append(
            CommandSpec(
                route=("legacy", name.removeprefix("causal4d-")),
                target=target,
                summary=f"Compatibility route for {name}.",
                lifecycle="legacy",
                legacy_name=name,
            )
        )
    return tuple(sorted(commands, key=lambda command: command.route))


def command_inventory(*, include_legacy: bool = False) -> tuple[CommandSpec, ...]:
    commands = list(grouped_commands())
    if include_legacy:
        commands.extend(installed_legacy_commands())
    return tuple(commands)


def validate_runtime_command_inventory(
    *,
    require_installed: bool = False,
) -> dict[str, Any]:
    """Check grouped legacy mappings against installed console-script metadata.

    This validation imports no command target modules. It prevents a grouped
    route from silently drifting away from the historical executable retained by
    a frozen manifest, while reporting the still-ungrouped compatibility surface.
    """

    commands = grouped_commands()
    scripts = _installed_console_scripts()
    installed = scripts is not None
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    grouped_legacy = {
        command.legacy_name: command
        for command in commands
        if command.legacy_name is not None
    }
    if scripts is not None:
        for legacy_name, command in grouped_legacy.items():
            target = scripts.get(legacy_name)
            if target is None:
                missing.append(legacy_name)
            elif target != command.target:
                mismatched.append(
                    {
                        "legacy_name": legacy_name,
                        "registered_target": command.target,
                        "installed_target": target,
                    }
                )
        ungrouped = sorted(
            name
            for name in scripts
            if name.startswith("causal4d-") and name not in grouped_legacy
        )
    else:
        ungrouped = []
    valid = not missing and not mismatched and (installed or not require_installed)
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DCommandInventoryValidation",
        "valid": valid,
        "installed_distribution_present": installed,
        "require_installed": require_installed,
        "grouped_route_count": len(commands),
        "grouped_legacy_count": len(grouped_legacy),
        "missing_grouped_legacy_executables": sorted(missing),
        "target_mismatches": mismatched,
        "ungrouped_legacy_executables": ungrouped,
        "ungrouped_legacy_count": len(ungrouped),
    }


def find_command(name: str, *, include_legacy: bool = True) -> CommandSpec:
    """Resolve a route, slash route, or historical executable name."""

    normalized = " ".join(name.replace("/", " ").split())
    for command in command_inventory(include_legacy=include_legacy):
        if normalized == command.route_name or normalized == command.legacy_name:
            return command
    raise KeyError(name)
