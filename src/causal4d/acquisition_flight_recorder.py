"""Public acquisition doctor, health, and flight-recorder surface."""

from causal4d.acquisition_doctor import (
    DOCTOR_REPORT_KIND,
    DoctorThresholds,
    build_acquisition_doctor_report,
)
from causal4d.acquisition_health import (
    HEALTH_SNAPSHOT_KIND,
    HealthThresholds,
    evaluate_health_snapshot,
)
from causal4d.acquisition_journal import (
    JOURNAL_EVENT_KIND,
    JOURNAL_SCHEMA_VERSION,
    JOURNAL_SEAL_KIND,
    append_journal_event,
    build_journal_event,
    journal_seal_path,
    seal_acquisition_journal,
    validate_acquisition_journal,
    validate_acquisition_journal_seal,
)

__all__ = [
    "DOCTOR_REPORT_KIND",
    "HEALTH_SNAPSHOT_KIND",
    "JOURNAL_EVENT_KIND",
    "JOURNAL_SCHEMA_VERSION",
    "JOURNAL_SEAL_KIND",
    "DoctorThresholds",
    "HealthThresholds",
    "append_journal_event",
    "build_acquisition_doctor_report",
    "build_journal_event",
    "evaluate_health_snapshot",
    "journal_seal_path",
    "seal_acquisition_journal",
    "validate_acquisition_journal",
    "validate_acquisition_journal_seal",
]
