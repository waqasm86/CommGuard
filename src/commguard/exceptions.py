"""Typed CommGuard failures."""


class CommGuardError(Exception):
    """Base exception for expected SDK failures."""


class ValidationError(CommGuardError):
    """An artifact or configuration violates its contract."""


class ReadinessError(CommGuardError):
    """The environment cannot support the requested experiment."""


class CalibrationError(CommGuardError):
    """The communication calibration gate was not satisfied."""


class WorkloadError(CommGuardError):
    """A workload failed or its participation evidence was invalid."""


class ArtifactExistsError(CommGuardError):
    """A create-only artifact already exists."""
