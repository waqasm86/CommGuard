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


class DurationRequirementError(WorkloadError):
    """A workload stopped before its declared measurement interval was met."""


class CoverageError(CommGuardError):
    """Primary evaluation coverage requirements were not satisfied."""


class ArtifactExistsError(CommGuardError):
    """A create-only artifact already exists."""
