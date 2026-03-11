class ProjectError(Exception):
    """Base exception for the project."""


class InputFileError(ProjectError):
    """Raised when an input file is missing or invalid."""


class OutputFileError(ProjectError):
    """Raised when the output file cannot be written."""


class ValidationError(ProjectError):
    """Raised when input data fails validation."""
