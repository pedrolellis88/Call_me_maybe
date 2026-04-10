class ProjectError(Exception):
    """Base exception for the project."""


class InputFileError(ProjectError):
    """Raised when an input file is missing, unreadable, or invalid."""


class OutputFileError(ProjectError):
    """Raised when the output file cannot be serialized or written."""


class ValidationError(ProjectError):
    """Raised when parsed input data fails schema validation."""


class FunctionSelectionError(ProjectError):
    """Raised when no valid function can be selected for a prompt."""


class ParameterExtractionError(ProjectError):
    """Raised when one or more parameters cannot be extracted safely."""


class DecoderError(ProjectError):
    """Raised when constrained decoding fails."""
