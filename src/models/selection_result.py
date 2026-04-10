from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SelectionResult(BaseModel):
    """Internal result for function selection and argument extraction."""

    model_config = ConfigDict(
        extra="forbid",
    )

    prompt: str
    name: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
