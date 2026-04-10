from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FunctionCallResult(BaseModel):
    """Schema for one function-calling output item."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    prompt: str
    name: str | None = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_input_keys(cls, data: Any) -> Any:
        """
        Accept both supported schemas:

        Subject-style:
        {
            "prompt": "...",
            "name": "...",
            "parameters": {...}
        }

        Scale-style:
        {
            "prompt": "...",
            "fn_name": "...",
            "args": {...}
        }
        """
        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        if "name" not in normalized and "fn_name" in normalized:
            normalized["name"] = normalized["fn_name"]

        if "parameters" not in normalized and "args" in normalized:
            normalized["parameters"] = normalized["args"]

        if "parameters" not in normalized or normalized["parameters"] is None:
            normalized["parameters"] = {}

        return normalized

    def to_subject_dict(self) -> Dict[str, Any]:
        """Export using the subject schema."""
        return {
            "prompt": self.prompt,
            "name": self.name,
            "parameters": self.parameters,
        }

    def to_scale_dict(self) -> Dict[str, Any]:
        """Export using the correction scale schema."""
        return {
            "prompt": self.prompt,
            "fn_name": self.name,
            "args": self.parameters,
        }
