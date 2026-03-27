from typing import Any, Dict

from pydantic import BaseModel


class FunctionCallResult(BaseModel):
    """Schema for one output item."""

    prompt: str
    name: str | None
    parameters: Dict[str, Any]
