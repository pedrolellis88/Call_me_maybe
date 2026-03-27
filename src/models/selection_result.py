from typing import Any, Dict, Optional

from pydantic import BaseModel


class SelectionResult(BaseModel):
    """Internal result for function selection/extraction."""

    prompt: str
    name: str | None
    parameters: Dict[str, Any]
    error: Optional[str] = None
