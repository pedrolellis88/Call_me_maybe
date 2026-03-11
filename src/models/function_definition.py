from typing import Dict

from pydantic import BaseModel, Field


class ParameterDefinition(BaseModel):
    """Schema for a single function parameter."""

    type: str = Field(..., description="Parameter type, e.g. string, number, boolean.") # noqa


class ReturnDefinition(BaseModel):
    """Schema for a function return type."""

    type: str = Field(..., description="Return type.")


class FunctionDefinition(BaseModel):
    """Schema for an available callable function."""

    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: ReturnDefinition
