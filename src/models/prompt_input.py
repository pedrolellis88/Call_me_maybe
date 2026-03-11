from pydantic import BaseModel


class PromptInput(BaseModel):
    """Schema for one prompt item from the input file."""

    prompt: str
