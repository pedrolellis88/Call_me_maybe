from pydantic import BaseModel, ConfigDict, field_validator


class PromptInput(BaseModel):
    """Schema for one prompt item from the input file."""

    model_config = ConfigDict(
        extra="forbid",
    )

    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Validate and normalize the input prompt."""
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("Prompt cannot be empty.")

        return cleaned
