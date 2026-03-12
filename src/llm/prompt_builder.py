from typing import List

from src.models.function_definition import FunctionDefinition


def build_system_prompt(functions: List[FunctionDefinition]) -> str:
    """Build the system prompt with the available functions."""
    functions_desc = []

    for fn in functions:
        params = ", ".join(
            f"{name}:{param.type}" for name, param in fn.parameters.items()
        )
        functions_desc.append(f"{fn.name}({params}) -> {fn.returns.type}")

    joined = "\n".join(functions_desc)

    return f"""
You are a function calling assistant.

Available functions:
{joined}

You must choose the most appropriate function and extract arguments.

Return ONLY valid JSON in this format:
{{
    "name": "function_name",
    "parameters": {{
        "arg1": value
    }}
}}
""".strip()


def build_full_prompt(system_prompt: str, user_prompt: str) -> str:
    """Combine the system prompt with the user request."""
    return f"{system_prompt}\n\nUser request:\n{user_prompt}\n"
