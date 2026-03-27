# call-me-maybe

This project has been created as part of the 42 curriculum by `pdiniz-l`.

## Description

**call-me-maybe** is a Python project that converts natural-language prompts into structured function calls using a constrained decoding approach for small LLMs.

The program reads:

- a JSON file containing function definitions,
- a JSON file containing prompt inputs,

and generates a final JSON file where each prompt is represented as one structured output object.

The main goal of the project is to build a reliable function-calling pipeline that is:

- structured,
- safe,
- predictable,
- and robust against ambiguous or incomplete prompts.

## Project Goal

The purpose of the project is to simulate an LLM function-calling workflow.

Given a set of available functions and a batch of user prompts, the program must determine:

1. which function best matches each prompt,
2. which parameters can be extracted,
3. how to serialize the result into a clean JSON output.

The project prioritizes correctness and output consistency over aggressive guessing.

## Repository Structure

```text
.
├── .gitignore
├── LICENSE
├── Makefile
├── pyproject.toml
├── uv.lock
├── llm_sdk/
│   └── __init__.py
├── data/
│   └── input/
│       ├── function_calling_tests.json
│       └── functions_definition.json
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── errors.py
│   ├── file_io/
│   │   ├── __init__.py
│   │   ├── reader.py
│   │   └── writer.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── constrained_decoder.py
│   │   ├── prompt_builder.py
│   │   └── vocabulary.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── function_call_result.py
│   │   ├── function_definition.py
│   │   ├── prompt_input.py
│   │   └── selection_result.py
│   └── services/
│       ├── __init__.py
│       ├── argument_extractor.py
│       ├── function_selector.py
│       ├── parameter_extractor.py
│       ├── pipeline.py
│       └── schema_validator.py
└── tests/
    ├── test_function_selector.py
    ├── test_output_contract.py
    ├── test_parameter_extractor.py
    └── test_pipeline.py
```

## Installation

### Requirements

- Python 3.10 or higher
- `uv`
- `make`

### Dependency Installation

This project uses `uv` for dependency management.

Install dependencies with:

```bash
make install
```

The `Makefile` installs the virtual environment into:

```text
/goinfre/$(USER)/call_me_maybe_venv
```

It also configures Hugging Face caches under:

```text
/goinfre/$(USER)/.hf
```

This setup is useful in environments where the home directory has limited storage.

## Instructions

### Run the Project

```bash
make run
```

By default, the project uses:

- `data/input/functions_definition.json`
- `data/input/function_calling_tests.json`

and writes the output to:

- `data/output/function_calling_results.json`

### Run the Tests

```bash
make test
```

### Run Linting

```bash
make lint
```

### Clean Generated Files

```bash
make clean
```

## Input and Output

### Function Definitions

The program expects a JSON file containing the available function definitions.

Each function definition includes information such as:

- name,
- description,
- parameters,
- expected return type.

### Prompt Input

The program also expects a JSON file containing input prompts.

Example:

```json
[
  { "prompt": "Add 2 and 3" },
  { "prompt": "Reverse \"banana\"" },
  { "prompt": "Greet Alice" }
]
```

### Output Format

The output is a JSON array containing one object per input prompt.

Example:

```json
[
  {
    "prompt": "Add 2 and 3",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Add two numbers",
    "name": null,
    "parameters": {}
  }
]
```

Each output object always contains exactly these fields:

- `prompt`
- `name`
- `parameters`

## Resources

This project relies on concepts from:

- constrained decoding,
- structured generation,
- function calling with LLMs,
- JSON-based interfaces,
- schema validation,
- automated testing,
- static analysis.

Relevant learning resources include:

- Python official documentation
- Pydantic documentation
- JSON documentation
- general articles and documentation about function calling in LLMs
- documentation for the tooling used in the project, such as `pytest`, `flake8`, and `uv`

## Algorithm Explanation

The project uses a constrained decoding strategy to guide the LLM toward valid structured outputs.

The full process is:

1. Load the function definitions from the input JSON file.
2. Load the prompts from the input prompt file.
3. For each prompt, invoke the function selection layer.
4. The selector asks the constrained decoder to generate a candidate function call.
5. The generated candidate is validated before being accepted.
6. If the result is valid, the function name and extracted parameters are returned.
7. If the prompt is ambiguous, incomplete, or invalid, the system preserves the prompt and emits a null function call instead.
8. The pipeline writes one final output object for every prompt.

The final behavior is intentionally conservative.

If the system cannot safely determine a valid function call, it returns:

```json
{
  "prompt": "...",
  "name": null,
  "parameters": {}
}
```

This avoids hallucinated calls and keeps the output predictable.

## Design Decisions

### 1. One Output Object per Prompt

The pipeline preserves the batch structure by generating exactly one output object per input prompt.

This makes the final output easier to inspect, test, and compare against the input.

### 2. Separation Between Internal and External Contracts

Internally, the project uses a richer structure for selection results, including error information.

Externally, the final JSON output remains minimal and clean, containing only:

- `prompt`
- `name`
- `parameters`

This keeps internal debugging concerns separate from the required public output format.

### 3. Safe Handling of Unresolved Prompts

When the system cannot determine a valid function or cannot extract enough required parameters, it does not invent a function call.

Instead, it returns:

```json
{
  "prompt": "...",
  "name": null,
  "parameters": {}
}
```

This decision favors reliability over unsafe guessing.

### 4. Validation at the Selection Boundary

The selector validates decoder output before sending it to the pipeline.

It checks that:

- the decoder returned a dictionary,
- the function name is either a string or `null`,
- the parameters field is a dictionary.

This prevents malformed decoder outputs from contaminating the final result.

### 5. Robust Batch Behavior

A single invalid prompt must not abort the entire pipeline.

The project is designed so that problematic prompts are represented safely while the rest of the batch continues normally.

## Performance Analysis

### Accuracy

For direct and well-formed prompts, the system correctly selects the intended function and extracts the required parameters.

Examples include:

- arithmetic prompts,
- string reversal prompts,
- greeting prompts,
- square root prompts,
- regex substitution prompts.

### Reliability

The solution is robust against:

- incomplete prompts,
- ambiguous prompts,
- malformed decoder outputs,
- runtime exceptions during selection.

Instead of crashing or producing malformed JSON, the pipeline emits a safe null call.

### Speed

The pipeline itself is lightweight.

Most of the complexity is concentrated in the decoding and validation steps.
For typical project-sized batches, execution is efficient and well suited to repeated local testing.

### Output Consistency

One of the strongest aspects of the implementation is output consistency.

Every input prompt generates one output object, which simplifies testing and downstream usage.

## Challenges Faced

### 1. Ambiguous and Incomplete Prompts

Some prompts clearly indicated an intent but did not provide enough information to extract all required parameters.

A key challenge was deciding whether to:

- omit those prompts,
- force a guess,
- or preserve them safely in the output.

The chosen solution was to preserve them with `name: null` and empty parameters.

### 2. Distinguishing Internal Errors from Final Output

The project needed good internal observability without polluting the final JSON contract.

This was solved by separating:

- the internal `SelectionResult`,
- the external `FunctionCallResult`.

### 3. Decoder Robustness

The decoder could theoretically return malformed data.

To solve this, the selection layer validates the returned structure before accepting it.

### 4. Batch Stability

Another challenge was ensuring that a single bad prompt would not interrupt the entire batch.

The final design guarantees that the pipeline continues processing every prompt.

## Testing Strategy

The project was validated with automated tests and linting.

### Unit Tests

Unit tests cover:

- valid function selection,
- correct parameter extraction,
- incomplete prompts,
- ambiguous prompts,
- preservation of prompt text,
- selector exception handling,
- malformed decoder responses.

### Pipeline Tests

Pipeline-level tests verify:

- input loading,
- prompt iteration,
- result transformation,
- one output object per prompt,
- correct final JSON contract.

### Output Contract Tests

Dedicated contract checks verify that final output objects always contain the expected public fields and that unresolved prompts are represented safely.

### Static Analysis

Linting is used to maintain code quality and consistency across the codebase.

## Example Usage

### Example Input

```json
[
  { "prompt": "Add 2 and 3" },
  { "prompt": "Reverse \"banana\"" },
  { "prompt": "Greet Alice" },
  { "prompt": "Add two numbers" }
]
```

### Example Output

```json
[
  {
    "prompt": "Add 2 and 3",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Reverse \"banana\"",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "banana"
    }
  },
  {
    "prompt": "Greet Alice",
    "name": "fn_greet",
    "parameters": {
      "name": "Alice"
    }
  },
  {
    "prompt": "Add two numbers",
    "name": null,
    "parameters": {}
  }
]
```

### Example Commands

```bash
make install
make lint
make test
make run
```

## Technical Summary

In summary, this project implements a structured LLM function-calling pipeline with:

- constrained decoding,
- function selection,
- parameter extraction,
- output validation,
- safe fallback behavior,
- clean JSON serialization,
- stable batch processing.

The result is a robust and predictable system that maps natural-language prompts to structured function calls while handling unresolved prompts safely.
