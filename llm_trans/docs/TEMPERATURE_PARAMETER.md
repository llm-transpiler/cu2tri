# Temperature Parameter Support

## Overview
Added temperature parameter support to the LLM translation system, allowing users to control the randomness of model generation.

## Changes Made

### 1. CLI Argument Support
- **File**: `cu2til/llm_trans/config/args.py`
- **Change**: Added `--temperature` argument with default value of 0.3
- **Usage**: `--temperature 1.0` to set temperature to 1.0

### 2. API Parameter Passing
- **Files**:
  - `cu2til/llm_trans/services/attempts.py`
  - `cu2til/llm_trans/services/testing.py`
- **Change**: Modified `get_api_param()` calls to pass temperature parameter to model clients
- **Implementation**: Extract temperature from args and pass as extra kwargs

## Usage Examples

### Set temperature to 1.0
```bash
python -m llm_trans --temperature 1.0 --model gpt_5_mini
```

### Set temperature to 0.7 (more conservative)
```bash
python -m llm_trans --temperature 0.7 --model gpt_5_mini
```

### Use default temperature (0.3)
```bash
python -m llm_trans --model gpt_5_mini
```

## Temperature Values
- **Default**: 0.3 (as requested)
- **Range**: 0.0 to 2.0 (typical for most models)
- **Effect**:
  - Lower values (0.1-0.3): More deterministic, focused outputs
  - Higher values (0.7-1.0+): More creative, varied outputs

## Notes
- Temperature parameter is passed to all model APIs that support it
- If a model doesn't support temperature, the parameter will be ignored
- The parameter works with all supported model providers (OpenAI, OpenRouter, local endpoints, etc.)