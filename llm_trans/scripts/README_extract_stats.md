# Stats Extraction Script Usage

The script `extract_stats.py` is designed to extract statistics from xpiler JSONL files and organize them in the stats directory with the same structure as the runs directory.

## Features

- Extracts metadata and case information from xpiler JSONL files
- Organizes cases by name and sorts attempts by number (ascending)
- Supports filtering by maximum attempt number and round number
- Maintains the same directory structure as runs/
- Adds appropriate suffixes to output filenames based on filters used
- Copies metadata configuration from the original JSONL files

## Usage

```bash
# Basic usage - extract all data from a single file
python3 extract_stats.py /path/to/xpiler.jsonl

# Extract from a directory (will process all *_xpiler.jsonl files)
python3 extract_stats.py /path/to/directory/

# Filter by maximum attempt number (only include attempts 1-3)
python3 extract_stats.py /path/to/xpiler.jsonl --max-attempts 3

# Filter by maximum round number (only include rounds 1-2)
python3 extract_stats.py /path/to/xpiler.jsonl --max-rounds 2

# Combine both filters
python3 extract_stats.py /path/to/xpiler.jsonl --max-attempts 5 --max-rounds 2

# Specify custom output directory
python3 extract_stats.py /path/to/xpiler.jsonl --output-dir /custom/output/path
```

## Output

The script generates JSON files in the stats directory with the following structure:

```
stats/
└── cu2tri/
    └── xpiler/
        └── {model_name}/
            └── {timestamp}/
                ├── {model}_xpiler.json              # All data
                ├── {model}_xpiler_at3.json          # Max 3 attempts
                └── {model}_xpiler_at5_r2.json       # Max 5 attempts and 2 rounds
```

## Output Format

The generated JSON files contain:

1. **metadata**: Original metadata from the JSONL file plus filter information
2. **summary**: Summary statistics (total cases, case types, total attempts)
3. **cases**: Detailed case information organized by case name with attempts sorted by number

Each case includes:
- case_type and case_name
- total_attempts and successful_attempts count
- attempts array with full details for each attempt

## Examples

```bash
# Process a specific file with max 3 attempts
python3 extract_stats.py /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_oss_120b/20251107_154318/gpt_oss_120b_xpiler.jsonl --max-attempts 3

# Process all files in a directory
python3 extract_stats.py /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_5_mini/20251024_065400/

# Process with both attempt and round limits
python3 extract_stats.py /data/apps/project/cu2tri/cu2til/llm_trans/runs/cu2tri/xpiler/gpt_oss_120b/20251107_154318/gpt_oss_120b_xpiler.jsonl --max-attempts 5 --max-rounds 2
```

The script ensures proper sorting of cases (natural sorting) and attempts (numerical order), making the output consistent and easy to analyze.