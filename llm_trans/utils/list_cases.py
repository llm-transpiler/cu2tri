from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable

from ..services.cases import resolve_cases_for_testset


def _project_root() -> Path:
    env = os.getenv("PROJECT_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def _abs_folders(testset: str, mapping: Dict[str, Iterable[str]]) -> Dict[str, list[str]]:
    root = _project_root()
    cases_root = root / Path(f"llm_trans/cases/{testset}")

    result: Dict[str, list[str]] = {}
    for op, names in mapping.items():
        abs_list = [str((cases_root / name).resolve()) for name in names]
        result[op] = abs_list
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="List test case folders for a testset")
    parser.add_argument("--testset", default="xpiler", help="Testset name (default: xpiler)")
    parser.add_argument(
        "--case-types",
        nargs="*",
        default=None,
        help="Optional list of operator types to include (default: all)",
    )
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="Only include the first case per operator (default: False)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "folders", "bash"],
        default="json",
        help="Output format: json mapping, raw folder lines, or bash assignments",
    )
    args = parser.parse_args(argv)

    all_cases = resolve_cases_for_testset(args.testset)
    if args.case_types:
        all_cases = {k: v for k, v in all_cases.items() if k in set(args.case_types)}
    if args.first_only:
        all_cases = {k: [v[0]] for k, v in all_cases.items() if v}

    folders = _abs_folders(args.testset, all_cases)

    if args.format == "json":
        print(json.dumps(folders, ensure_ascii=False, indent=2))
        return
    if args.format == "folders":
        for op, items in folders.items():
            for folder in items:
                print(folder)
        return
    if args.format == "bash":
        # Emit associative array assignments compatible with: declare -A case_types; eval $(python -m ... --format bash)
        for op, items in folders.items():
            joined = "\n".join(items).replace("'", "'\\''")
            # Note: no trailing ')' here; assignment to array element does not need parentheses
            print(f"case_types['{op}']=$'" + joined + "'")
        return


if __name__ == "__main__":
    main()


