#!/usr/bin/env python3
"""
Script to filter out test cases that don't have flash algorithm in performance log
"""

import re
import sys

def filter_flash_cases(input_file):
    """
    Read the log file and filter out cases that don't contain '(flash)' algorithm
    """
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split content into lines
    lines = content.split('\n')
    
    # Find the header part (before first test case)
    header_lines = []
    test_case_start_idx = 0
    
    for i, line in enumerate(lines):
        if '----' in line and 'B=' in line and 'H=' in line:
            test_case_start_idx = i
            break
        header_lines.append(line)
    
    # Parse test cases
    filtered_lines = header_lines[:]
    i = test_case_start_idx
    
    while i < len(lines):
        # Find start of test case
        if '----' in lines[i] and 'B=' in lines[i] and 'H=' in lines[i]:
            case_start = i
            case_lines = [lines[i]]  # Include the header line
            i += 1
            
            # Read until we find the end separator or another case starts
            has_flash = False
            while i < len(lines):
                line = lines[i]
                case_lines.append(line)
                
                # Check if this line contains flash algorithm
                if '(flash)' in line:
                    has_flash = True
                
                # Check if we've reached the end separator
                if '------------------------------------------------------------------------------------------------------------------------------------------------------' in line:
                    i += 1
                    break
                    
                # Check if we've reached another test case (fallback)
                if i + 1 < len(lines) and '----' in lines[i + 1] and 'B=' in lines[i + 1]:
                    i += 1
                    break
                    
                i += 1
            
            # Only add this case if it has flash
            if has_flash:
                filtered_lines.extend(case_lines)
            else:
                print(f"Removing case: {lines[case_start].strip()}")
        else:
            i += 1
    
    return '\n'.join(filtered_lines)

def main():
    input_file = "cases/fa/leetcuda_l20/l20_naive_perf_wfa.log"
    
    print(f"Processing {input_file}...")
    
    try:
        filtered_content = filter_flash_cases(input_file)
        
        # Write back to the same file
        with open(input_file, 'w') as f:
            f.write(filtered_content)
            
        print(f"Successfully filtered {input_file}")
        print("Cases without flash algorithm have been removed.")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
