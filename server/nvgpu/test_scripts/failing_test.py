#!/usr/bin/env python3
"""
Test script that intentionally fails.
Used for testing error handling in NVGPU server.
"""
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Failing test script')
    parser.add_argument('--error-code', type=int, default=1, help='Exit code to return')
    parser.add_argument('--error-message', default='Test failure', help='Error message to print')
    args = parser.parse_args()
    
    print("Starting test that will fail...")
    print(f"This test will exit with code {args.error_code}")
    
    # Print error message to stderr
    print(f"ERROR: {args.error_message}", file=sys.stderr)
    
    # Exit with specified error code
    sys.exit(args.error_code)

if __name__ == "__main__":
    main()

