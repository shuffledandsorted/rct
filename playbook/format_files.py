#!/usr/bin/env python3

import subprocess
import os

def format_file(file_path):
    # Use the fmt command to wrap lines at 120 characters
    with open(file_path, 'r') as file:
        content = file.read()

    # Use fmt via subprocess
    result = subprocess.run(['fmt', '-w', '120'], input=content, text=True, capture_output=True)
    formatted_content = result.stdout

    # Write the formatted content back to the file
    with open(file_path, 'w') as file:
        file.write(formatted_content)

def main():
    # Get the list of staged files
    result = subprocess.run(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'], capture_output=True, text=True)
    files = result.stdout.splitlines()

    # Filter for .txt and .md files
    for file in files:
        if file.endswith('.txt') or file.endswith('.md'):
            format_file(file)
            # Add the formatted file back to the staging area
            subprocess.run(['git', 'add', file])

if __name__ == "__main__":
    main() 