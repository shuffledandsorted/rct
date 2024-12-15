#!/usr/bin/env python3
"""
normalize.py

This module formats and validates text and markdown files to ensure a consistent and sane codebase state.
It is used as a pre-commit hook to automatically process files before committing.

Related Files:
- .git/hooks/pre-commit: Invokes this script.

Functions:
- format_file(file_path): Formats a file and updates the Git index.
- main(): Processes all staged files.
"""

import subprocess
import os


def format_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    # Format the content
    formatted_content = subprocess.run(
        ["fmt", "-w", "120"], input=content, text=True, capture_output=True
    ).stdout

    # Create a blob from the formatted content and get its hash
    blob_hash = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        input=formatted_content,
        text=True,
        capture_output=True,
    ).stdout.strip()

    # Update the index with the blob hash for the original file
    subprocess.run(
        ["git", "update-index", "--cacheinfo", "100644", blob_hash, file_path],
        cwd=os.path.dirname(file_path),
    )


def main():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
    )
    files = result.stdout.splitlines()

    for file in files:
        if file.endswith(".txt") or file.endswith(".md"):
            format_file(file)
            subprocess.run(["git", "add", file])


if __name__ == "__main__":
    main()
