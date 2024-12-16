#!/bin/bash

# Get a list of staged Python files
staged_files=$(git diff --name-only --cached -- '*.py')
echo $staged_files

# Run flake8 to check for syntax and style issues on staged Python files
# Use an array to handle spaces in filenames
if [ -n "$staged_files" ]; then
    flake8_files=($staged_files)
    flake8 "${flake8_files[@]}" --max-line-length=121
    if [ $? -ne 0 ]; then
        echo "Linting issues found in staged files. Please fix them before committing."
        exit 1
    fi
fi
