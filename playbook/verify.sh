#!/bin/bash

# Run flake8 to check for syntax and style issues on all Python files
flake8 . --max-line-length=121
if [ $? -ne 0 ]; then
    echo "Linting issues found. Please fix them before committing."
    exit 1
fi 
