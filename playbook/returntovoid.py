#!/usr/bin/env python3
"""Looks for files older than one year and deletes them."""

import os
import subprocess
import time


REPO_DIR = "."
ONE_YEAR_DAYS = 365


def days_since_last_modification(file_path):
    cmd = ["git", "log", "-1", "--pretty=format:%ct", "--", file_path]
    try:
        timestamp = subprocess.check_output(cmd, cwd=REPO_DIR, stderr=subprocess.DEVNULL).decode().strip()
        if timestamp:
            last_commit_time = int(timestamp)
            now = int(time.time())
            diff_days = (now - last_commit_time) / 86400.0
            return diff_days
    except subprocess.CalledProcessError:
        return None
    return None


def main():
    for root, dirs, files in os.walk(REPO_DIR):
        if '.git' in root:
            continue
        for f in files:
            if f.endswith(".md") or f.endswith(".txt"):
                rel_path = os.path.relpath(os.path.join(root, f), REPO_DIR)
                age = days_since_last_modification(rel_path)
                if age is not None and age > ONE_YEAR_DAYS:
                    # Delete the file
                    print(f"Deleting old file: {rel_path}")
                    os.remove(rel_path)


if __name__ == "__main__":
    main()
