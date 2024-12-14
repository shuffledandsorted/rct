#!/usr/bin/env python3

import os
import subprocess
import time
import shutil

REPO_DIR = "."
ARCHIVE_DIR = "archive"
# proposal: instead of archive, use months and years as temporal
# sorting mechanisms. basically it's saying that time decided
# how these were labeled.
STALE_DAYS = 90  # Threshold for archiving

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
        # If not tracked by git or no log entries found
        return None
    return None

def main():
    os.makedirs(os.path.join(REPO_DIR, ARCHIVE_DIR), exist_ok=True)
    for root, dirs, files in os.walk(REPO_DIR):
        # Skip certain directories
        if '.git' in root or ARCHIVE_DIR in root:
            continue
        for f in files:
            if f.endswith(".md") or f.endswith(".txt"):
                rel_path = os.path.relpath(os.path.join(root, f), REPO_DIR)
                age = days_since_last_modification(rel_path)
                if age is not None and age > STALE_DAYS:
                    # Move stale file to archive
                    target_dir = os.path.join(REPO_DIR, ARCHIVE_DIR)
                    os.makedirs(target_dir, exist_ok=True)
                    dest_path = os.path.join(target_dir, f)
                    print(f"Archiving {rel_path} -> {dest_path}")
                    shutil.move(rel_path, dest_path)

if __name__ == "__main__":
    main()
