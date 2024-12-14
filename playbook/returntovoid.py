#!/usr/bin/env python3

import os
import subprocess
import time
import shutil

# We don't need to hold on to everything that comes across our plate.
# Let things go. We have a memory, and nothing is truly ever gone.
# But at least in this way of organizing ourselves into tidy messes,
# it's time to say goodbye.
REPO_DIR = "."
# Naw, I think repository history and our deep storage are similar.
# The true place where we let go I suppose is when we start a new
# repository. It's a bit of a manual process to figure out what to
# bring forward then. Heurestics won't save us.
ARCHIVE_DIR = "archive/deep_storage"
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
    os.makedirs(os.path.join(REPO_DIR, ARCHIVE_DIR), exist_ok=True)
    for root, dirs, files in os.walk(REPO_DIR):
        if '.git' in root or ARCHIVE_DIR in root:
            continue
        for f in files:
            if f.endswith(".md") or f.endswith(".txt"):
                rel_path = os.path.relpath(os.path.join(root, f), REPO_DIR)
                age = days_since_last_modification(rel_path)
                if age is not None and age > ONE_YEAR_DAYS:
                    # Move to deep archive
                    dest_path = os.path.join(REPO_DIR, ARCHIVE_DIR, f)
                    print(f"Moving old file to deep storage: {rel_path} -> {dest_path}")
                    shutil.move(rel_path, dest_path)

if __name__ == "__main__":
    main()
