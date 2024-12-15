#!/usr/bin/env python3

import os
import time
import shutil

REPO_DIR = "."
ARCHIVE_DIR = "archive"

# TBD calculate datetime object and check year and month
STALE_DAYS = 30  # Threshold for archiving
ONE_YEAR_DAYS = 365  # Days in a year


def get_modification_info(path):
    """Get the age and last modification time of a file or folder."""
    last_mod_time = os.path.getmtime(path)
    now = time.time()
    diff_days = (now - last_mod_time) / 86400.0
    return diff_days, last_mod_time


def move_old_month_folders():
    for month_dir in os.listdir(os.path.join(REPO_DIR, ARCHIVE_DIR)):
        month_path = os.path.join(REPO_DIR, ARCHIVE_DIR, month_dir)
        if os.path.isdir(month_path):
            # Check if the directory is a month folder
            try:
                time.strptime(month_dir, "%Y-%m")
                # Move to year/month directory if older than a year
                age, last_mod_time = get_modification_info(month_path)
                if age > ONE_YEAR_DAYS:
                    year_month = month_dir
                    year, month = year_month.split("-")
                    year_dir = os.path.join(REPO_DIR, ARCHIVE_DIR, year)
                    os.makedirs(year_dir, exist_ok=True)
                    shutil.move(month_path, os.path.join(year_dir, month))
            except ValueError:
                continue


def get_files_and_folders_to_move():
    """Determine which files and folders need to be moved based on their age."""
    items_to_move = []
    for root, dirs, files in os.walk(REPO_DIR):
        # Skip certain directories
        if '.git' in root or ARCHIVE_DIR in root:
            continue
        for name in files + dirs:
            path = os.path.join(root, name)
            age, last_mod_time = get_modification_info(path)
            if age > STALE_DAYS:
                items_to_move.append((path, age, last_mod_time))
    return items_to_move


def move_item(path, age, last_mod_time):
    """Move the file or folder to the appropriate directory based on its age."""
    # Extract year and month from last modification time
    last_mod_time_struct = time.gmtime(last_mod_time)
    year = str(last_mod_time_struct.tm_year)
    month = str(last_mod_time_struct.tm_mon).zfill(2)
    if age < ONE_YEAR_DAYS:
        # Place in month directory
        target_dir = os.path.join(REPO_DIR, ARCHIVE_DIR, month)
    else:
        # Place in year/month directory
        target_dir = os.path.join(REPO_DIR, ARCHIVE_DIR, year, month)
    os.makedirs(target_dir, exist_ok=True)
    dest_path = os.path.join(target_dir, os.path.basename(path))
    print(f"Archiving {path} -> {dest_path}")
    shutil.move(path, dest_path)


def main():
    os.makedirs(os.path.join(REPO_DIR, ARCHIVE_DIR), exist_ok=True)
    move_old_month_folders()
    items_to_move = get_files_and_folders_to_move()
    for path, age, last_mod_time in items_to_move:
        move_item(path, age, last_mod_time)


if __name__ == "__main__":
    main()
