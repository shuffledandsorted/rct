#!/usr/bin/env python3

import os
import shutil

REPO_DIR = "."
MAX_LINES = 200
ARCHIVE_DIR = "misc"
# Proposal: it's like I need a simpler seed of me to do this.
# I would propose doing a frequency analysis of the files and
# do clustering. The cluster's name is the label name.
# For the divergent, we just leave them for now I think until
# the pattern emerges. If we are desperate for fewer things
# to be in that space, we could throw them in a misc directory
# or...even combine them if they are small files?
# The very nature of their instability is interesting in that
# it might be some new pattern emerging, or it's just the
# leaky nature of any organizational pattern.

def split_by_heading(lines):
    sections = []
    current_section = []
    current_title = None

    def flush_section():
        if current_section:
            sections.append((current_title, current_section[:]))

    for line in lines:
        if line.startswith("# "):
            # New section
            flush_section()
            current_section = [line]
            current_title = line.strip("# ").strip()
        else:
            current_section.append(line)
    flush_section()
    return sections

def clean_filename(title):
    safe_name = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).strip()
    if not safe_name:
        safe_name = "untitled"
    return safe_name.lower().replace(' ', '_') + ".md"

def main():
    os.makedirs(os.path.join(REPO_DIR, ARCHIVE_DIR), exist_ok=True)

    for root, dirs, files in os.walk(REPO_DIR):
        if '.git' in root or ARCHIVE_DIR in root:
            continue
        for f in files:
            if f.endswith(".md"):
                filepath = os.path.join(root, f)
                with open(filepath, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()
                
                if len(lines) > MAX_LINES:
                    # Split this file
                    sections = split_by_heading(lines)
                    if len(sections) > 1:
                        # Move original large file to archive
                        rel_path = os.path.relpath(filepath, REPO_DIR)
                        archived_path = os.path.join(REPO_DIR, ARCHIVE_DIR, f)
                        print(f"Archiving original large file: {rel_path} -> {archived_path}")
                        shutil.move(rel_path, archived_path)

                        # Write out new split files
                        base_name = os.path.splitext(f)[0]
                        for i, (title, sec_lines) in enumerate(sections, start=1):
                            fname = clean_filename(title)
                            # Avoid name collisions by adding index if needed
                            if fname == "untitled.md":
                                fname = f"{base_name}_part{i}.md"

                            outpath = os.path.join(root, fname)
                            print(f"Creating {outpath}")
                            with open(outpath, "w", encoding="utf-8") as out:
                                out.writelines(sec_lines)

if __name__ == "__main__":
    main()
