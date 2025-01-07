from typing import Dict, Optional, Tuple
import time
import os
import numpy as np
from watchdog.events import FileSystemEventHandler
from ..quantum.utils import text_to_quantum_pattern


def _process_file(args) -> Optional[Tuple[str, str, np.ndarray]]:
    """Process a single file and return (name, content, pattern) or None if error"""
    file_path, dims = args
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        function_name = os.path.splitext(os.path.basename(file_path))[0]
        pattern = text_to_quantum_pattern(content, dims)
        return function_name, content, pattern
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


class FileChangeHandler(FileSystemEventHandler):
    """Handle file system events for repository consciousness"""

    def __init__(self, consciousness):
        self.consciousness = consciousness
        self.last_modified: Dict[str, float] = {}
        self.debounce_time = 1.0  # Seconds to wait before processing changes

    def on_modified(self, event):
        if not event.is_directory:
            file_path = str(event.src_path)  # Ensure string type
            if file_path.endswith(".py"):
                current_time = time.time()
                last_time = self.last_modified.get(file_path, 0)

                # Debounce to avoid processing the same file multiple times
                if current_time - last_time > self.debounce_time:
                    self.last_modified[file_path] = current_time
                    print(f"\n[WATCHER] Change detected in {file_path}")
                    self.consciousness.process_file_change(file_path)
