import numpy as np
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict


def get_repository_files() -> List[str]:
    """Get all files tracked in the git repository"""
    try:
        result = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        )
        return result.stdout.splitlines()
    except subprocess.CalledProcessError:
        print("Error: Failed to list git files")
        return []


def categorize_files(files: List[str]) -> Dict[str, List[str]]:
    """Categorize files by type for different pattern generation"""
    categories = {
        "code": [],  # Python, JavaScript, etc.
        "docs": [],  # Markdown, RST, etc.
        "data": [],  # JSON, YAML, etc.
        "config": [],  # Configuration files
        "other": [],  # Other file types
    }

    for file in files:
        ext = Path(file).suffix.lower()
        if ext in [".py", ".js", ".cpp", ".h", ".c"]:
            categories["code"].append(file)
        elif ext in [".md", ".rst", ".txt"]:
            categories["docs"].append(file)
        elif ext in [".json", ".yaml", ".yml"]:
            categories["data"].append(file)
        elif ext in [".toml", ".ini", ".cfg"]:
            categories["config"].append(file)
        else:
            categories["other"].append(file)

    return categories


def build_word_transitions(text: str) -> Dict[str, List[str]]:
    """Build word transition probabilities from text"""
    words = text.split()
    transitions = {}

    for i in range(len(words) - 1):
        current = words[i].lower()
        next_word = words[i + 1]
        if current not in transitions:
            transitions[current] = []
        transitions[current].append(next_word)

    return transitions


def build_word_knowledge(text: str) -> Dict[str, np.ndarray]:
    """Build semantic vectors for words based on their context and code structure"""
    words = text.lower().split()
    word_vectors = {}
    window_size = 5

    # Track code context
    in_code_block = False
    code_depth = 0
    code_context = set()

    for i, word in enumerate(words):
        # Detect code context
        if word in ["def", "class", "import", "from"] or "{" in word or "(" in word:
            in_code_block = True
            code_depth += 1
        if "}" in word or ")" in word:
            code_depth -= 1
            if code_depth <= 0:
                in_code_block = False
                code_depth = 0

        # Add word with its context
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        context = words[start:i] + words[i + 1 : end]

        if word not in word_vectors:
            word_vectors[word] = {}

        # Count context words with code awareness
        for context_word in context:
            if context_word not in word_vectors[word]:
                word_vectors[word][context_word] = 0
            # Weight code context differently
            if in_code_block:
                word_vectors[word][context_word] += 0.8  # Code context
                code_context.add(word)
            else:
                word_vectors[word][context_word] += 1.0  # Natural language context

    # Convert to normalized vectors with code awareness
    all_context_words = list(
        set(sum([list(v.keys()) for v in word_vectors.values()], []))
    )
    vector_size = len(all_context_words) + 1  # +1 for code context flag
    word_context_map = {word: idx for idx, word in enumerate(all_context_words)}

    normalized_vectors = {}
    for word, contexts in word_vectors.items():
        vector = np.zeros(vector_size)
        for context_word, count in contexts.items():
            if context_word in word_context_map:
                vector[word_context_map[context_word]] = count
        # Add code context flag
        vector[-1] = 1.0 if word in code_context else 0.0

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        normalized_vectors[word] = vector

    return normalized_vectors


def process_repository_content() -> tuple[str, Dict[str, List[str]], Dict[str, np.ndarray]]:
    """Process repository content and build knowledge structures"""
    print("Scanning repository for pattern sources...")
    all_files = get_repository_files()
    categorized_files = categorize_files(all_files)

    # Build word knowledge from repository content
    print("Building semantic understanding...")
    all_text = ""
    for category, files in categorized_files.items():
        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    all_text += f.read() + " "
            except Exception:
                continue

    word_vectors = build_word_knowledge(all_text)
    word_transitions = build_word_transitions(all_text)

    return all_text, word_transitions, word_vectors 