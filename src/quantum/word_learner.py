from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import random
import os
from ..quantum.utils import text_to_quantum_pattern, quantum_normalize


@dataclass
class QuantumWord:
    """Represents a word in the quantum system"""

    word: str
    amplitude: complex
    phase: float
    semantic_vector: np.ndarray


@dataclass
class QuantumFile:
    """Represents a file in the quantum system"""

    path: str
    state: np.ndarray  # Quantum state of file
    last_modified: float  # Last modification time
    word_states: Dict[str, np.ndarray]  # States of words in this file


@dataclass
class CodePattern:
    """Represents a hierarchical code pattern"""

    pattern_type: str  # 'function', 'class', 'block', etc.
    content: str
    children: List["CodePattern"]
    parent: Optional["CodePattern"] = None
    quantum_state: Optional[np.ndarray] = None


class QuantumWordLearner:

    def __init__(self, dims: Tuple[int, int], embedding_dim: int = 50):
        """Initialize the quantum word learner

        Args:
            dims: Tuple of dimensions for quantum patterns (required)
            embedding_dim: Dimension of word embeddings
        """
        self.dims = dims
        self.embedding_dim = embedding_dim
        self.vocabulary: Dict[str, np.ndarray] = {}  # Word embeddings
        self.patterns: Dict[str, np.ndarray] = {}  # Quantum patterns
        self.code_patterns: Dict[str, CodePattern] = {}  # Hierarchical code patterns
        self.hamiltonian: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenstates: Optional[np.ndarray] = None
        self.recent_words: List[str] = []
        self.context_window = 5
        self.file_states: Dict[str, QuantumFile] = {}  # Quantum states of files

    def add_word(self, word: str, embedding: np.ndarray) -> None:
        """Add a word to the quantum system with its embedding"""
        # Normalize embedding
        normalized_embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        self.vocabulary[word] = normalized_embedding

        # Create quantum pattern if not exists
        if word not in self.patterns:
            pattern = np.random.randn(*self.dims) + 1j * np.random.randn(*self.dims)
            pattern = pattern / np.linalg.norm(pattern)
            self.patterns[word] = pattern

        # Add to recent words
        self.recent_words.append(word)
        if len(self.recent_words) > self.context_window:
            self.recent_words = self.recent_words[-self.context_window :]

        # Mark Hamiltonian as needing rebuild
        self.hamiltonian = None

    def _build_hamiltonian(self) -> None:
        """Construct Hamiltonian matrix from word relationships"""
        n = len(self.vocabulary)
        H = np.zeros((n, n), dtype=complex)

        words = list(self.vocabulary.keys())
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i == j:
                    # Diagonal elements from pattern energy
                    H[i, i] = np.abs(self.patterns[word1].mean())
                else:
                    # Off-diagonal elements from word similarity and pattern interaction
                    embedding_sim = np.dot(
                        self.vocabulary[word1], self.vocabulary[word2]
                    )
                    pattern_interaction = np.vdot(
                        self.patterns[word1], self.patterns[word2]
                    )
                    H[i, j] = embedding_sim * pattern_interaction

        self.hamiltonian = H
        # Calculate eigenvalues and eigenstates
        self.eigenvalues, self.eigenstates = np.linalg.eigh(H)

    def evolve_state(self, time_steps: int = 100) -> np.ndarray:
        """Evolve the quantum state through time"""
        if self.hamiltonian is None:
            self._build_hamiltonian()

        assert self.hamiltonian is not None  # Type assertion for mypy

        initial_state = np.ones(len(self.vocabulary)) / np.sqrt(len(self.vocabulary))
        dt = 0.01  # Smaller time step for stability
        current_state = initial_state

        for t in range(time_steps):
            # Evolve and normalize at each step to prevent overflow
            U = np.exp(-1j * dt * self.hamiltonian)
            current_state = U @ current_state
            # Normalize to prevent overflow
            norm = np.linalg.norm(current_state)
            if norm > 0:
                current_state = current_state / norm

        return current_state

    def _normalize_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Helper to ensure valid probability distribution"""
        # Handle NaN and negative values
        probabilities = np.nan_to_num(probabilities, nan=0.0)
        probabilities = np.abs(probabilities)  # Ensure non-negative

        # Handle zero sum
        if np.sum(probabilities) < 1e-10:
            probabilities = np.ones(len(probabilities))

        # First normalization
        probabilities = probabilities / np.sum(probabilities)

        # Handle numerical instability
        probabilities = np.maximum(
            probabilities, 0
        )  # Ensure no negative values from floating point
        probabilities = probabilities / np.sum(probabilities)  # Renormalize

        # Final safety check
        if not np.all(np.isfinite(probabilities)):
            probabilities = np.ones(len(probabilities)) / len(probabilities)

        return probabilities

    def sample_words(self, n_samples: int = 5, temperature: float = 1.0) -> List[str]:
        """Sample words based on evolved quantum state"""
        final_state = self.evolve_state()
        probabilities = np.abs(final_state) ** 2
        probabilities = self._normalize_probabilities(probabilities)

        if temperature != 1.0:
            probabilities = np.power(probabilities, 1.0 / temperature)
            probabilities = self._normalize_probabilities(probabilities)

        words = list(self.vocabulary.keys())
        selected_indices = np.random.choice(len(words), size=n_samples, p=probabilities)
        return [words[i] for i in selected_indices]

    def blend_concepts(
        self, concept_words: List[str], weights: List[float]
    ) -> List[str]:
        """Blend multiple concepts through quantum interference"""
        weights_array = np.array(weights, dtype=float)
        weights_array = self._normalize_probabilities(weights_array)

        state = np.zeros(len(self.vocabulary), dtype=complex)
        for word, weight in zip(concept_words, weights_array):
            if word in self.vocabulary and word in self.patterns:
                idx = list(self.vocabulary.keys()).index(word)
                # Combine embedding and pattern influence
                embedding_norm = np.linalg.norm(self.vocabulary[word])
                pattern_phase = np.angle(self.patterns[word].mean())
                state[idx] = weight * embedding_norm * np.exp(1j * pattern_phase)

        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        else:
            state = np.ones(len(state), dtype=complex) / np.sqrt(len(state))

        probabilities = np.abs(state) ** 2
        probabilities = self._normalize_probabilities(probabilities)

        words = list(self.vocabulary.keys())
        selected_indices = np.random.choice(len(words), size=5, p=probabilities)
        return [words[i] for i in selected_indices]

    def extract_code_patterns(self, content: str) -> CodePattern:
        """Extract hierarchical patterns from code"""
        lines = content.split("\n")
        root = CodePattern(pattern_type="root", content="", children=[])
        current_pattern = root
        indent_stack = [(0, root)]

        for line in lines:
            if not line.strip():
                continue

            # Calculate indentation level
            indent = len(line) - len(line.lstrip())

            # Handle indentation changes
            while indent_stack and indent < indent_stack[-1][0]:
                indent_stack.pop()

            if not indent_stack:
                indent_stack = [(0, root)]

            current_pattern = indent_stack[-1][1]

            # Detect pattern type
            stripped = line.strip()
            if stripped.startswith("def "):
                pattern_type = "function"
            elif stripped.startswith("class "):
                pattern_type = "class"
            elif stripped.endswith(":"):
                pattern_type = "block"
            else:
                pattern_type = "statement"

            # Create new pattern
            new_pattern = CodePattern(
                pattern_type=pattern_type,
                content=stripped,
                children=[],
                parent=current_pattern,
            )

            # Add to hierarchy
            current_pattern.children.append(new_pattern)

            # Update stack for blocks
            if pattern_type in ("function", "class", "block"):
                indent_stack.append((indent, new_pattern))

        return root

    def update_file_state(self, file_path: str, content: str) -> None:
        """Update quantum state of a file based on its content"""
        print(f"\nProcessing file: {os.path.basename(file_path)}")

        # Extract hierarchical patterns
        code_pattern = self.extract_code_patterns(content)
        self.code_patterns[file_path] = code_pattern
        print("  - Extracted hierarchical code patterns")

        # Process patterns recursively
        def process_pattern(pattern: CodePattern) -> np.ndarray:
            # Create quantum state for this pattern
            if pattern.pattern_type == "root":
                state = np.zeros(self.dims, dtype=complex)
            else:
                # Create pattern-specific state
                state = text_to_quantum_pattern(pattern.content, self.dims)

            # Combine with children states
            if pattern.children:
                child_states = [process_pattern(child) for child in pattern.children]
                if child_states:
                    # Quantum interference between parent and children
                    combined = np.mean([state] + child_states, axis=0)
                    state = combined / np.linalg.norm(combined)

            pattern.quantum_state = state
            return state

        # Process entire hierarchy
        process_pattern(code_pattern)
        print("  - Created quantum states for code patterns")

        # Continue with normal file processing
        mod_time = os.path.getmtime(file_path)

        if file_path in self.file_states:
            if mod_time <= self.file_states[file_path].last_modified:
                print(f"  - No changes detected (last modified: {mod_time})")
                return
            print(
                f"  - Changes detected (last: {self.file_states[file_path].last_modified}, new: {mod_time})"
            )

        # Extract words
        words = content.split()
        print(f"  - Found {len(words)} words")
        word_states = {}

        # Process words in context
        n_processed = 0
        for i, word in enumerate(words):
            if word not in self.vocabulary:
                # Create random embedding for new word
                embedding = np.random.randn(self.embedding_dim)
                self.add_word(word, embedding)
                n_processed += 1

            # Get context window
            start = max(0, i - 5)
            end = min(len(words), i + 6)
            context = words[start:i] + words[i + 1 : end]

            # Create quantum state for word in context
            if word in self.patterns:
                # Base state from word's pattern
                state = self.patterns[word].copy()

                # Influence from context words
                n_context = 0
                for ctx_word in context:
                    if ctx_word in self.patterns:
                        n_context += 1
                        # Quantum interference between patterns
                        interaction = np.vdot(
                            self.patterns[word], self.patterns[ctx_word]
                        )
                        state += 0.1 * interaction * self.patterns[ctx_word]

                # Normalize
                norm = np.linalg.norm(state)
                if norm > 0:
                    state = state / norm
                word_states[word] = state

            if i % 100 == 0:  # Progress for large files
                print(
                    f"  - Processed {i}/{len(words)} words ({n_context} context words)"
                )

        print(f"  - Added {n_processed} new words to vocabulary")
        print(f"  - Created quantum states for {len(word_states)} words")

        # Create overall file state from word states
        if word_states:
            file_state = np.mean([state for state in word_states.values()], axis=0)
            file_state = file_state / np.linalg.norm(file_state)
            print(f"  - Created file quantum state from {len(word_states)} word states")
        else:
            file_state = np.zeros(self.dims, dtype=complex)
            print("  - Created empty file quantum state (no known words)")

        # Update file quantum state
        self.file_states[file_path] = QuantumFile(
            path=file_path,
            state=file_state,
            last_modified=mod_time,
            word_states=word_states,
        )

        # Update patterns based on file context
        for word, state in word_states.items():
            self.patterns[word] = 0.9 * self.patterns[word] + 0.1 * state
            self.patterns[word] = self.patterns[word] / np.linalg.norm(
                self.patterns[word]
            )

        print(f"  - Updated quantum patterns for {len(word_states)} words")

        # Mark Hamiltonian as needing rebuild
        self.hamiltonian = None

    def get_file_similarity(self, file1: str, file2: str) -> float:
        """Calculate quantum similarity between two files"""
        if file1 in self.file_states and file2 in self.file_states:
            # Get quantum states
            state1 = self.file_states[file1].state
            state2 = self.file_states[file2].state

            # Calculate quantum fidelity
            fidelity = np.abs(np.vdot(state1, state2)) ** 2

            # Get common words
            words1 = set(self.file_states[file1].word_states.keys())
            words2 = set(self.file_states[file2].word_states.keys())
            common_words = words1.intersection(words2)

            # Add contribution from word-level similarity
            word_sim = 0.0
            if common_words:
                for word in common_words:
                    state1 = self.file_states[file1].word_states[word]
                    state2 = self.file_states[file2].word_states[word]
                    word_sim += np.abs(np.vdot(state1, state2)) ** 2
                word_sim /= len(common_words)

                # Combine file and word similarities
                return float(0.6 * fidelity + 0.4 * word_sim)

            return float(fidelity)
        return 0.0

    def suggest_related_files(self, file_path: str, n_samples: int = 3) -> List[str]:
        """Suggest related files based on quantum similarity"""
        if file_path not in self.file_states:
            return []

        # Calculate similarities with all other files
        similarities = []
        for other_path in self.file_states:
            if other_path != file_path:
                similarity = self.get_file_similarity(file_path, other_path)
                similarities.append((other_path, similarity))

        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in similarities[:n_samples]]

    def inspect_function_patterns(self, function_name: Optional[str] = None) -> None:
        """Print insights about learned function patterns"""
        if function_name:
            if function_name not in self.code_patterns:
                print(f"No pattern found for function: {function_name}")
                return

            pattern = self.code_patterns[function_name]
            print(f"\nFunction: {function_name}")
            print(f"Type: {pattern.pattern_type}")
            print(f"Children: {len(pattern.children)}")

            # Show quantum properties
            if pattern.quantum_state is not None:
                energy = np.abs(pattern.quantum_state.mean())
                phase = np.angle(pattern.quantum_state.mean())
                print(f"Quantum Energy: {energy:.3f}")
                print(f"Quantum Phase: {phase:.3f}")

            # Show related functions
            related = self.suggest_related_files(function_name)
            if related:
                print("\nRelated functions:")
                for rel in related:
                    print(f"- {os.path.basename(rel)}")
        else:
            # Show summary of all functions
            print("\nLearned Function Patterns:")
            for path, pattern in self.code_patterns.items():
                name = os.path.basename(path)
                n_children = len(pattern.children)
                energy = (
                    np.abs(pattern.quantum_state.mean())
                    if pattern.quantum_state is not None
                    else 0
                )
                print(f"- {name}: {n_children} blocks, energy={energy:.3f}")

    def get_state_from_words(self, words: List[str]) -> Optional[np.ndarray]:
        """Generate a quantum state representation for a list of words.

        Args:
            words: List of words to generate state for

        Returns:
            Quantum state vector or None if no valid words
        """
        if not words:
            return None

        # Initialize state vector
        state = np.zeros(self.dims, dtype=np.complex128)
        n_valid = 0

        # Add contribution from each word
        for word in words:
            if word in self.patterns:
                state += self.patterns[word]
                n_valid += 1
            elif word in self.vocabulary:
                # Create pattern from vocabulary embedding
                pattern = text_to_quantum_pattern(word, self.dims)
                state += pattern
                n_valid += 1

        # Return normalized state if we found any valid words
        if n_valid > 0:
            return quantum_normalize(state)
        return None
