from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import random
import os


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


class QuantumWordLearner:
    def __init__(self, embedding_dim: int = 300, n_eigenstates: int = 50):
        self.embedding_dim = embedding_dim
        self.n_eigenstates = n_eigenstates
        self.vocabulary: Dict[str, QuantumWord] = {}
        self.patterns: Dict[str, np.ndarray] = {}  # Quantum patterns for each word
        self.file_states: Dict[str, QuantumFile] = {}  # Quantum states for files
        self.hamiltonian: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.eigenstates: Optional[np.ndarray] = None
        self.recent_words: List[str] = []  # Track recent words for context
        self.context_window = 50  # How many recent words to remember

    def update_quantum_state(self, word: str, new_state: np.ndarray) -> None:
        """Update the quantum state of a word based on new context"""
        if word in self.vocabulary:
            # Extract amplitude and phase from new state
            amplitude = np.linalg.norm(new_state)
            phase = np.angle(new_state.mean())

            # Keep existing semantic vector
            old_word = self.vocabulary[word]

            # Blend old and new quantum properties for smooth evolution
            old_amplitude = np.abs(old_word.amplitude)
            old_phase = old_word.phase

            # Exponential moving average for smooth updates
            alpha = 0.3  # Learning rate
            new_amplitude = (1 - alpha) * old_amplitude + alpha * amplitude
            new_phase = (1 - alpha) * old_phase + alpha * phase

            # Update the word's quantum properties
            self.vocabulary[word] = QuantumWord(
                word=word,
                amplitude=complex(new_amplitude),
                phase=new_phase,
                semantic_vector=old_word.semantic_vector,
            )

            # Add to recent words for context
            self.recent_words.append(word)
            if len(self.recent_words) > self.context_window:
                self.recent_words = self.recent_words[-self.context_window :]

            # Mark Hamiltonian as needing rebuild
            self.hamiltonian = None
            self.eigenvalues = None
            self.eigenstates = None

    def add_word(self, word: str, embedding: np.ndarray) -> None:
        """Add a word to the quantum system with its embedding"""
        # Generate random phase for quantum interference
        phase = random.uniform(0, 2 * np.pi)

        # Create amplitude based on embedding norm and recent context
        base_amplitude = np.linalg.norm(embedding)

        # Adjust amplitude based on recent word context if available
        if self.recent_words:
            context_boost = 0.0
            for recent in self.recent_words[-5:]:  # Look at last 5 words
                if recent in self.vocabulary:
                    context_sim = np.dot(
                        embedding / base_amplitude,
                        self.vocabulary[recent].semantic_vector,
                    )
                    context_boost += max(
                        0, context_sim
                    )  # Only boost for positive similarity
            base_amplitude *= 1.0 + 0.2 * context_boost  # Up to 20% boost from context

        # Create amplitude with context influence
        amplitude = complex(base_amplitude)

        # Normalize embedding to unit vector
        normalized_embedding = embedding / np.linalg.norm(embedding)

        # Create quantum word
        self.vocabulary[word] = QuantumWord(
            word=word,
            amplitude=amplitude,
            phase=phase,
            semantic_vector=normalized_embedding,
        )

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
            qword1 = self.vocabulary[word1]
            for j, word2 in enumerate(words):
                qword2 = self.vocabulary[word2]
                if i == j:
                    # Diagonal elements represent word "energy"
                    H[i, i] = np.abs(qword1.amplitude)
                else:
                    # Off-diagonal elements represent word interactions
                    similarity = np.dot(qword1.semantic_vector, qword2.semantic_vector)
                    phase_diff = qword1.phase - qword2.phase
                    H[i, j] = similarity * np.exp(1j * phase_diff)

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
            if word in self.vocabulary:
                idx = list(self.vocabulary.keys()).index(word)
                qword = self.vocabulary[word]
                state[idx] = weight * qword.amplitude * np.exp(1j * qword.phase)

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

    def update_file_state(self, file_path: str, content: str) -> None:
        """Update quantum state of a file based on its content"""
        # Get file modification time
        mod_time = os.path.getmtime(file_path)

        # Check if file has actually changed
        if file_path in self.file_states:
            if mod_time <= self.file_states[file_path].last_modified:
                return  # File hasn't changed

        # Extract words and create word states
        words = content.split()
        word_states = {}

        # Create quantum states for each word in context
        for i, word in enumerate(words):
            # Get context window around word
            start = max(0, i - 5)
            end = min(len(words), i + 6)
            context = words[start:i] + words[i + 1 : end]

            # Create quantum state from context
            state = np.zeros(self.embedding_dim, dtype=complex)
            if word in self.vocabulary:
                qword = self.vocabulary[word]
                # Base state from word's quantum properties
                state = (
                    qword.amplitude * np.exp(1j * qword.phase) * qword.semantic_vector
                )

                # Influence from context words
                for ctx_word in context:
                    if ctx_word in self.vocabulary:
                        ctx_qword = self.vocabulary[ctx_word]
                        interaction = np.dot(
                            qword.semantic_vector, ctx_qword.semantic_vector
                        )
                        state += (
                            0.1
                            * interaction
                            * ctx_qword.amplitude
                            * np.exp(1j * ctx_qword.phase)
                            * ctx_qword.semantic_vector
                        )

                # Normalize
                norm = np.linalg.norm(state)
                if norm > 0:
                    state = state / norm
                word_states[word] = state

        # Create overall file state from word states
        if word_states:
            file_state = np.mean([state for state in word_states.values()], axis=0)
            file_state = file_state / np.linalg.norm(file_state)
        else:
            file_state = np.zeros(self.embedding_dim, dtype=complex)

        # Update file quantum state
        self.file_states[file_path] = QuantumFile(
            path=file_path,
            state=file_state,
            last_modified=mod_time,
            word_states=word_states,
        )

        # Update word states based on file context
        for word, state in word_states.items():
            self.update_quantum_state(word, state)

    def get_file_similarity(self, file1: str, file2: str) -> float:
        """Calculate quantum similarity between two files"""
        if file1 in self.file_states and file2 in self.file_states:
            # Use quantum fidelity as similarity measure
            state1 = self.file_states[file1].state
            state2 = self.file_states[file2].state
            return float(np.abs(np.vdot(state1, state2)) ** 2)
        return 0.0

    def suggest_related_files(self, file_path: str, n_samples: int = 3) -> List[str]:
        """Suggest related files based on quantum similarity"""
        if file_path not in self.file_states:
            return []

        # Calculate similarities with all other files
        similarities = []
        for other_path, other_file in self.file_states.items():
            if other_path != file_path:
                similarity = self.get_file_similarity(file_path, other_path)
                similarities.append((other_path, similarity))

        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in similarities[:n_samples]]
