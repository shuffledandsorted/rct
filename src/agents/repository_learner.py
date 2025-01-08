"""Repository learning functionality.

This module handles learning from code files in a repository, including:
1. Processing Python files
2. Learning from code text
3. Creating and managing contracts
4. Maintaining embeddings and patterns
"""

import os
import time
import numpy as np
from typing import Dict, Optional, Tuple, Set, List
from multiprocessing import Pool, cpu_count

from ..quantum.word_learner import QuantumWordLearner
from ..contracts.temporal import TemporalContract
from ..quantum.utils import (
    text_to_quantum_pattern,
    quantum_normalize,
    calculate_coherence,
)
from ..quantum.game_theory import cooperative_collapse
from .self_aware import SelfAwareAgent
from .file_watcher import _process_file


class RepositoryLearner:
    def __init__(self, dims: Tuple[int, int]):
        self.dims = dims
        self.word_learner = QuantumWordLearner(dims=dims, embedding_dim=50)
        self.temporal_contracts: Dict[str, TemporalContract] = {}
        self.function_contracts: Dict[str, np.ndarray] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.watched_files: Set[str] = set()
        self.agent_lifetime = 3600  # 1 hour

    def initialize_repository(self, repository_path: str) -> None:
        """Initialize quantum states and contracts for existing repository files"""
        print("\nInitializing repository quantum states...")

        # Get all Python files in repository
        python_files = []
        for root, _, files in os.walk(repository_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    python_files.append(full_path)

        total_files = len(python_files)
        print(f"Found {total_files} Python files")
        processed_files = 0

        # Set up parallel processing
        n_cores = max(1, cpu_count() - 1)  # Leave one core free
        min_files_per_process = 5
        max_processes = min(n_cores, total_files // min_files_per_process)

        def update_progress():
            nonlocal processed_files
            processed_files += 1
            progress = (processed_files / total_files) * 100
            print(
                f"\rLoading: {progress:.1f}% ({processed_files}/{total_files} files)",
                end="",
                flush=True,
            )

        if max_processes > 1:
            print(f"Processing files using {max_processes} cores...")
            with Pool(max_processes) as pool:
                # Process in batches to avoid memory issues
                batch_size = max(min_files_per_process, total_files // max_processes)
                for i in range(0, total_files, batch_size):
                    batch = python_files[i : i + batch_size]
                    args = [(file_path, self.dims) for file_path in batch]
                    results = pool.map(_process_file, args)

                    # Process results
                    for file_path, result in zip(batch, results):
                        if result is not None:
                            function_name, content, pattern = result
                            self.function_contracts[function_name] = pattern
                            self.learn_from_code(content)
                            self.learn_from_function(content, function_name)
                            self.watched_files.add(file_path)
                        update_progress()
        else:
            # Process sequentially for small number of files
            for file_path in python_files:
                result = _process_file((file_path, self.dims))
                if result is not None:
                    function_name, content, pattern = result
                    self.function_contracts[function_name] = pattern
                    self.learn_from_code(content)
                    self.learn_from_function(content, function_name)
                    self.watched_files.add(file_path)
                update_progress()

        print(f"\n\nInitialized {len(self.function_contracts)} function contracts")
        print(f"Created {len(self.temporal_contracts)} temporal contracts")
        print(f"Learned {len(self.word_learner.vocabulary)} words")

    def learn_from_code(self, code_text: str) -> None:
        """Learn from code text using quantum word learning"""
        # Extract words and create simple embeddings
        words = code_text.split()
        for i, word in enumerate(words):
            if word not in self.embeddings:
                # Create embedding based on word's context
                context_size = 5
                start = max(0, i - context_size)
                end = min(len(words), i + context_size + 1)
                context = words[start:i] + words[i + 1 : end]

                # Simple embedding: average of random vectors for context words
                if context:
                    embedding = np.mean(
                        [self.embeddings.get(w, np.random.randn(50)) for w in context],
                        axis=0,
                    )
                else:
                    embedding = np.random.randn(50)

                self.embeddings[word] = embedding / np.linalg.norm(embedding)
                self.word_learner.add_word(word, self.embeddings[word])

    def learn_from_function(self, code_text: str, function_name: str) -> None:
        """Create or update temporal contract for a function"""
        # Create new temporal contract with extended lifetime
        contract = TemporalContract(
            agent1=SelfAwareAgent(dims=self.dims),
            agent2=SelfAwareAgent(dims=self.dims),
            lifetime=self.agent_lifetime,
        )
        contract.creation_time = time.time()

        # Initialize quantum states
        pattern = text_to_quantum_pattern(code_text, self.dims)

        # Have agents negotiate understanding of the pattern
        state1, state2 = self.negotiate_understanding(pattern)

        # Update contract with negotiated states
        contract.psi = lambda x: quantum_normalize(x + 0.1 * pattern)
        contract.agent1.wave_fn.amplitude = state1
        contract.agent2.wave_fn.amplitude = state2

        # Add to contracts
        self.temporal_contracts[function_name] = contract

    def negotiate_understanding(
        self, pattern: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Have agents negotiate understanding of a pattern."""
        # Initialize random states for both agents
        state1 = np.random.randn(*pattern.shape) + 1j * np.random.randn(*pattern.shape)
        state2 = np.random.randn(*pattern.shape) + 1j * np.random.randn(*pattern.shape)

        # Ensure we have numpy arrays
        state1 = np.asarray(state1, dtype=np.complex128)
        state2 = np.asarray(state2, dtype=np.complex128)

        # Normalize states
        state1 = quantum_normalize(state1)
        state2 = quantum_normalize(state2)

        # Iteratively refine understanding
        for _ in range(5):
            # Update first agent's state based on pattern
            state1 = quantum_normalize(state1 + 0.2 * pattern)

            # Update second agent's state through entanglement
            state2 = cooperative_collapse(
                states=[state1, state2],
                weights=[0.3, 0.7],
            )
            state2 = quantum_normalize(state2)

        return state1, state2

    def _reached_stability(self, state1: np.ndarray, state2: np.ndarray) -> bool:
        """Check if agents have reached a stable agreement (fixed point)"""
        # Y(C₁(C₂)) = Y(C₂(C₁)) from the paper
        similarity = np.abs(np.vdot(state1, state2)) ** 2
        return similarity > 0.9  # High agreement threshold

    def _energy_conserved(
        self,
        state1_old: np.ndarray,
        state2_old: np.ndarray,
        state1_new: np.ndarray,
        state2_new: np.ndarray,
    ) -> bool:
        """Check if energy is conserved in the negotiation"""
        energy_old = np.sum(np.abs(state1_old) ** 2) + np.sum(np.abs(state2_old) ** 2)
        energy_new = np.sum(np.abs(state1_new) ** 2) + np.sum(np.abs(state2_new) ** 2)
        return np.abs(energy_new - energy_old) < 0.1  # Small energy change threshold

    def process_response(
        self, valid_states: List[np.ndarray], temperature: float = 1.2
    ) -> Tuple[List[str], List[str], float]:
        """Process quantum states to generate a response.

        Args:
            valid_states: List of quantum states to process
            temperature: Temperature for word sampling (higher = more random)

        Returns:
            Tuple of (response words, relevant function names, response coherence)
        """
        # Combine states using cooperative collapse
        weights = [1.0 / len(valid_states)] * len(valid_states)
        response_state = cooperative_collapse(states=valid_states, weights=weights)

        # Sample words based on evolved state
        response_words = self.word_learner.sample_words(
            n_samples=5, temperature=temperature
        )

        # Get relevant functions based on quantum similarity
        relevant_funcs = []
        for name, state in self.function_contracts.items():
            if isinstance(state, np.ndarray):
                # Normalize states before computing similarity
                norm_response = response_state / np.sqrt(
                    np.vdot(response_state, response_state)
                )
                norm_state = state / np.sqrt(np.vdot(state, state))
                similarity = np.abs(np.vdot(norm_response, norm_state)) ** 2
                if similarity > 0.3:  # Threshold for relevance
                    relevant_funcs.append(name)

        # Calculate response coherence
        response_state = self.word_learner.get_state_from_words(response_words)
        coherence = 0.0
        if isinstance(response_state, np.ndarray):
            coherence = calculate_coherence(response_state)

        return response_words, relevant_funcs, coherence
