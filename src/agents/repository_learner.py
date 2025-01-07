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
from typing import Dict, Optional, Tuple, Set
from multiprocessing import Pool, cpu_count

from ..quantum.word_learner import QuantumWordLearner
from ..contracts.temporal import TemporalContract
from ..quantum.utils import (
    text_to_quantum_pattern,
    quantum_normalize,
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

        print(f"Found {len(python_files)} Python files")

        # Set up parallel processing
        n_cores = max(1, cpu_count() - 1)  # Leave one core free
        min_files_per_process = 5
        max_processes = min(n_cores, len(python_files) // min_files_per_process)

        if max_processes > 1:
            print(f"Processing files using {max_processes} cores...")
            with Pool(max_processes) as pool:
                # Process in batches to avoid memory issues
                batch_size = max(
                    min_files_per_process, len(python_files) // max_processes
                )
                for i in range(0, len(python_files), batch_size):
                    batch = python_files[i : i + batch_size]
                    args = [(file_path, self.dims) for file_path in batch]
                    results = pool.map(_process_file, args)

                    # Process results
                    for result in results:
                        if result is not None:
                            function_name, content, pattern = result
                            self.function_contracts[function_name] = pattern
                            self.learn_from_code(content)
                            self.learn_from_function(content, function_name)
                            self.watched_files.add(function_name)
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

        print(f"\nInitialized {len(self.function_contracts)} function contracts")
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
