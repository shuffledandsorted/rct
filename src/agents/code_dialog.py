"""Interactive dialog system for code analysis and interaction.

This module provides a dialog system that:
1. Watches repository files for changes
2. Captures and processes screen state
3. Maintains an interactive conversation with the user
4. Uses quantum-inspired algorithms for code understanding
"""

from typing import List, Dict, Optional, Union, Tuple, Set, Callable, Any
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import time
from watchdog.observers import Observer
import asyncio

from src.agents.dialog import command, default, SelfDiscoveringDialog, run_async_dialog
from ..quantum.word_learner import QuantumWordLearner
from ..contracts.temporal import TemporalContract
from ..quantum.utils import (
    quantum_normalize,
    calculate_coherence,
    calculate_cohesion,
    text_to_quantum_pattern,
)
from ..quantum.game_theory import cooperative_collapse
from .self_aware import SelfAwareAgent
from .screen_capture import ScreenCaptureAgent
from .file_watcher import FileChangeHandler, _process_file


class CodeDialog(SelfDiscoveringDialog):

    def __init__(self, dims: Tuple[int, int]):
        super().__init__()
        self.dims = dims  # Already a tuple for quantum word learner
        self.word_learner = QuantumWordLearner(dims=self.dims, embedding_dim=50)
        self.temporal_contracts: Dict[str, TemporalContract] = (
            {}
        )  # Keyed by function name
        self.function_contracts: Dict[str, np.ndarray] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.pattern = None
        self.file_observer = None
        self.file_handler = None
        self.watched_files: Set[str] = set()
        self.agent_lifetime = 3600  # Extend agent lifetime to 1 hour

        # Initialize screen capture agent
        self.screen_agent = ScreenCaptureAgent(dims=dims)
        self.is_capturing = False

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
        min_files_per_process = 5  # Minimum files to justify spawning a process
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
                    # Pass dims along with file path
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

    def start_file_watching(self, repository_path: str) -> None:
        """Start watching repository files for changes"""
        print("\n[WATCHER] Starting file watcher...")
        self.file_handler = FileChangeHandler(self)
        self.file_observer = Observer()
        self.file_observer.schedule(self.file_handler, repository_path, recursive=True)
        self.file_observer.start()
        print(f"[WATCHER] Now watching repository at {repository_path}")

    def stop_file_watching(self) -> None:
        """Stop watching repository files"""
        if self.file_observer:
            print("\n[WATCHER] Stopping file watcher...")
            self.file_observer.stop()
            self.file_observer.join()
            print("[WATCHER] File watcher stopped")

    def process_file_change(self, file_path: str) -> None:
        """Process changes in a file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            function_name = os.path.splitext(os.path.basename(file_path))[0]
            pattern = text_to_quantum_pattern(content, self.dims)
            self.function_contracts[function_name] = pattern
            self.learn_from_code(content)
            self.learn_from_function(content, function_name)

        except Exception as e:
            print(f"Error processing file change {file_path}: {e}")

    def run_with_file_watching(self, repository_path: str) -> None:
        """Run dialog with file watching"""
        try:
            # Initialize repository first
            self.initialize_repository(repository_path)

            # Start file watching
            self.start_file_watching(repository_path)

            # Run interactive session
            async def main():
                await run_async_dialog(self, "You: ")

            asyncio.run(main())

        finally:
            # Ensure we stop watching files even if an error occurs
            self.stop_file_watching()

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

    def negotiate_understanding(
        self, agent1: SelfAwareAgent, agent2: SelfAwareAgent, pattern: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Agents negotiate until they reach a stable understanding of the pattern"""
        # Start with their current states
        state1 = agent1.wave_fn.amplitude if agent1.wave_fn is not None else pattern
        state2 = (
            agent2.wave_fn.amplitude if agent2.wave_fn is not None else np.conj(pattern)
        )

        # Track iterations to prevent infinite loops
        max_iterations = 10
        iterations = 0

        # Negotiate until reaching fixed point or energy minimum
        while (
            not self._reached_stability(state1, state2) and iterations < max_iterations
        ):
            # Agent 1's perspective
            state1_new = quantum_normalize(state1 + 0.1 * pattern)

            # Agent 2's perspective (conjugate/symmetric)
            state2_new = cooperative_collapse(
                states=[state1_new, state2], weights=[0.3, 0.7]
            )
            state2_new = quantum_normalize(state2_new)

            # Check energy conservation
            if self._energy_conserved(state1, state2, state1_new, state2_new):
                state1 = state1_new
                state2 = state2_new
                iterations += 1
            else:
                break

        return state1, state2

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
        state1, state2 = self.negotiate_understanding(
            contract.agent1, contract.agent2, pattern
        )

        # Update contract with negotiated states
        contract.psi = lambda x: quantum_normalize(x + 0.1 * pattern)
        contract.agent1.wave_fn.amplitude = state1
        contract.agent2.wave_fn.amplitude = state2

        # Add to contracts
        self.temporal_contracts[function_name] = contract

    def _update_agent_counts(self):
        """Update screen agent with current agent counts"""
        if self.is_capturing and self.screen_agent:
            # Count total active contracts and their agents
            active_contracts = [
                contract
                for name, contract in self.temporal_contracts.items()
                if hasattr(contract, "creation_time")
                and time.time() - contract.creation_time < contract.lifetime
            ]
            total_agents = len(active_contracts) * 2  # Each contract has 2 agents

            # Count interested agents (those with high coherence with screen state)
            interested = 0
            if hasattr(self.screen_agent.wave_fn, "amplitude"):
                screen_state = self.screen_agent.wave_fn.amplitude
                for contract in active_contracts:
                    for agent in [contract.agent1, contract.agent2]:
                        if isinstance(agent.wave_fn, np.ndarray) and isinstance(
                            screen_state, np.ndarray
                        ):
                            # Calculate quantum similarity
                            similarity = (
                                np.abs(np.vdot(screen_state, agent.wave_fn)) ** 2
                            )
                            if similarity > 0.3:  # Threshold for interest
                                interested += 1

            self.screen_agent.update_agent_counts(total_agents, interested)

    @command
    def start_screen_capture(self) -> None:
        """Start screen capture thread"""
        if not self.is_capturing:
            print("\n[SCREEN] Starting screen capture...")
            self.screen_agent.start_capture()
            self.is_capturing = True
            print("[SCREEN] Screen capture active")

    @command
    def stop_screen_capture(self) -> None:
        """Stop screen capture thread"""
        if self.is_capturing:
            print("\n[SCREEN] Stopping screen capture...")
            self.screen_agent.stop_capture()
            self.is_capturing = False
            print("[SCREEN] Screen capture stopped")

    @default
    def say(self, user_input: str) -> "CodeDialog":
        """Process natural language input."""

        # Create quantum pattern from input
        input_pattern = text_to_quantum_pattern(user_input, self.dims)

        # Get active contracts and their agents
        active_contracts = [
            contract
            for name, contract in self.temporal_contracts.items()
            if hasattr(contract, "creation_time")
            and time.time() - contract.creation_time < contract.lifetime
        ]

        if not active_contracts:
            print("No active agents available. Refreshing context...")
            # Refresh context by re-initializing from repository files
            self.temporal_contracts.clear()  # Clear old contracts
            for name, pattern in self.function_contracts.items():
                contract = TemporalContract(
                    agent1=SelfAwareAgent(dims=self.dims),
                    agent2=SelfAwareAgent(dims=self.dims),
                    lifetime=self.agent_lifetime,
                )
                contract.creation_time = time.time()
                contract.psi = lambda x: quantum_normalize(x + 0.1 * pattern)
                contract.agent1.wave_fn.amplitude = pattern
                contract.agent2.wave_fn.amplitude = quantum_normalize(np.conj(pattern))
                self.temporal_contracts[name] = contract
            print(f"Refreshed {len(self.temporal_contracts)} contracts")
            return self

        # Update agent states based on input and screen state if available
        for contract in active_contracts:
            if (
                contract.psi is not None
                and contract.agent1.wave_fn is not None
                and contract.agent2.wave_fn is not None
            ):
                # Combine input with screen state if capturing
                if self.is_capturing and self.screen_agent.wave_fn is not None:
                    combined_input = quantum_normalize(
                        0.7 * input_pattern + 0.3 * self.screen_agent.wave_fn.amplitude
                    )
                else:
                    combined_input = input_pattern

                # Update first agent through contract
                state1 = contract.agent1.wave_fn.amplitude
                state1 = quantum_normalize(state1 + 0.1 * combined_input)
                contract.agent1.wave_fn.amplitude = state1

                # Ensure states are complex numpy arrays
                state1_complex = np.asarray(state1, dtype=np.complex128)
                state2_complex = np.asarray(
                    contract.agent2.wave_fn.amplitude, dtype=np.complex128
                )

                # Update second agent through entanglement
                if state1_complex.size > 0 and state2_complex.size > 0:
                    state2 = cooperative_collapse(
                        states=[state1_complex, state2_complex],
                        weights=[0.3, 0.7],
                    )
                    contract.agent2.wave_fn.amplitude = quantum_normalize(state2)

        # Generate response using valid agent states
        valid_states = []
        for contract in active_contracts:
            if contract.agent1.wave_fn is not None:
                state = np.asarray(
                    contract.agent1.wave_fn.amplitude, dtype=np.complex128
                )
                if state.size > 0:
                    valid_states.append(state)

        if valid_states:
            # Include screen state in response generation if capturing
            if self.is_capturing and self.screen_agent.wave_fn is not None:
                screen_state = np.asarray(
                    self.screen_agent.wave_fn.amplitude, dtype=np.complex128
                )
                valid_states.append(screen_state)

            weights = [1.0 / len(valid_states)] * len(valid_states)
            response_state = cooperative_collapse(states=valid_states, weights=weights)

            # Sample words based on evolved state
            response_words = self.word_learner.sample_words(
                n_samples=5, temperature=1.2
            )

            # Get relevant functions based on quantum similarity
            relevant_funcs = []
            for name, state in self.function_contracts.items():
                if isinstance(state, np.ndarray):
                    similarity = np.abs(np.vdot(response_state, state)) ** 2
                    if similarity > 0.3:  # Threshold for relevance
                        relevant_funcs.append(name)

            # Format response
            print("Response:", " ".join(response_words))
            if relevant_funcs:
                print("Related functions:", ", ".join(relevant_funcs))

            # Update word learner with interaction
            self.learn_from_code(user_input)
        else:
            print("No valid agent states available.")

        return self

    @command
    def details(self) -> "CodeDialog":
        """Print current status of agents and contracts"""
        current_time = time.time()
        active_contracts = [
            contract
            for name, contract in self.temporal_contracts.items()
            if hasattr(contract, "creation_time")
            and current_time - contract.creation_time < contract.lifetime
        ]

        # Get average coherence and most active function
        avg_coherence = 0.0
        n_coherent = 0
        for contract in active_contracts:
            if isinstance(contract.agent1.wave_fn, np.ndarray):
                avg_coherence += calculate_coherence(contract.agent1.wave_fn)
                n_coherent += 1
            if isinstance(contract.agent2.wave_fn, np.ndarray):
                avg_coherence += calculate_coherence(contract.agent2.wave_fn)
                n_coherent += 1

        # Include screen agent coherence if active
        if self.is_capturing and isinstance(
            self.screen_agent.wave_fn.amplitude, np.ndarray
        ):
            avg_coherence += calculate_coherence(self.screen_agent.wave_fn.amplitude)
            n_coherent += 1

        avg_coherence = avg_coherence / n_coherent if n_coherent > 0 else 0.0

        most_active = ""
        max_activity = 0.0
        for name, state in self.function_contracts.items():
            if isinstance(state, np.ndarray):
                activity = np.abs(state).mean()
                if activity > max_activity:
                    max_activity = activity
                    most_active = name

        # Print status including screen capture state
        status = f"\rStatus: {len(active_contracts)} contracts | {len(active_contracts) * 2} agents"
        status += f" | {avg_coherence:.3f} coherence | Most active: {most_active}"
        if self.is_capturing:
            status += " | Screen capture: Active"
        else:
            status += " | Screen capture: Inactive"

        print(status, end="", flush=True)
        return self


if __name__ == "__main__":
    print("Starting Code Dialog System")
    print("==========================")

    # Create dialog instance
    dialog = CodeDialog(dims=(512, 512))

    # Run with file watching
    dialog.run_with_file_watching(".")
