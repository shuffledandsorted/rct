"""Interactive dialog system for code analysis and interaction.

This module provides a dialog system that:
1. Watches repository files for changes
2. Captures and processes screen state
3. Maintains an interactive conversation with the user
4. Uses quantum-inspired algorithms for code understanding
"""

import os
import time
import asyncio
import numpy as np
from typing import Tuple, Dict
from watchdog.observers import Observer

from src.agents.dialog import (
    command,
    default,
    SelfDiscoveringDialog,
    async_dialog,
)
from ..quantum.utils import (
    calculate_coherence,
    text_to_quantum_pattern,
    quantum_normalize,
    calculate_geodesic_collapse,
)
from ..quantum.game_theory import cooperative_collapse
from ..quantum.contract_crystallizer import ContractCrystallizer
from ..quantum.word_learner import QuantumWordLearner
from ..contracts.temporal import TemporalContract
from .self_aware import SelfAwareAgent
from .screen_capture import ScreenCaptureAgent
from .file_watcher import FileChangeHandler
from .repository_learner import RepositoryLearner
from ..contracts.query import QueryContract
from ..quantum.contract_formation import QuantumContractFormation


class CodeDialog(SelfDiscoveringDialog):
    """Interactive dialog system for code analysis and interaction.

    This module provides a dialog system that:
    1. Watches repository files for changes
    2. Captures and processes screen state
    3. Maintains an interactive conversation with the user
    4. Uses quantum-inspired algorithms for code understanding

    System Architecture Flow:

                   Cycle Begins
                        ↓
    ┌─────────────────────────────────────────────┐
    │                                            │
    │  ┌──────────────────┐  ┌──────────────────┐│
    │  │   CodeDialog     │  │RepositoryLearner ││
    │  │                  │  │                  ││
    │  │   Environment    │◄►│    Knowledge     ││
    │  │   Interaction    │  │     Growth       ││
    │  └──────────────────┘  └──────────────────┘│
    │                                            │
    │             Negotiation                    │
    │             of Contracts                   │
    │                   │                        │
    │                   ▼                        │
    │            Crystallization                 │
    │                                            │
    └────────────────────┬───────────────────────┘
                         │
                         ▼
                   Cycle Ends

    The system operates through continuous cycles where temporal and function
    contracts negotiate understanding. These negotiations crystallize into
    stable patterns, which seed the next cycle of learning and growth.
    """

    def __init__(self, dims: Tuple[int, int]):
        super().__init__()
        self.dims = dims
        self.learner = RepositoryLearner(dims=dims)
        self.screen_agent = ScreenCaptureAgent(dims=dims)
        self.user_agent = SelfAwareAgent(dims=dims)
        self.is_capturing = False
        self.file_observer = None
        self.file_handler = None

        # Create contract formation tool
        self.contract_formation = QuantumContractFormation(dims)

        # Register state providers
        self.contract_formation.register_state_provider(
            "word_learner",
            lambda ctx: (
                self.word_learner.get_state_from_words(ctx["input"].split())
                if hasattr(self.word_learner, "get_state_from_words")
                else None
            ),
            base_weight=0.4,
        )

        self.contract_formation.register_state_provider(
            "screen",
            lambda ctx: (
                self.screen_agent.wave_fn.amplitude
                if self.is_capturing and self.screen_agent.wave_fn is not None
                else None
            ),
            base_weight=0.2,
        )

        self.contract_formation.register_state_provider(
            "functions",
            lambda ctx: [
                state
                for name, state in self.function_contracts.items()
                if isinstance(state, np.ndarray)
                and bool(
                    set(ctx["input"].lower().split()) & set(name.lower().split("_"))
                )
            ],
            base_weight=0.2,
        )

    def initialize_repository(self, repository_path: str) -> None:
        """Initialize quantum states and contracts for existing repository files"""
        self.learner.initialize_repository(repository_path)

    def process_file_change(self, file_path: str) -> None:
        """Process changes in a file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            function_name = os.path.splitext(os.path.basename(file_path))[0]
            pattern = text_to_quantum_pattern(content, self.dims)
            self.learner.function_contracts[function_name] = pattern
            self.learner.learn_from_code(content)
            self.learner.learn_from_function(content, function_name)

        except Exception as e:
            print(f"Error processing file change {file_path}: {e}")

    @property
    def temporal_contracts(self) -> Dict[str, TemporalContract]:
        return self.learner.temporal_contracts

    @property
    def function_contracts(self) -> Dict[str, np.ndarray]:
        return self.learner.function_contracts

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

    async def run(self, repository_path: str) -> None:
        """Run the dialog system with background initialization."""
        try:
            # Validate repository path
            repository_path = os.path.abspath(repository_path)
            if not os.path.exists(repository_path):
                raise ValueError(f"Repository path does not exist: {repository_path}")

            # Start file watching first
            self.start_file_watching(repository_path)

            # Start initialization in background
            init_task = asyncio.create_task(
                asyncio.to_thread(self.initialize_repository, repository_path)
            )

            print(f"\nInitializing repository at {repository_path}")
            print("You can start interacting while I learn about your code.\n")

            # Start dialog loop
            try:
                while True:
                    # Update prompt based on initialization status
                    prompt = "(loading) " if not init_task.done() else ""
                    prompt += "You: "

                    # Get and process user input
                    try:
                        dialog_task = await async_dialog(self, prompt)
                        if dialog_task is None:  # Exit condition
                            break
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        print(f"Error in dialog: {e}")
                        continue

                    # Update agent counts if screen capture is active
                    self._update_agent_counts()

            except asyncio.CancelledError:
                pass

            # Wait for initialization to complete if still running
            if not init_task.done():
                try:
                    await init_task
                except Exception as e:
                    print(f"Error during initialization: {e}")

        finally:
            # Clean up watchers and resources
            self.stop_file_watching()
            if self.is_capturing:
                self.stop_screen_capture()

            print("\nDialog session ended.")

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

    def align_dimensions(self, state: np.ndarray, target_dims: tuple) -> np.ndarray:
        """Align state dimensions using geodesic collapse"""
        if state.size != np.prod(target_dims):
            # Reshape while preserving phase relationships
            # Calculate geodesic path
            path = calculate_geodesic_collapse(
                state.reshape(-1), np.zeros(np.prod(target_dims), dtype=np.complex128)
            )
            return path.reshape(target_dims)
        return state.reshape(target_dims)

    def calculate_weight(self, state: np.ndarray) -> float:
        """Calculate weight based on state coherence"""
        coherence = calculate_coherence(state)
        return np.exp(coherence) / (1 + np.exp(coherence))  # Sigmoid scaling

    @default
    def say(self, user_input: str) -> None:
        """Process user input and generate a response using quantum RCT."""
        # Form contract using current context
        context = {
            "input": user_input,
            "previous_state": self.user_agent.self_model.amplitude,
        }

        # Get combined state from valid states
        combined_state = self.contract_formation.form_contract(context)

        if combined_state is not None:
            # Let agent naturally converge to eigenvector
            self.user_agent.update_with_state(combined_state)

            # Create query contract using the converged self-model
            query_contract = QueryContract(
                user_agent=self.user_agent,
                knowledge_agent=self.learner,
            )

            # Process query through natural quantum evolution
            result = query_contract.process_query(user_input)

            if result:
                print("Response:", result["knowledge"])
                if "relevant_funcs" in result:
                    print("Related functions:", ", ".join(result["relevant_funcs"]))
                print(f"Coherence: {result['coherence']:.3f}")
                print(f"Energy: {result['energy']:.3f}")
            else:
                print("Could not reach sufficient coherence for response.")
        else:
            print("No valid agent states available.")

    @command
    def details(self) -> None:
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
        status = f"Status: {len(active_contracts)} contracts | {len(active_contracts) * 2} agents"
        status += f" | {avg_coherence:.3f} coherence | Most active: {most_active}"
        if self.is_capturing:
            status += " | Screen capture: Active"
        else:
            status += " | Screen capture: Inactive"

        # Move up a line, print status, restore cursor
        print(f"\033[F\r{status}\033[E", end="", flush=True)

    @command
    def crystallize(self, contract_name: str, temperature: float = 1.0) -> None:
        """Grow a computational crystal from a contract's patterns."""
        if contract_name not in self.temporal_contracts:
            print(f"Contract {contract_name} not found")
            return

        contract = self.temporal_contracts[contract_name]
        if not contract.is_valid():
            print("Contract must be valid to crystallize")
            return

        print(f"\nGrowing crystal from {contract_name}...")
        print(f"Temperature: {temperature:.1f}")

        # Create crystallizer and observe the contract
        crystallizer = ContractCrystallizer()
        crystallizer.observe_contract(contract)

        # Check initial state
        coherence = crystallizer.wave_fn.measure_coherence()
        print(f"Initial coherence: {coherence:.3f}")

        # Try to crystallize the pattern
        if coherence > 0.3:  # Has some structure worth growing
            # Generate a new function name
            func_name = f"crystal_{contract_name}"

            # Save the crystallized pattern
            filename = f"src/crystals/{func_name}.py"
            os.makedirs("src/crystals", exist_ok=True)

            if crystallizer.save_to_file(filename, func_name):
                print(f"\nSuccessfully crystallized pattern to {filename}")
                print("You can now import and use this function!")
            else:
                print("\nFailed to save crystallized pattern")
        else:
            print("\nPattern too chaotic to crystallize")

    @command
    def inspect_crystal(self, contract_name: str) -> None:
        """Analyze the quantum properties of a contract's patterns."""
        if contract_name not in self.temporal_contracts:
            print(f"Contract {contract_name} not found")
            return

        contract = self.temporal_contracts[contract_name]
        crystallizer = ContractCrystallizer()
        crystallizer.observe_contract(contract)

        print(f"\nQuantum Analysis of {contract_name}")
        print("================================")
        coherence = crystallizer.wave_fn.measure_coherence()
        print(f"Coherence:   {coherence:.3f}")

        # Get pattern frequencies
        amplitude = np.abs(crystallizer.wave_fn.amplitude)
        freqs = np.fft.fft2(amplitude)
        main_freqs = np.argsort(np.abs(freqs).flatten())[-3:]  # Top 3 frequencies

        # Analyze the patterns
        print("\nDominant Patterns:")
        max_strength = 0.0
        max_period = 0.0

        for i, freq_idx in enumerate(main_freqs, 1):
            freq = np.unravel_index(freq_idx, freqs.shape)
            strength = float(np.abs(freqs[freq]) / np.abs(freqs).max())
            # Add 1 to avoid division by zero
            period = float(
                amplitude.shape[0] / (freq[0] + 1)
            )  # Convert frequency to period
            print(f"{i}. Period {period:.1f} steps: {strength:.2%} strength")

            if strength > max_strength:
                max_strength = strength
                max_period = period

        # Try to interpret the patterns
        if max_strength > 0.8:
            print("\nPattern Interpretation:")
            if max_period > amplitude.shape[0] / 2:
                print("- Long-term transformation (global behavior)")
            elif max_period > 5:
                print("- Medium-term cycle (repeated behavior)")
            else:
                print("- Short-term oscillation (local behavior)")

    @property
    def word_learner(self) -> QuantumWordLearner:
        return self.learner.word_learner

    @property
    def embeddings(self) -> Dict[str, np.ndarray]:
        return self.learner.embeddings

    @property
    def agent_lifetime(self) -> float:
        return self.learner.agent_lifetime


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Code Dialog - Interactive code analysis system"
    )
    parser.add_argument(
        "repository_path",
        nargs="?",
        default=".",
        help="Path to the repository to analyze (default: current directory)",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Dimensions for quantum state space (default: 512 512)",
    )
    parser.add_argument(
        "--no-screen", action="store_true", help="Disable screen capture"
    )

    args = parser.parse_args()

    print("\nCode Dialog")
    print("===========")

    # Create dialog instance with specified dimensions
    dialog = CodeDialog(dims=(args.dims[0], args.dims[1]))

    # Start screen capture unless disabled
    if not args.no_screen:
        dialog.start_screen_capture()

    # Run the dialog with specified repository path
    try:
        asyncio.run(dialog.run(args.repository_path))
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nError: {e}")
        raise
