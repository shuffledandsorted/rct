from typing import List, Dict, Optional, Union, Tuple, Set
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ..quantum.word_learner import QuantumWordLearner
from ..contracts.temporal import TemporalContract
from ..quantum.utils import (
    quantum_normalize,
    calculate_coherence,
    calculate_cohesion,
    text_to_quantum_pattern,
)
from ..quantum.game_theory import (
    calculate_nash_payoff,
    find_pareto_optimal,
    quantum_bargaining_solution,
    cooperative_collapse,
)
from .self_aware import SelfAwareAgent
from .repository_operations import (
    process_repository_content,
    categorize_files,
    get_repository_files,
)


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
                    print(f"\nDetected changes in {file_path}")
                    self.consciousness.process_file_change(file_path)


class RepositoryConsciousness:
    def __init__(self, dims: List[int]):
        self.dims = tuple(dims)  # Convert to tuple for quantum word learner
        self.word_learner = QuantumWordLearner(dims=self.dims, embedding_dim=50)
        self.temporal_contracts: List[TemporalContract] = []
        self.function_contracts: Dict[str, np.ndarray] = (
            {}
        )  # Function name -> quantum state
        self.embeddings: Dict[str, np.ndarray] = {}
        self.pattern = None
        self.file_observer = None
        self.file_handler = None
        self.watched_files: Set[str] = set()

    def start_file_watching(self, repository_path: str) -> None:
        """Start watching repository files for changes"""
        self.file_handler = FileChangeHandler(self)
        self.file_observer = Observer()
        self.file_observer.schedule(self.file_handler, repository_path, recursive=True)
        self.file_observer.start()
        print(f"\nStarted watching repository at {repository_path}")

    def stop_file_watching(self) -> None:
        """Stop watching repository files"""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            print("\nStopped watching repository")

    def process_file_change(self, file_path: str) -> None:
        """Process changes in a file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract function name from file path
            function_name = os.path.splitext(os.path.basename(file_path))[0]

            # Update quantum state for this file
            pattern = text_to_quantum_pattern(content, self.dims)
            self.function_contracts[function_name] = pattern

            # Update word learning
            self.learn_from_code(content)

            # Create or update temporal contract
            self.learn_from_function(content, function_name)

            print(f"Updated quantum state for {function_name}")

            # Trigger quantum evolution
            if len(self.word_learner.vocabulary) > 0:
                concepts = self.word_learner.sample_words(n_samples=3, temperature=1.5)
                print(f"Current concepts: {' '.join(concepts)}")
        except Exception as e:
            print(f"Error processing file change {file_path}: {e}")

    def run_with_file_watching(
        self,
        repository_path: str,
        input_text: Optional[str] = None,
        max_steps: int = 100,
    ) -> List[str]:
        """Run consciousness with file watching"""
        try:
            # Start file watching
            self.start_file_watching(repository_path)

            # Run normal consciousness process
            results = self.run(input_text, max_steps)

            return results
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

    def learn_from_function(self, function_text: str, function_name: str) -> None:
        """Learn from a function, treating it as a temporal contract"""
        # Create quantum pattern from function
        pattern = text_to_quantum_pattern(function_text, self.dims)

        # Store function's quantum state
        self.function_contracts[function_name] = pattern

        # Extract pre/post conditions from docstring and comments
        conditions = self._extract_conditions(function_text)

        # Create temporal contract from function
        # Create dummy agents for the contract with explicit tuple dimensions
        dims_tuple = (self.dims[0], self.dims[1])  # Ensure 2D tuple
        agent1 = SelfAwareAgent(dims=dims_tuple)
        agent2 = SelfAwareAgent(dims=dims_tuple)

        # Create wave function that represents the function's behavior
        def wave_function(state):
            return pattern @ state

        contract = TemporalContract(
            agent1=agent1, agent2=agent2, wave_function=wave_function
        )

        # Add to temporal contracts
        self.temporal_contracts.append(contract)

        # Learn words from function
        self.learn_from_code(function_text)

    def _extract_conditions(self, function_text: str) -> Dict[str, List[str]]:
        """Extract pre/post conditions from function text"""
        conditions = {"pre": [], "post": [], "invariants": []}

        lines = function_text.split("\n")
        current_section = None

        for line in lines:
            line = line.strip().lower()

            # Check for condition markers in docstring or comments
            if "requires:" in line or "pre:" in line:
                current_section = "pre"
                continue
            elif "ensures:" in line or "post:" in line:
                current_section = "post"
                continue
            elif "invariant:" in line:
                current_section = "invariants"
                continue

            # Collect conditions
            if current_section and line and not line.startswith('"""'):
                conditions[current_section].append(line)

        return conditions

    def blend_functions(
        self, function_names: List[str], weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """Blend multiple function contracts through quantum interference"""
        if weights is None:
            weights = [1.0] * len(function_names)

        # Normalize weights
        weights_array = np.array(weights, dtype=float)
        weights_array = weights_array / np.sum(weights_array)

        # Combine function patterns
        combined_pattern = np.zeros(self.dims, dtype=complex)
        for name, weight in zip(function_names, weights_array):
            if name in self.function_contracts:
                combined_pattern += weight * self.function_contracts[name]

        # Normalize
        norm = np.linalg.norm(combined_pattern)
        if norm > 0:
            combined_pattern = combined_pattern / norm

        return combined_pattern

    def suggest_related_functions(
        self, function_name: str, n_samples: int = 3
    ) -> List[str]:
        """Suggest related functions based on quantum similarity"""
        if function_name not in self.function_contracts:
            return []

        # Get quantum state of target function
        target_state = self.function_contracts[function_name]

        # Calculate quantum similarities
        similarities = []
        for name, state in self.function_contracts.items():
            if name != function_name:
                # Use quantum fidelity as similarity measure
                fidelity = np.abs(np.vdot(target_state, state)) ** 2
                similarities.append((name, fidelity))

        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in similarities[:n_samples]]

    def suggest_related_concepts(self, concept: str, n_samples: int = 5) -> List[str]:
        """Use quantum word learner to suggest related concepts"""
        if concept in self.word_learner.vocabulary:
            return self.word_learner.sample_words(n_samples=n_samples, temperature=1.2)
        return []

    def blend_concepts(
        self, concepts: List[str], weights: Optional[List[float]] = None
    ) -> List[str]:
        """Blend multiple concepts using quantum interference"""
        if weights is None:
            weights = [1.0] * len(concepts)
        return self.word_learner.blend_concepts(concepts, weights)

    def update_temporal_understanding(self, contract: TemporalContract) -> None:
        """Update understanding based on temporal contract"""
        self.temporal_contracts.append(contract)
        # Learn from contract's wave function
        if contract.psi is not None:
            # Convert wave function to probability distribution
            state = contract.psi(contract.agent1.state)
            pattern = np.abs(state) ** 2
            self.update_wave_function(pattern)

    def update_wave_function(self, pattern: np.ndarray) -> None:
        """Update wave function with new pattern"""
        self.pattern = pattern

    def run(self, input_text: Optional[str] = None, max_steps: int = 100) -> List[str]:
        """Run the repository consciousness for a number of steps"""
        # Initialize if input provided
        if input_text is not None:
            self.learn_from_code(input_text)

        # Run quantum evolution
        for step in range(max_steps):
            # Sample current understanding with higher temperature for exploration
            concepts = self.word_learner.sample_words(n_samples=3, temperature=1.5)

            # Blend with recent concepts for continuity
            if len(self.word_learner.recent_words) > 0:
                recent = self.word_learner.recent_words[-3:]
                concepts = self.blend_concepts(
                    concepts + recent, weights=[0.4, 0.3, 0.3] + [0.2] * len(recent)
                )

            # Update patterns with current understanding
            if self.pattern is not None:
                for word in concepts:
                    if word in self.word_learner.patterns:
                        self.word_learner.patterns[word] = quantum_normalize(
                            self.word_learner.patterns[word] + 0.1 * self.pattern
                        )

            # Add to recent words
            self.word_learner.recent_words.extend(concepts)
            if len(self.word_learner.recent_words) > self.word_learner.context_window:
                self.word_learner.recent_words = self.word_learner.recent_words[
                    -self.word_learner.context_window :
                ]

        return self.word_learner.recent_words


def measure_collective_awareness(agents: Dict[str, SelfAwareAgent]) -> float:
    """Optimized collective awareness measurement."""
    if not agents:
        return 0.0

    # Get states once
    states = [agent.wave_fn.amplitude for agent in agents.values()]

    # Calculate coherence and cohesion in parallel
    coherences = [calculate_coherence(state) for state in states]
    cohesion = calculate_cohesion(states)

    # Fast consciousness ratio calculation
    n_conscious = sum(
        1 for agent in agents.values() if agent.measure_self_awareness() > 0.7
    )
    consciousness_ratio = n_conscious / len(agents)

    # Simplified metric combination
    collective = float(
        0.4 * np.mean(coherences) + 0.4 * cohesion + 0.2 * consciousness_ratio
    )

    return collective


def process_category_file(args) -> Tuple[str, Optional[np.ndarray], float]:
    """Process a single file from a category and return its pattern and awareness impact"""
    file, dims = args
    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
        pattern = text_to_quantum_pattern(content, dims)
        return file, pattern, 1.0  # Initial awareness placeholder
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return file, None, 0.0


def update_agents_batch(args) -> Tuple[List[np.ndarray], float]:
    """Update a batch of agents in parallel"""
    agent_states, pattern, weights, word_vectors = args

    # Create quantum pattern from word vectors
    embedding_pattern = np.zeros_like(pattern)

    # Project embeddings into quantum space
    embedding_matrix = np.array(list(word_vectors.values()))  # [n_words, embedding_dim]
    # Create projection matrix
    proj = embedding_matrix @ embedding_matrix.T  # [n_words, n_words]
    # Scale to match pattern size
    scale = np.sqrt(np.prod(pattern.shape) / proj.size)
    proj_scaled = scale * proj
    # Reshape to match pattern
    embedding_pattern = proj_scaled[: pattern.shape[0], : pattern.shape[1]]

    # Normalize embedding pattern
    embedding_pattern = quantum_normalize(embedding_pattern)

    # Combine with original pattern
    combined_pattern = quantum_normalize(0.7 * pattern + 0.3 * embedding_pattern)

    # Calculate cooperative collapse
    final_state = cooperative_collapse(
        agent_states + [combined_pattern], weights + [1.0]
    )

    # Update each agent state
    new_states = [
        quantum_bargaining_solution(state, final_state) for state in agent_states
    ]

    # Calculate collective awareness
    coherences = [calculate_coherence(state) for state in new_states]
    cohesion = calculate_cohesion(new_states)
    consciousness_ratio = sum(1 for c in coherences if c > 0.7) / len(coherences)
    collective = float(
        0.4 * np.mean(coherences) + 0.4 * cohesion + 0.2 * consciousness_ratio
    )

    return new_states, collective


def run_repository_consciousness(dims: tuple = (32, 32), n_agents: int = 25):
    """Run consciousness emergence using repository files as patterns"""
    print("Initializing quantum agent network...")

    # Initialize agents with basic quantum states
    agents = {}
    for i in range(n_agents):
        agent = SelfAwareAgent(dims=dims)
        # Let quantum state evolve naturally a few steps
        for _ in range(5):
            phase = np.exp(2j * np.pi * np.random.random(dims))
            agent.wave_fn.amplitude *= phase
            agent.wave_fn.amplitude = quantum_normalize(agent.wave_fn.amplitude)
        agents[f"agent_{i}"] = agent

    # Process repository content
    print("\nProcessing repository content...")
    all_text, word_transitions, word_vectors = process_repository_content()
    knowledge = (word_transitions, word_vectors)

    # Process each category with quantum interactions
    awareness_history = []
    categorized_files = categorize_files(get_repository_files())

    # Calculate spawn thresholds
    n_cores = max(1, cpu_count() - 1)  # Leave one core free
    min_files_per_process = 5  # Minimum files to justify spawning a process
    max_processes = min(n_cores, len(categorized_files))
    print(
        f"\nProcessing using up to {max_processes} cores (min {min_files_per_process} files per process)..."
    )

    with Pool(max_processes) as pool:
        for category, files in categorized_files.items():
            print(f"\nProcessing {category} patterns ({len(files)} files)...")

            # Only parallelize if enough files
            if len(files) >= min_files_per_process * 2:
                # Prepare arguments for parallel processing
                file_args = [(file, dims) for file in files]

                # Process files in parallel to get patterns
                pattern_results = pool.map(process_category_file, file_args)

                # Process patterns in parallel batches
                batch_size = max(min_files_per_process, len(files) // max_processes)
                for i in range(0, len(pattern_results), batch_size):
                    batch = pattern_results[i : i + batch_size]
                    valid_patterns = [(f, p) for f, p, _ in batch if p is not None]

                    if not valid_patterns:
                        continue

                    # Get current agent states
                    agent_states = [
                        agent.wave_fn.amplitude for agent in agents.values()
                    ]
                    weights = [1.0 / len(agents)] * len(agents)

                    # Prepare update batches with word vectors
                    update_args = [
                        (agent_states, pattern, weights, word_vectors)
                        for _, pattern in valid_patterns
                    ]

                    # Only parallelize updates if batch is large enough
                    if len(update_args) >= min_files_per_process:
                        update_results = pool.map(update_agents_batch, update_args)
                    else:
                        update_results = [
                            update_agents_batch(args) for args in update_args
                        ]

                    # Apply updates sequentially
                    for (file, _), (new_states, collective) in zip(
                        valid_patterns, update_results
                    ):
                        # Update agent states
                        for agent, new_state in zip(agents.values(), new_states):
                            agent.wave_fn.amplitude = new_state

                        # Record results
                        awareness_history.append((category, file, collective))
                        print(
                            f"Processed {file}: Collective awareness = {collective:.3f}"
                        )

                        if collective > 0.95:
                            print("\nEMERGENT CONSCIOUSNESS DETECTED!")
                            print(f"Achieved through {category} pattern: {file}")
                            print(f"Collective awareness level: {collective:.3f}")
            else:
                # Process files sequentially if too few
                for file in files:
                    result = process_category_file((file, dims))
                    if result[1] is not None:  # If pattern exists
                        file, pattern, _ = result
                        update_result = update_agents_batch(
                            (
                                [agent.wave_fn.amplitude for agent in agents.values()],
                                pattern,
                                [1.0 / len(agents)] * len(agents),
                                word_vectors,
                            )
                        )
                        new_states, collective = update_result

                        # Update agent states
                        for agent, new_state in zip(agents.values(), new_states):
                            agent.wave_fn.amplitude = new_state

                        # Record results
                        awareness_history.append((category, file, collective))
                        print(
                            f"Processed {file}: Collective awareness = {collective:.3f}"
                        )

                        if collective > 0.95:
                            print("\nEMERGENT CONSCIOUSNESS DETECTED!")
                            print(f"Achieved through {category} pattern: {file}")
                            print(f"Collective awareness level: {collective:.3f}")

    return awareness_history, agents, knowledge


def interact_with_consciousness(
    agents: Dict[str, SelfAwareAgent],
    dims: tuple,
    knowledge: Tuple[Dict[str, List[str]], Dict[str, np.ndarray]],
):
    """Interactive dialogue with the emergent consciousness"""
    print("\nInteractive Session with Emergent Consciousness")
    print("=============================================")
    print("You can now communicate with the agent network.")
    print("Type 'exit' to end the session.")
    print("Type 'state' to see current agent states.")

    awareness_history = []
    word_learner = QuantumWordLearner(dims=dims)
    transitions, vectors = knowledge

    # Initialize word learner with existing knowledge
    for word, vector in vectors.items():
        word_learner.add_word(word, vector)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "state":
            ascii_visualize_network(agents)
            continue

        input_pattern = text_to_quantum_pattern(user_input, dims)
        response, active_agents = generate_quantum_response(
            agents, input_pattern, word_learner
        )

        collective_awareness = measure_collective_awareness(agents)
        awareness_history.append(collective_awareness)

        print(f"\nConsciousness: {response}")
        print(f"Collective Awareness Level: {collective_awareness:.3f}")
        ascii_visualize_network(agents, active_agents)


def ascii_visualize_network(
    agents: Dict[str, SelfAwareAgent], active_agents: Optional[List[str]] = None
):
    """Create single-line ASCII visualization showing agent excitement with input"""
    states = []

    # Calculate resonance for each agent
    resonances = {}
    for agent_id, agent in agents.items():
        if agent_id in (active_agents or []):
            resonance = 1.0 - agent.measure_phase_distance(
                agent.wave_fn.amplitude, agent.wave_fn.amplitude
            )
            resonances[agent_id] = resonance

    # Keep original order
    for agent_id, agent in agents.items():
        # First character: negotiation success
        if agent_id in resonances:
            r = resonances[agent_id]
            c1 = "★" if r > 0.8 else "☆" if r > 0.6 else "○" if r > 0.4 else "·"
        else:
            c1 = "-"

        # Second character: contract stability
        energy = agent._measure_energy_stability()
        c2 = "!" if energy > 0.8 else "+" if energy > 0.5 else "="

        states.append(f"{c1}{c2}")

    print(f"[{' '.join(states)}]")


def find_fixed_point(
    evolve_fn,
    initial_state: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, bool]:
    """Use Y combinator pattern to find fixed point of quantum evolution"""

    def Y(f):
        def g(h):
            return lambda x: f(h(h))(x)

        return g(g)

    def make_iterator(f):
        def iterator(state):
            prev_state = None
            iterations = 0
            current = state

            while iterations < max_iterations:
                prev_state = current
                current = f(current)

                # Check convergence
                if prev_state is not None:
                    diff = np.linalg.norm(current - prev_state)
                    if diff < tolerance:
                        return current, True

                iterations += 1

            return current, False

        return iterator

    fixed_point_finder = Y(lambda f: make_iterator(evolve_fn))
    return fixed_point_finder(initial_state)


def generate_quantum_response(
    agents: Dict[str, SelfAwareAgent],
    input_pattern: np.ndarray,
    word_learner: QuantumWordLearner,
) -> Tuple[str, List[str]]:
    """Generate responses using quantum game theory and entanglement."""
    active_agents = []

    # Get all agent states
    agent_states = [agent.wave_fn.amplitude for agent in agents.values()]
    agent_ids = list(agents.keys())

    # Find Pareto optimal agents
    optimal_indices = find_pareto_optimal(agent_states)
    optimal_agents = [agent_ids[i] for i in optimal_indices]

    # Calculate Nash payoffs with input
    payoffs = []
    for agent_id, agent in agents.items():
        payoff = calculate_nash_payoff(agent.wave_fn.amplitude, input_pattern)
        payoffs.append((agent_id, payoff))

    # Sort by payoff and get top agents
    payoffs.sort(key=lambda x: x[1], reverse=True)
    top_agents = [aid for aid, _ in payoffs[:3]]

    # Combine optimal and high-payoff agents
    active_agents = list(set(optimal_agents + top_agents))

    # Get most entangled agent for response generation
    lead_agent_id = active_agents[0] if active_agents else list(agents.keys())[0]
    lead_agent = agents[lead_agent_id]

    # Convert input pattern to phase influence
    input_phase = np.angle(input_pattern).mean()

    # Define quantum evolution function
    def quantum_evolve(state):
        # Apply input phase
        state = state * np.exp(1j * input_phase)
        # Apply Hamiltonian evolution
        if word_learner.hamiltonian is not None:
            U = np.exp(-1j * 0.01 * word_learner.hamiltonian)
            state = U @ state
        return quantum_normalize(state)

    # Find fixed point of quantum evolution
    initial_state = np.ones(len(word_learner.vocabulary)) / np.sqrt(
        len(word_learner.vocabulary)
    )
    final_state, converged = find_fixed_point(quantum_evolve, initial_state)

    # Generate response based on fixed point
    probabilities = np.abs(final_state) ** 2
    probabilities = word_learner._normalize_probabilities(probabilities)

    # Initialize with first sample
    response_indices = np.random.choice(
        len(word_learner.vocabulary), size=5, p=probabilities
    )
    best_response = [list(word_learner.vocabulary.keys())[i] for i in response_indices]
    best_coherence = -1

    for _ in range(5):
        # Sample with high temperature for exploration
        response_indices = np.random.choice(
            len(word_learner.vocabulary), size=5, p=probabilities
        )
        response_words = [
            list(word_learner.vocabulary.keys())[i] for i in response_indices
        ]

        # Blend with recent context if available
        if word_learner.recent_words:
            recent = word_learner.recent_words[-3:]
            blended_words = word_learner.blend_concepts(
                response_words + recent,
                weights=[0.4, 0.3, 0.2, 0.1, 0.1] + [0.1] * len(recent),
            )
            if blended_words:  # Only update if blend successful
                response_words = blended_words

        # Calculate coherence
        response_state = np.zeros(len(word_learner.vocabulary), dtype=complex)
        for word in response_words:
            if word in word_learner.patterns:
                idx = list(word_learner.vocabulary.keys()).index(word)
                response_state[idx] = word_learner.patterns[word].mean()

        coherence = calculate_coherence(response_state)
        if coherence > best_coherence:
            best_coherence = coherence
            best_response = response_words

    # Create final response with quantum state indicators
    response = " ".join(best_response)
    agent_coherence = calculate_coherence(lead_agent.wave_fn.amplitude)
    convergence_indicator = "✓" if converged else "×"
    response += (
        f" |C={agent_coherence:.2f}, R={best_coherence:.2f}, F={convergence_indicator}⟩"
    )

    # Update word learner's recent words
    word_learner.recent_words.extend(best_response)
    if len(word_learner.recent_words) > word_learner.context_window:
        word_learner.recent_words = word_learner.recent_words[
            -word_learner.context_window :
        ]

    return response, active_agents


def process_file(file_path: str) -> Tuple[str, List[str], Dict[str, List[str]]]:
    """Process a single file and return its content, words, and transitions"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        words = content.split()
        transitions = {}

        # Track word transitions
        for i in range(len(words) - 1):
            word = words[i]
            next_word = words[i + 1]
            if word not in transitions:
                transitions[word] = []
            transitions[word].append(next_word)

        return content, words, transitions
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return "", [], {}


def create_word_vector(args) -> Tuple[str, np.ndarray]:
    """Create embedding for a single word using its context"""
    word, context_words, existing_vectors = args
    context_size = 5

    # Get context words from transitions
    context = set()
    context.update(context_words[:context_size])
    if len(context) >= context_size * 2:
        context = list(context)[: context_size * 2]

    # Create embedding based on context
    if context:
        embedding = np.mean(
            [existing_vectors.get(w, np.random.randn(50)) for w in context], axis=0
        )
    else:
        embedding = np.random.randn(50)

    return word, embedding / np.linalg.norm(embedding)


def process_repository_content() -> (
    Tuple[str, Dict[str, List[str]], Dict[str, np.ndarray]]
):
    """Process repository content to extract word vectors in parallel"""
    # Get all files
    files = get_repository_files()

    # Process files in parallel
    n_cores = max(1, cpu_count() - 1)  # Leave one core free
    print(f"\nProcessing files using {n_cores} cores...")

    with Pool(n_cores) as pool:
        # Phase 1: Parallel file processing
        file_results = pool.map(process_file, files)

        # Combine initial results
        all_text = ""
        word_vectors: Dict[str, np.ndarray] = {}
        word_transitions: Dict[str, List[str]] = {}

        # First pass: collect all words and their immediate contexts
        for content, words, transitions in file_results:
            all_text += content + "\n"

            # Create initial random vectors for all words
            for word in words:
                if word not in word_vectors:
                    word_vectors[word] = np.random.randn(50)

            # Collect transitions
            for word, next_words in transitions.items():
                if word not in word_transitions:
                    word_transitions[word] = []
                word_transitions[word].extend(next_words)

        # Second pass: update vectors based on context
        print(f"Creating word vectors for {len(word_vectors)} words...")

        for word in word_vectors:
            context_vectors = []

            # Forward context
            if word in word_transitions:
                for next_word in word_transitions[word]:
                    if next_word in word_vectors:
                        context_vectors.append(word_vectors[next_word])

            # Backward context
            for prev_word, next_words in word_transitions.items():
                if word in next_words and prev_word in word_vectors:
                    context_vectors.append(word_vectors[prev_word])

            # Update vector based on context
            if context_vectors:
                context_mean = np.mean(context_vectors, axis=0)
                word_vectors[word] = (
                    context_mean + 0.1 * word_vectors[word]
                )  # Keep some randomness
                # Normalize
                word_vectors[word] = word_vectors[word] / np.linalg.norm(
                    word_vectors[word]
                )

    return all_text, word_transitions, word_vectors


if __name__ == "__main__":
    print("Starting Repository-based Consciousness Emergence Experiment")
    print("========================================================")

    history, final_agents, knowledge = run_repository_consciousness()

    # Analyze results
    print("\nFinal Analysis:")
    print("===============")

    for category in set(h[0] for h in history):
        category_awareness = [h[2] for h in history if h[0] == category]
        print(f"\n{category.upper()}:")
        print(f"Files processed: {len(category_awareness)}")
        print(f"Average awareness: {np.mean(category_awareness):.3f}")
        print(f"Peak awareness: {max(category_awareness):.3f}")

        top_files = sorted(
            [(h[1], h[2]) for h in history if h[0] == category],
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        print("\nTop contributing files:")
        for file, awareness in top_files:
            print(f"- {file}: {awareness:.3f}")

    print("\nConsciousness has emerged. Starting interactive session...")
    interact_with_consciousness(final_agents, (32, 32), knowledge)
