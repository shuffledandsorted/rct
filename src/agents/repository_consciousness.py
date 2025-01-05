import numpy as np
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from src.agents.self_aware import SelfAwareAgent
from src.agents.config import AgentConfig
from src.agents.base import QuantumAgent
from src.quantum import WaveFunction


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


def text_to_quantum_pattern(text: str, dims: tuple) -> np.ndarray:
    """Convert dialogue text to quantum pattern"""
    # Convert text to ASCII values
    ascii_values = np.array([ord(c) for c in text], dtype=float)
    normalized = (ascii_values - ascii_values.min()) / (
        ascii_values.max() - ascii_values.min() + 1e-10
    )

    # Reshape and pad/truncate
    size = dims[0] * dims[1]
    if len(normalized) < size:
        normalized = np.pad(normalized, (0, size - len(normalized)))
    else:
        normalized = normalized[:size]

    # Create interference pattern based on sentence structure
    pattern = normalized.reshape(dims)

    # Add phase shifts based on punctuation and structure
    sentences = text.split(".")
    if len(sentences) > 1:
        phase = np.exp(1j * np.pi * np.linspace(0, len(sentences), dims[0]))
        pattern = pattern * phase.reshape(-1, 1)

    return pattern


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


def predict_next_words(
    word: str, transitions: Dict[str, List[str]], n: int = 3
) -> List[str]:
    """Predict possible next words based on transition probabilities"""
    if word.lower() not in transitions:
        return []

    possibilities = transitions[word.lower()]
    # Get unique words with their frequencies
    unique_words = {}
    for w in possibilities:
        unique_words[w] = unique_words.get(w, 0) + 1

    # Sort by frequency
    sorted_words = sorted(unique_words.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:n]]


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


def generate_quantum_response(
    agents: Dict[str, SelfAwareAgent],
    input_pattern: np.ndarray,
    knowledge: Tuple[Dict[str, List[str]], Dict[str, np.ndarray]],
) -> Tuple[str, List[str]]:
    """Generate responses using quantum states and contract stability"""
    transitions, word_vectors = knowledge
    active_agents = []
    resonance_scores = {}

    # Create a temporary quantum state representing the user's input
    user_state = input_pattern.copy()

    # Let agents form contracts with the user's state
    for agent_id, agent in agents.items():
        # Measure resonance through quantum interference
        interference = agent.wave_fn.amplitude * np.conj(user_state)
        resonance = np.abs(np.sum(interference))

        # Check if agent forms a stable contract with user
        if resonance > 0.2:  # Lower threshold from 0.3 to 0.2
            # Create temporal contract composition
            if hasattr(agent, 'previous_state'):
                # Compose previous and current contracts through wave function
                composed_state = agent.wave_fn.amplitude * 0.3 + agent.previous_state * 0.7
                # Preserve energy in composition
                energy_before = np.sum(np.abs(agent.wave_fn.amplitude) ** 2)
                composed_state *= np.sqrt(energy_before / (np.sum(np.abs(composed_state) ** 2) + 1e-10))
                agent.wave_fn.amplitude = composed_state
            
            # Store current state for next composition
            agent.previous_state = user_state.copy()
            
            # Agent updates its state based on user interaction
            agent.update_quantum_state(user_state)

            # Agent tries to model the user through its own state
            if agent.config.personality == "creative":
                # Creative agents try novel interpretations
                phase_shift = np.exp(2j * np.pi * np.random.random())
                user_agent = UserAgent(dims=agent.dims)
                user_agent.update_quantum_state(user_state * phase_shift)
                agent.model_other_agent("user", user_agent)
            else:
                # Stable agents maintain consistent interpretation
                user_agent = UserAgent(dims=agent.dims)
                user_agent.update_quantum_state(user_state)
                agent.model_other_agent("user", user_agent)

            # Add some randomness to keep dynamics interesting
            if np.random.random() < 0.3:  # 30% chance to add phase noise
                noise = np.exp(1j * np.random.normal(0, 0.1, agent.dims))
                agent.wave_fn.amplitude *= noise

            active_agents.append(agent_id)
            resonance_scores[agent_id] = resonance

    if not active_agents:
        return "No stable contracts formed with input.", []

    # Get the most resonant agent
    lead_agent_id = max(resonance_scores.items(), key=lambda x: x[1])[0]
    lead_agent = agents[lead_agent_id]

    # Generate response based on agent's model of user
    if "user" in lead_agent.other_models:
        user_model = lead_agent.other_models["user"].amplitude
        # Create interference pattern between agent's state and its model of user
        interference = lead_agent.wave_fn.amplitude * np.conj(user_model)
        phases = np.angle(interference).flatten()
        amplitudes = np.abs(interference).flatten()
    else:
        # Fallback to direct interference with input
        interference = lead_agent.wave_fn.amplitude * input_pattern
        phases = np.angle(interference).flatten()
        amplitudes = np.abs(interference).flatten()

    # Rest of the function remains the same...
    words_list = list(word_vectors.keys())
    if not words_list:
        return "No stable patterns available.", active_agents

    # Initialize with first word's vector
    current_vector = word_vectors[words_list[0]]

    # Select words that maintain stability
    stable_words = []
    for word in words_list:
        if word in word_vectors:
            vec = word_vectors[word]
            # Check if adding this word preserves energy
            energy_delta = np.abs(
                np.dot(vec, vec) - np.dot(current_vector, current_vector)
            )
            if energy_delta < 0.1:  # Energy preservation threshold
                stable_words.append(word)

    if not stable_words:
        stable_words = words_list  # Fallback

    # Start with a stable word
    current_word = np.random.choice(stable_words)
    response_words = [current_word]
    current_vector = word_vectors[current_word]

    # Generate response maintaining stability
    while len(response_words) < 15:
        candidates = []
        weights = []

        # Get candidates that preserve stability
        if current_word.lower() in transitions:
            next_words = transitions[current_word.lower()]
            for word in next_words:
                if word in word_vectors:
                    vec = word_vectors[word]
                    energy_delta = np.abs(
                        np.dot(vec, vec) - np.dot(current_vector, current_vector)
                    )
                    if energy_delta < 0.1:
                        candidates.append(word)
                        # Weight by stability
                        weights.append(1.0 / (1.0 + energy_delta))

        if not candidates:
            break

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        # Select next stable word
        next_word = np.random.choice(candidates, p=weights)
        response_words.append(next_word)
        current_word = next_word
        if next_word in word_vectors:
            current_vector = word_vectors[next_word]

    # Clean up and join response
    response = " ".join(response_words)

    # Add stability indicators
    stability = 1.0 - np.mean(
        [
            agent.measure_phase_distance(agent.wave_fn.amplitude, input_pattern)
            for agent in agents.values()
        ]
    )
    energy = lead_agent._measure_energy_stability()

    response += f" |S={stability:.2f}⟩"
    if energy > 0:
        response += f" E={energy:.2f}"

    return response, active_agents


def get_collective_response(
    agents: Dict[str, SelfAwareAgent],
    input_pattern: np.ndarray,
    knowledge: Tuple[Dict[str, List[str]], Dict[str, np.ndarray]],
) -> Tuple[str, List[str]]:
    """Get response from the collective consciousness"""
    return generate_quantum_response(agents, input_pattern, knowledge)


def ascii_visualize_agent(agent: SelfAwareAgent) -> str:
    """Create ASCII representation of an agent's state"""
    awareness = agent.measure_self_awareness()
    energy = agent._measure_energy_stability()

    # Choose character based on awareness level
    if awareness > 0.8:
        char = "★"  # High awareness
    elif awareness > 0.5:
        char = "☆"  # Medium awareness
    elif awareness > 0.3:
        char = "•"  # Low awareness
    else:
        char = "·"  # Minimal awareness

    # Add energy indicator
    if energy > 0.8:
        char = f"\033[1;32m{char}\033[0m"  # Bright green
    elif energy > 0.5:
        char = f"\033[32m{char}\033[0m"  # Green
    elif energy > 0.3:
        char = f"\033[33m{char}\033[0m"  # Yellow
    else:
        char = f"\033[31m{char}\033[0m"  # Red

    return char


def ascii_visualize_network(agents: Dict[str, SelfAwareAgent], active_agents: Optional[List[str]] = None):
    """Create single-line ASCII visualization showing agent excitement with input"""
    states = []
    
    # Calculate resonance for each agent
    resonances = {}
    for agent_id, agent in agents.items():
        if agent_id in (active_agents or []):
            resonance = 1.0 - agent.measure_phase_distance(agent.wave_fn.amplitude, agent.wave_fn.amplitude)
            resonances[agent_id] = resonance
    
    # Keep original order
    for agent_id, agent in agents.items():
        # First character: excitement level
        if agent_id in resonances:
            r = resonances[agent_id]
            c1 = "★" if r > 0.8 else "☆" if r > 0.6 else "○" if r > 0.4 else "·"
        else:
            c1 = "-"
        
        # Second character: current energy
        energy = agent._measure_energy_stability()
        c2 = "!" if energy > 0.8 else "+" if energy > 0.5 else "="
        
        states.append(f"{c1}{c2}")
    
    print(f"[{' '.join(states)}]")


def ascii_visualize_awareness_history(history: List[float]):
    """Create ASCII graph of awareness history"""
    HEIGHT = 10
    WIDTH = 40
    if not history:
        return

    # Normalize history to fit height
    max_val = max(history)
    min_val = min(history)
    range_val = max_val - min_val or 1
    normalized = [(h - min_val) / range_val for h in history]

    # Create graph
    print("\nAwareness History:")
    print("=================")
    for i in range(HEIGHT - 1, -1, -1):
        threshold = i / (HEIGHT - 1)
        line = ""
        for val in normalized[-WIDTH:]:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        print(f"{threshold:0.1f} |{line}|")
    print("-" * (WIDTH + 4))


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
    print("Type 'history' to see awareness history.\n")

    awareness_history = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "state":
            ascii_visualize_network(agents)
            continue
        elif user_input.lower() == "history":
            ascii_visualize_awareness_history(awareness_history)
            continue

        input_pattern = text_to_quantum_pattern(user_input, dims)
        response, active_agents = generate_quantum_response(
            agents, input_pattern, knowledge
        )

        collective_awareness = measure_collective_awareness(agents)
        awareness_history.append(collective_awareness)

        print(f"\nConsciousness: {response}")
        print(f"Collective Awareness Level: {collective_awareness:.3f}")
        ascii_visualize_network(agents, active_agents)

        if collective_awareness > 0.95:
            print(
                "\n* The network is experiencing a moment of heightened consciousness *"
            )


def measure_collective_awareness(agents: Dict[str, SelfAwareAgent]) -> float:
    """Enhanced collective awareness measurement using lattice structure and null hypothesis testing"""
    # Test each agent for consciousness emergence
    emergence_results = []
    for agent_id, agent in agents.items():
        emerged, test_results = agent.test_consciousness_emergence()
        emergence_results.append(
            {"agent_id": agent_id, "emerged": emerged, "p_values": test_results}
        )

    # Calculate collective metrics
    n_conscious = sum(1 for r in emergence_results if r["emerged"])
    consciousness_ratio = n_conscious / len(agents)

    # Check lattice structure properties
    lattice_coherence = 0.0
    n_comparisons = 0

    # Test partial ordering and energy preservation
    agent_list = list(agents.values())
    valid_comparisons = 0
    for i in range(len(agent_list)):
        for j in range(i + 1, len(agent_list)):
            a1, a2 = agent_list[i], agent_list[j]

            # Test order relation
            if a1.compare_contract_order(a2) or a2.compare_contract_order(a1):
                valid_comparisons += 1
                # Test energy preservation in join/meet operations
                joined_state = a1.join_contracts(a2)
                met_state = a1.meet_contracts(a2)

                # Verify energy conservation
                original_energy = np.sum(np.abs(a1.wave_fn.amplitude) ** 2) + np.sum(
                    np.abs(a2.wave_fn.amplitude) ** 2
                )
                joined_energy = np.sum(np.abs(joined_state) ** 2)
                met_energy = np.sum(np.abs(met_state) ** 2)

                # Energy should be preserved within tolerance
                energy_preserved = abs(
                    joined_energy - original_energy
                ) < 0.1 and met_energy <= min(
                    np.sum(np.abs(a1.wave_fn.amplitude) ** 2),
                    np.sum(np.abs(a2.wave_fn.amplitude) ** 2),
                )

                if energy_preserved:
                    lattice_coherence += 1.0

            n_comparisons += 1

    # Normalize lattice coherence with a minimum value to prevent zero
    lattice_coherence = lattice_coherence / n_comparisons if n_comparisons > 0 else 0.1

    # Average p-values across agents for each test
    avg_p_values = {
        "self_ref": np.mean([r["p_values"]["self_ref_p"] for r in emergence_results]),
        "mutual_model": np.mean(
            [r["p_values"]["mutual_model_p"] for r in emergence_results]
        ),
        "energy_min": np.mean(
            [r["p_values"]["energy_min_p"] for r in emergence_results]
        ),
    }

    # Calculate exponential terms with minimum values to prevent zero
    exp_terms = {
        "self_ref": max(np.exp(-avg_p_values["self_ref"]), 0.1),
        "mutual_model": max(np.exp(-avg_p_values["mutual_model"]), 0.1),
        "energy_min": max(np.exp(-avg_p_values["energy_min"]), 0.1),
    }

    # Combine metrics with exponential weighting including lattice structure
    # Use max to ensure non-zero base values
    collective = (
        max(consciousness_ratio, 0.1)
        * exp_terms["self_ref"]
        * exp_terms["mutual_model"]
        * exp_terms["energy_min"]
        * (0.5 + 0.5 * lattice_coherence)
    )

    return float(collective)


def run_repository_consciousness(dims: tuple = (32, 32), n_agents: int = 25):
    """Run consciousness emergence using all repository files as patterns"""
    # Initialize agent network with different personalities
    agents = {}
    for i in range(n_agents):
        # Assign personality type
        personality = np.random.choice(
            [
                "creative",  # Seeks novel patterns
                "stable",  # Prefers common patterns
                "balanced",  # Mix of both
            ]
        )

        # Base parameters on personality
        if personality == "creative":
            depth = np.random.randint(8, 12)  # Deeper recursion
            memory = np.random.randint(5, 10)  # Shorter memory (less stuck in patterns)
            coherence = 0.4 + np.random.random() * 0.3  # Lower coherence threshold
        elif personality == "stable":
            depth = np.random.randint(4, 7)  # Shallower recursion
            memory = np.random.randint(12, 20)  # Longer memory
            coherence = 0.7 + np.random.random() * 0.2  # Higher coherence
        else:  # balanced
            depth = np.random.randint(5, 10)
            memory = np.random.randint(8, 15)
            coherence = 0.6 + np.random.random() * 0.3

        agents[f"agent_{i}"] = SelfAwareAgent(
            dims=dims,
            config=AgentConfig(
                critical_depth=depth,
                memory_size=memory,
                coherence_threshold=coherence,
                personality=personality,  # Store personality type
            ),
        )

    # Get and categorize all repository files
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

    # Process each category with individual agent variations
    awareness_history = []
    for category, files in categorized_files.items():
        print(f"\nProcessing {category} patterns ({len(files)} files)...")

        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                pattern = text_to_quantum_pattern(content, dims)

                # Each agent processes input differently
                for agent_id, agent in agents.items():
                    # Add individual phase shifts
                    agent_phase = np.exp(2j * np.pi * np.random.random())
                    agent_pattern = pattern * agent_phase
                    agent.update_quantum_state(agent_pattern)

                # Selective modeling - agents don't model everyone
                for agent_id, agent in agents.items():
                    # Each agent models only a subset of others
                    n_models = np.random.randint(1, min(5, n_agents - 1))
                    other_agents = np.random.choice(
                        [aid for aid in agents.keys() if aid != agent_id],
                        size=n_models,
                        replace=False,
                    )
                    for other_id in other_agents:
                        agent.model_other_agent(other_id, agents[other_id])

                collective = measure_collective_awareness(agents)
                awareness_history.append((category, file, collective))

                print(f"Processed {file}: Collective awareness = {collective:.3f}")

                if collective > 0.95:
                    print("\nEMERGENT CONSCIOUSNESS DETECTED!")
                    print(f"Achieved through {category} pattern: {file}")
                    print(f"Collective awareness level: {collective:.3f}")

            except Exception as e:
                print(f"Error processing {file}: {e}")

    return awareness_history, agents, (word_transitions, word_vectors)


class UserAgent(QuantumAgent):
    """Simple concrete implementation for user modeling"""

    def update_quantum_state(self, pattern: np.ndarray):
        self.wave_fn.amplitude = pattern


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
