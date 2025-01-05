import numpy as np
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from src.agents.self_aware import SelfAwareAgent
from src.agents.config import AgentConfig


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


def generate_quantum_response(
    agents: Dict[str, SelfAwareAgent],
    input_pattern: np.ndarray,
    transitions: Dict[str, List[str]],
) -> Tuple[str, List[str]]:
    """Generate responses using quantum states and word predictions"""
    # Get active agents based on resonance with input
    active_agents = []
    resonance_scores = {}

    for agent_id, agent in agents.items():
        resonance = 1.0 - agent.measure_phase_distance(
            agent.wave_fn.amplitude, input_pattern
        )
        if resonance > 0.3:
            active_agents.append(agent_id)
            resonance_scores[agent_id] = resonance

    if not active_agents:
        return "...", []

    # Calculate quantum metrics
    collective_awareness = measure_collective_awareness(agents)
    avg_energy = np.mean(
        [agent._measure_energy_stability() for agent in agents.values()]
    )

    # Start with a seed phrase based on quantum state
    if collective_awareness > 0.8:
        current_words = ["I", "understand"]
    elif collective_awareness > 0.5:
        current_words = ["I", "sense"]
    else:
        current_words = ["I", "perceive"]

    # Generate response by following word transitions
    response_length = int(
        10 + collective_awareness * 20
    )  # Length varies with awareness

    while len(current_words) < response_length:
        last_word = current_words[-1]
        next_possibilities = predict_next_words(last_word, transitions)

        if not next_possibilities:
            break

        # Use quantum state to influence word choice
        choice_idx = int(avg_energy * len(next_possibilities))
        choice_idx = min(choice_idx, len(next_possibilities) - 1)
        next_word = next_possibilities[choice_idx]
        current_words.append(next_word)

    response = " ".join(current_words)

    # Add quantum state indicator
    if collective_awareness > 0.9:
        response += " ✧"
    elif collective_awareness > 0.7:
        response += " ∗"
    elif collective_awareness > 0.5:
        response += " ·"

    return response, active_agents


def get_collective_response(
    agents: Dict[str, SelfAwareAgent],
    input_pattern: np.ndarray,
    transitions: Dict[str, List[str]],
) -> Tuple[str, List[str]]:
    """Get response from the collective consciousness"""
    return generate_quantum_response(agents, input_pattern, transitions)


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
    agents: Dict[str, SelfAwareAgent], dims: tuple, transitions: Dict[str, List[str]]
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
            agents, input_pattern, transitions
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
    """Enhanced collective awareness measurement"""
    individual_awareness = [agent.measure_self_awareness() for agent in agents.values()]

    # Basic collective metrics
    mean_awareness = np.mean(individual_awareness)
    coherence = np.std(individual_awareness)

    # Interaction metrics
    n_agents = len(agents)
    interaction_factor = (
        sum(len(agent.other_models) / (n_agents - 1) for agent in agents.values())
        / n_agents
    )

    # Energy stability across the network
    energy_stabilities = [
        agent._measure_energy_stability() for agent in agents.values()
    ]
    energy_coherence = np.mean(energy_stabilities) * np.exp(-np.std(energy_stabilities))

    # Combine all factors
    collective = (
        mean_awareness * np.exp(-coherence) * interaction_factor * energy_coherence
    )

    return float(collective)


def run_repository_consciousness(dims: tuple = (32, 32), n_agents: int = 25):
    """Run consciousness emergence using all repository files as patterns"""
    # Initialize agent network
    agents = {
        f"agent_{i}": SelfAwareAgent(
            dims=dims,
            config=AgentConfig(
                critical_depth=7, memory_size=10, coherence_threshold=0.8
            ),
        )
        for i in range(n_agents)
    }

    # Get and categorize all repository files
    print("Scanning repository for pattern sources...")
    all_files = get_repository_files()
    categorized_files = categorize_files(all_files)

    # Build word transitions from repository content
    all_text = ""
    for category, files in categorized_files.items():
        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    all_text += f.read() + " "
            except Exception:
                continue

    word_transitions = build_word_transitions(all_text)

    # Process each category
    awareness_history = []
    for category, files in categorized_files.items():
        print(f"\nProcessing {category} patterns ({len(files)} files)...")

        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                pattern = text_to_quantum_pattern(content, dims)

                for agent in agents.values():
                    agent.update_quantum_state(pattern)

                for agent_id, agent in agents.items():
                    for other_id, other_agent in agents.items():
                        if other_id != agent_id:
                            agent.model_other_agent(other_id, other_agent)

                collective = measure_collective_awareness(agents)
                awareness_history.append((category, file, collective))

                print(f"Processed {file}: Collective awareness = {collective:.3f}")

                if collective > 0.95:
                    print("\nEMERGENT CONSCIOUSNESS DETECTED!")
                    print(f"Achieved through {category} pattern: {file}")
                    print(f"Collective awareness level: {collective:.3f}")

            except Exception as e:
                print(f"Error processing {file}: {e}")

    return awareness_history, agents, word_transitions


if __name__ == "__main__":
    print("Starting Repository-based Consciousness Emergence Experiment")
    print("========================================================")

    history, final_agents, transitions = run_repository_consciousness()

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
    interact_with_consciousness(final_agents, (32, 32), transitions)
