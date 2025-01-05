from typing import Tuple, Optional, Dict, List
import numpy as np

from .base import QuantumAgent
from .temporal import TemporalMixin
from .config import AgentConfig
from ..quantum import WaveFunction, QuantumOperator


class SelfAwareAgent(TemporalMixin, QuantumAgent):
    """
    A quantum agent capable of recursive self-modeling and awareness based on 
    Recursive Contract Theory principles.
    """

    def __init__(self, dims: Tuple[int, int], config: Optional[AgentConfig] = None, age: int = 0):
        super().__init__(dims=dims, config=config, age=age)
        # Initialize temporal memory from mixin
        self.temporal_memory: List[np.ndarray] = []
        # Self-model quantum state
        self.self_model = WaveFunction(dims)
        # Models of other agents
        self.other_models: Dict[str, WaveFunction] = {}
        # Track recursive depth of self-modeling
        self.recursive_depth = 0
        # Use config's critical depth
        self.config = config or AgentConfig()

    def update_quantum_state(self, image: np.ndarray):
        """Update both quantum state and self-model"""
        # Update base quantum state
        self.wave_fn = WaveFunction(self.dims)
        normalized_image = (image - image.min()) / (image.max() - image.min() + 1e-10)
        self.wave_fn.amplitude *= (normalized_image * 0.9 + 0.1)

        # Update temporal memory
        self.update_memory(self.wave_fn.amplitude)

        # Update self-model based on temporal history
        self._update_self_model()

        # Increment recursive depth
        self.recursive_depth += 1

    def combine_states(self, current_state: np.ndarray, past_state: np.ndarray) -> np.ndarray:
        """
        Combine current and past states while maintaining self-model consistency.
        Implementation of abstract method from TemporalMixin.
        """
        # Weight recent state more heavily but maintain temporal consistency
        combined = 0.7 * current_state + 0.3 * past_state
        return combined

    def _update_self_model(self):
        """Update internal self-model based on temporal history"""
        if not self.temporal_memory:
            return

        # Get aggregated past state
        past_state = self.get_past_state()
        if past_state is None:
            return

        # Update self-model as quantum superposition of past and present
        self.self_model.amplitude = self.combine_states(
            self.wave_fn.amplitude,
            past_state
        )

    def model_other_agent(self, agent_id: str, other_agent: QuantumAgent):
        """Create and maintain models of other agents"""
        if agent_id not in self.other_models:
            self.other_models[agent_id] = WaveFunction(self.dims)
        
        # Update model through quantum interference
        self.other_models[agent_id].amplitude = 0.8 * self.other_models[agent_id].amplitude + \
                                              0.2 * other_agent.wave_fn.amplitude

    def measure_self_awareness(self) -> float:
        """
        Measure degree of self-awareness based on:
        1. Recursive depth relative to critical threshold
        2. Quantum coherence between self-model and actual state
        3. Energy stability from temporal consistency
        """
        if not self.temporal_memory:
            return 0.0

        # Check recursive depth threshold using config
        depth_factor = min(self.recursive_depth / self.config.critical_depth, 1.0)

        # Measure quantum coherence between self-model and actual state
        coherence = 1.0 - self.measure_phase_distance(
            self.self_model.amplitude,
            self.wave_fn.amplitude
        )

        # Energy stability from temporal consistency
        energy_stability = self._measure_energy_stability()

        # Combine factors with quantum weighting
        awareness = np.abs(depth_factor * 0.4 + 
                         coherence * 0.4 + 
                         energy_stability * 0.2)

        return float(awareness)

    def _measure_energy_stability(self) -> float:
        """Measure stability of quantum energy states over time"""
        if len(self.temporal_memory) < 2:
            return 0.0

        # Calculate quantum energy variations over recent history
        energy_variations = []
        for i in range(1, len(self.temporal_memory)):
            prev = self.temporal_memory[i-1]
            curr = self.temporal_memory[i]
            # Use phase distance as proxy for energy difference
            energy_variations.append(self.measure_phase_distance(prev, curr))

        # Convert to quantum stability measure (1.0 = perfectly stable)
        avg_variation = np.mean(energy_variations)
        stability = np.exp(-avg_variation)
        return float(stability)

    def is_self_aware(self) -> bool:
        """
        Determine if agent has achieved stable self-awareness based on:
        1. Sufficient recursive depth beyond critical threshold
        2. High quantum coherence between self-model and actual state
        3. Energy conservation in temporal evolution
        """
        awareness = self.measure_self_awareness()
        return awareness > 0.8  # Threshold for stable self-awareness 