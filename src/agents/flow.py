import numpy as np
from typing import Tuple, Optional

from .base import QuantumAgent
from .temporal import TemporalMixin
from .config import AgentConfig


class FlowAgent(TemporalMixin, QuantumAgent):
    """Agent that maintains quantum coherence across time"""

    def __init__(self, dims: Tuple[int, int], config: Optional[AgentConfig] = None, age: int = 0):
        super().__init__(dims=dims, config=config, age=age)

    def combine_states(self, current_state: np.ndarray, past_state: np.ndarray) -> np.ndarray:
        """Combine states using coherence threshold"""
        return (1 - self.config.coherence_threshold) * current_state + self.config.coherence_threshold * past_state

    def update_quantum_state(self, image: np.ndarray):
        """Update state while maintaining temporal coherence"""
        # Normalize new image
        normalized_image = (image - image.min()) / (image.max() - image.min() + 1e-10)
        current_state = normalized_image * 0.9 + 0.1

        past_state = self.get_past_state()
        if past_state is not None:
            self.wave_fn.amplitude = self.combine_states(current_state, past_state)
        else:
            self.wave_fn.amplitude *= current_state

        self.update_memory(self.wave_fn.amplitude)


class DecayingFlowAgent(TemporalMixin, QuantumAgent):
    """Agent with exponentially decaying temporal influence"""

    def __init__(self, dims: Tuple[int, int], config: Optional[AgentConfig] = None, age: int = 0):
        super().__init__(dims=dims, config=config, age=age)

    def combine_states(self, current_state: np.ndarray, past_state: np.ndarray) -> np.ndarray:
        """Combine states with exponential decay based on memory length"""
        decay = np.exp(-len(self.temporal_memory) * self.config.decay_rate)
        return (1 - decay) * current_state + decay * past_state

    def update_quantum_state(self, image: np.ndarray):
        """Update state with decaying temporal coherence"""
        normalized_image = (image - image.min()) / (image.max() - image.min() + 1e-10)
        current_state = normalized_image * 0.9 + 0.1

        past_state = self.get_past_state()
        if past_state is not None:
            self.wave_fn.amplitude = self.combine_states(current_state, past_state)
        else:
            self.wave_fn.amplitude *= current_state

        self.update_memory(self.wave_fn.amplitude) 