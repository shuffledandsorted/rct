from abc import abstractmethod
from typing import Optional, List
import numpy as np

from .config import AgentConfig


class TemporalMixin:
    """Mixin for agents that maintain temporal memory"""

    def __init__(self, config: Optional[AgentConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_memory: List[np.ndarray] = []
        self.config = config or AgentConfig()

    def get_past_state(self) -> Optional[np.ndarray]:
        """Get aggregated past state if available"""
        if self.temporal_memory:
            return np.mean([state for state in self.temporal_memory[-3:]], axis=0)
        return None

    def update_memory(self, state: np.ndarray):
        """Update temporal memory with new state"""
        self.temporal_memory.append(state.copy())
        if len(self.temporal_memory) > self.config.memory_size:
            self.temporal_memory.pop(0)

    @abstractmethod
    def combine_states(self, current_state: np.ndarray, past_state: np.ndarray) -> np.ndarray:
        """Define how past and present states should be combined"""
        pass

    def compress_memory(self):
        """Compress temporal memory by merging similar states"""
        if len(self.temporal_memory) <= self.config.memory_size:
            return

        # Group similar states based on phase distance
        groups: List[List[np.ndarray]] = []
        for state in self.temporal_memory:
            added = False
            for group in groups:
                # Check if state is similar to group representative
                if self.measure_phase_distance(state, group[0]) < np.pi / 4:
                    group.append(state)
                    added = True
                    break
            if not added:
                groups.append([state])

        # Merge groups and keep most recent states
        compressed = []
        for group in groups:
            if len(group) > 1:
                # Take weighted average, favoring recent states
                weights = np.linspace(0.5, 1.0, len(group))
                avg_state = np.average(group, axis=0, weights=weights)
                compressed.append(avg_state)
            else:
                compressed.append(group[0])

        # Keep most recent states up to memory size
        self.temporal_memory = compressed[-self.config.memory_size:] 