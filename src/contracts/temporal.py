import numpy as np
from typing import Optional, Callable, Any
import time

from ..agents.temporal import TemporalMixin
from ..agents.config import AgentConfig
from .base import QuantumContract
from ..quantum import WaveFunction


class TemporalContract(QuantumContract, TemporalMixin):
    """Contract that maintains temporal memory of quantum states."""

    def __init__(
        self,
        agent1: Any,
        agent2: Any,
        wave_function: Optional[Callable] = None,
        energy_measure=None,
        allowed_transforms=None,
        config: Optional[AgentConfig] = None,
        lifetime: float = 10.0,
    ):
        QuantumContract.__init__(self, agent1, agent2, wave_function, energy_measure, allowed_transforms)
        TemporalMixin.__init__(self, config or AgentConfig())
        self.creation_time = time.time()
        self.lifetime = lifetime  # Lifetime in seconds
        self.wave_fn = WaveFunction(agent1.dims)

    def apply(self):
        """Execute contract interaction with temporal memory."""
        if not self.is_valid():
            raise ValueError("Cannot apply invalid contract")

        # Get current states
        new_state1 = self.psi(self.agent1.wave_fn.amplitude)
        new_state2 = self.psi(self.agent2.wave_fn.amplitude)

        # Update temporal memory
        self.update_memory(new_state1)

        # Combine with past states if available
        past_state = self.get_past_state()
        if past_state is not None:
            new_state1 = self.combine_states(new_state1, past_state)
            new_state2 = self.combine_states(new_state2, past_state)

        # Update agents
        self.agent1.wave_fn.amplitude = new_state1
        self.agent2.wave_fn.amplitude = new_state2

        # Compress memory periodically
        if len(self.temporal_memory) >= self.config.memory_size:
            self.compress_memory()

        return self.energy(new_state1)

    def combine_states(self, current_state, past_state):
        """Combine current and past states with phase coherence."""
        # Weight recent states more heavily
        coherence = self.config.coherence_threshold
        return (1 - coherence) * current_state + coherence * past_state

    def is_valid(self):
        """Check if contract is valid including temporal coherence."""
        # First check basic validity
        if not super().is_valid():
            return False

        # Then check temporal stability if we have history
        if len(self.temporal_memory) >= 2:
            recent_states = self.temporal_memory[-2:]
            phase_diffs = [
                np.angle(s1 @ np.conjugate(s2))
                for s1, s2 in zip(recent_states[:-1], recent_states[1:])
            ]
            # Check if phase differences are consistent
            return np.std(phase_diffs) < np.pi / 4

        return True 

    def evolve(self, dt: float) -> None:
        """Evolve the contract's quantum state"""
        if self.psi is not None and self.wave_fn is not None:
            # Apply wave function evolution
            self.wave_fn.amplitude = self.psi(self.wave_fn.amplitude)
            # Apply decoherence based on age
            age = time.time() - self.creation_time
            if age < self.lifetime:
                decoherence = np.exp(-age / self.lifetime)
                self.wave_fn.amplitude *= decoherence
                # Normalize
                norm = np.linalg.norm(self.wave_fn.amplitude)
                if norm > 0:
                    self.wave_fn.amplitude /= norm
