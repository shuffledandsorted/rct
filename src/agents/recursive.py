import numpy as np
from typing import Tuple, Optional

from .base import QuantumAgent
from .config import AgentConfig
from ..quantum import WaveFunction


class RecursiveAgent(QuantumAgent):
    """Agent that recursively spawns child agents to track features"""

    def __init__(self, dims: Tuple[int, int], config: Optional[AgentConfig] = None, age: int = 0):
        super().__init__(dims=dims, config=config, age=age)

    def update_quantum_state(self, image: np.ndarray):
        """Reset and update state based only on current image"""
        self.wave_fn = WaveFunction(self.dims)
        
        # Normalize image to [0.1, 1.0] range, maintaining quantum vacuum-like minimum
        normalized_image = (image - image.min()) / (image.max() - image.min() + 1e-10)
        self.wave_fn.amplitude *= (normalized_image * 0.9 + 0.1)  # Map to [0.1, 1.0] 