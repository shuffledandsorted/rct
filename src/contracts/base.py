import numpy as np
from typing import Optional, Set, Callable

from .utils import Y
from ..quantum import WaveFunction


class QuantumContract:
    """Base class for quantum contracts with non-zero state initialization"""
    
    def __init__(self, 
                 agent1,
                 agent2,
                 initial_state: Optional[np.ndarray] = None,
                 dims: tuple = (32, 32)):
        self.agent1 = agent1
        self.agent2 = agent2
        self.dims = dims
        
        # Initialize with non-zero quantum state
        if initial_state is None:
            # Create superposition of basis states
            n = np.prod(dims)
            amplitude = np.ones(dims) / np.sqrt(n)
            
            # Add phase variation for richer state
            y, x = np.mgrid[0:dims[0], 0:dims[1]]
            phase = np.exp(1j * (x + y) * np.pi / max(dims))
            
            self.state = amplitude * phase
        else:
            self.state = initial_state
            
        # Verify non-zero state
        if np.all(np.abs(self.state) < 1e-10):
            raise ValueError("Contract must have non-zero initial state")
            
        # Initialize contract energy
        self.energy = np.sum(np.abs(self.state) ** 2)
        
    def evolve(self, steps: int = 1) -> np.ndarray:
        """Evolve contract state while preserving energy"""
        current_state = self.state
        
        for _ in range(steps):
            # Create energy-preserving transformation
            phase_shift = np.exp(2j * np.pi * np.random.random(self.dims))
            next_state = current_state * phase_shift
            
            # Verify energy conservation
            next_energy = np.sum(np.abs(next_state) ** 2)
            if abs(next_energy - self.energy) < 1e-10:
                current_state = next_state
                
        return current_state
    
    def measure_stability(self) -> float:
        """Measure contract stability through state evolution"""
        evolved = self.evolve(steps=5)
        stability = 1.0 - np.mean(np.abs(evolved - self.state))
        return float(stability)
    
    def interact(self, other: 'QuantumContract') -> 'QuantumContract':
        """Create interaction between contracts"""
        # Combine states through quantum interference
        combined_state = (self.state + other.state) / np.sqrt(2)
        
        # Ensure non-zero state is maintained
        if np.all(np.abs(combined_state) < 1e-10):
            raise ValueError("Interaction resulted in zero state")
            
        return QuantumContract(
            self.agent1,
            other.agent2,
            initial_state=combined_state,
            dims=self.dims
        ) 