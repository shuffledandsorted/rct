import numpy as np
from typing import Optional, Set, Callable

from .utils import Y
from ..quantum import WaveFunction


class QuantumContract:
    """Base class for quantum contracts with non-zero state initialization"""
    
    def __init__(self, agent1, agent2, wave_fn, energy_fn=None, transforms=None):
        self.agent1 = agent1
        self.agent2 = agent2
        self.psi = wave_fn
        self.energy = energy_fn or self._default_energy
        self.transforms = transforms or set()
        self._fixed_point = None
        
    def negotiate(self, max_iterations=100, tolerance=1e-6):
        """Implement recursive contract negotiation to find fixed point."""
        state = self.agent1.state
        prev_state = None
        iteration = 0
        
        while iteration < max_iterations:
            # Apply contract transformations
            new_state = self.psi(state)
            for transform in self.transforms:
                new_state = transform(new_state)
                
            # Check for fixed point convergence
            if prev_state is not None:
                diff = np.linalg.norm(new_state - prev_state)
                if diff < tolerance:
                    self._fixed_point = new_state
                    return True
                    
            prev_state = new_state
            state = new_state
            iteration += 1
            
        return False
        
    def compose(self, other):
        """Implement proper contract composition C₁ ∘ C₂."""
        if self.agent2 != other.agent1:
            raise ValueError("Contracts must share intermediate agent")
            
        def composed_wave_fn(state):
            intermediate = self.psi(state)
            return other.psi(intermediate)
            
        def composed_energy(state):
            return self.energy(state) + other.energy(state)
            
        # Combine transforms preserving order
        composed_transforms = self.transforms.union(other.transforms)
        
        return QuantumContract(
            self.agent1,
            other.agent2,
            composed_wave_fn,
            composed_energy,
            composed_transforms
        )
        
    def intersect(self, other):
        """Find intersection of contracts through fixed point."""
        if not (self.agent1 == other.agent1 and self.agent2 == other.agent2):
            raise ValueError("Can only intersect contracts between same agents")
            
        def intersection_wave_fn(state):
            # Implement Y combinator for fixed point
            state1 = self.psi(state)
            state2 = other.psi(state)
            return (state1 + state2) / 2  # Initial approximation
            
        def intersection_energy(state):
            return max(self.energy(state), other.energy(state))
            
        contract = QuantumContract(
            self.agent1,
            self.agent2,
            intersection_wave_fn,
            intersection_energy,
            self.transforms.union(other.transforms)
        )
        
        # Find fixed point
        if not contract.negotiate():
            raise ValueError("No stable intersection found")
            
        return contract
        
    def is_valid(self):
        """Check if contract has reached valid fixed point."""
        if self._fixed_point is None:
            return self.negotiate()
        return True
        
    def _default_energy(self, state):
        """Default energy measure based on wave function."""
        psi = self.psi(state)
        return np.sum(np.abs(psi) ** 2) 