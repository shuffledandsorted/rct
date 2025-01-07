import numpy as np
from typing import List, Tuple, Optional
from .base import QuantumContract
from ..quantum.utils import quantum_normalize, calculate_geodesic_collapse

class EntanglementContract(QuantumContract):
    """Contract for managing quantum entanglement between agents."""
    
    def __init__(self, agent1, agent2, entanglement_strength: float = 0.5):
        # Create wave function for entangled state
        wave_fn = lambda state: np.kron(agent1.wave_fn.amplitude, agent2.wave_fn.amplitude)
        super().__init__(agent1, agent2, wave_fn=wave_fn)
        self.entanglement_strength = entanglement_strength
        self.bell_states = self._create_bell_states()
        
    def _create_bell_states(self) -> List[np.ndarray]:
        """Create Bell states for entanglement basis."""
        dims = self.agent1.dims
        size = np.prod(dims)
        
        # Create basic Bell states
        psi_plus = np.zeros(size * 2, dtype=np.complex128)
        psi_minus = np.zeros(size * 2, dtype=np.complex128)
        phi_plus = np.zeros(size * 2, dtype=np.complex128)
        phi_minus = np.zeros(size * 2, dtype=np.complex128)
        
        # Set up superpositions
        for i in range(size):
            # |ψ⁺⟩ = (|01⟩ + |10⟩)/√2
            psi_plus[i] = 1/np.sqrt(2)
            psi_plus[size+i] = 1/np.sqrt(2)
            
            # |ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            psi_minus[i] = 1/np.sqrt(2)
            psi_minus[size+i] = -1/np.sqrt(2)
            
            # |φ⁺⟩ = (|00⟩ + |11⟩)/√2
            phi_plus[i*2] = 1/np.sqrt(2)
            phi_plus[i*2+1] = 1/np.sqrt(2)
            
            # |φ⁻⟩ = (|00⟩ - |11⟩)/√2
            phi_minus[i*2] = 1/np.sqrt(2)
            phi_minus[i*2+1] = -1/np.sqrt(2)
            
        return [psi_plus, psi_minus, phi_plus, phi_minus]
    
    def measure_entanglement(self) -> float:
        """Measure degree of entanglement between agents."""
        # Calculate reduced density matrix
        joint_state = np.kron(self.agent1.wave_fn.amplitude, self.agent2.wave_fn.amplitude)
        density = np.outer(joint_state, np.conj(joint_state))
        
        # Calculate von Neumann entropy
        eigenvals = np.linalg.eigvalsh(density)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Remove zero eigenvalues
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        
        return float(entropy)
    
    def apply_entanglement(self) -> Tuple[np.ndarray, np.ndarray]:
        """Apply entanglement operation to both agents."""
        # Choose Bell state based on current states
        overlaps = []
        for bell_state in self.bell_states:
            joint_state = np.kron(self.agent1.wave_fn.amplitude, self.agent2.wave_fn.amplitude)
            overlap = np.abs(np.sum(np.conj(bell_state) * joint_state))
            overlaps.append(overlap)
            
        # Use Bell state with maximum overlap
        best_bell = self.bell_states[np.argmax(overlaps)]
        
        # Create entangled state
        joint_state = np.kron(self.agent1.wave_fn.amplitude, self.agent2.wave_fn.amplitude)
        entangled = calculate_geodesic_collapse(joint_state, best_bell, self.entanglement_strength)
        
        # Split back into individual states
        size = np.prod(self.agent1.dims)
        state1 = quantum_normalize(entangled[:size])
        state2 = quantum_normalize(entangled[size:])
        
        return state1, state2
    
    def negotiate(self, max_iterations: int = 100, tolerance: float = 1e-6) -> bool:
        """Negotiate entanglement between agents."""
        initial_entropy = self.measure_entanglement()
        
        for _ in range(max_iterations):
            # Apply entanglement operation
            state1, state2 = self.apply_entanglement()
            
            # Update agent states
            self.agent1.wave_fn.amplitude = state1
            self.agent2.wave_fn.amplitude = state2
            
            # Check convergence
            new_entropy = self.measure_entanglement()
            if abs(new_entropy - initial_entropy) < tolerance:
                return True
            initial_entropy = new_entropy
            
        return False 