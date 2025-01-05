import numpy as np
import pytest
from src.contracts import (
    create_measurement_contract,
    create_evolution_contract,
    create_entanglement_contract
)


class MockAgent:
    """Mock agent for testing contract creation."""
    def __init__(self, state=None):
        self.state = state or np.array([1.0, 0.0], dtype=np.complex128)
        self.basis = np.eye(2)
        
    def update_state(self, new_state):
        self.state = new_state
        
    def possible_states(self):
        return [np.array([1.0, 0.0]), np.array([0.0, 1.0])]


def test_measurement_contract():
    """Test creation of measurement contracts."""
    observer = MockAgent(np.array([1.0, 0.0]))
    system = MockAgent(np.array([0.0, 1.0]))
    
    contract = create_measurement_contract(observer, system)
    
    # Test measurement interaction
    measured_state = contract.psi(system.state)
    assert np.allclose(measured_state, np.array([0.0]))  # Projection onto observer basis
    
    # Test energy is information theoretic
    energy = contract.energy(system.state)
    assert energy > 0  # Should be positive for non-zero state


def test_evolution_contract():
    """Test creation of evolution contracts."""
    agent1 = MockAgent(np.array([1.0, 0.0], dtype=np.complex128))
    agent2 = MockAgent(np.array([0.0, 1.0], dtype=np.complex128))
    
    # Test with custom Hamiltonian
    hamiltonian = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # Pauli X
    contract = create_evolution_contract(agent1, agent2, hamiltonian=hamiltonian)
    
    # Evolution should flip the state
    evolved_state = contract.psi(agent1.state)
    assert np.allclose(evolved_state, np.array([0.0, 1.0]))
    
    # Energy should be real
    energy = contract.energy(agent1.state)
    assert np.isreal(energy)


def test_entanglement_contract():
    """Test creation of entanglement contracts."""
    agent1 = MockAgent(np.array([1.0, 0.0], dtype=np.complex128))
    agent2 = MockAgent(np.array([0.0, 1.0], dtype=np.complex128))
    
    contract = create_entanglement_contract(agent1, agent2)
    
    # Test entangled state creation
    entangled_state = contract.psi(agent1.state)
    
    # Should be in a superposition
    assert len(entangled_state) == 4  # 2x2 dimensional space
    assert not np.allclose(entangled_state, np.zeros_like(entangled_state))
    
    # Test entanglement entropy
    energy = contract.energy(entangled_state)
    assert energy > 0  # Should have non-zero entanglement


def test_contract_transforms():
    """Test that all contracts include basic quantum transforms."""
    agent1 = MockAgent()
    agent2 = MockAgent()
    
    contracts = [
        create_measurement_contract(agent1, agent2),
        create_evolution_contract(agent1, agent2),
        create_entanglement_contract(agent1, agent2)
    ]
    
    for contract in contracts:
        # Should have normalization transform
        test_state = np.array([2.0, 0.0])
        for transform in contract.transforms:
            transformed = transform(test_state)
            # Check if any transform normalizes
            if np.allclose(np.sum(np.abs(transformed)**2), 1.0):
                break
        else:
            pytest.fail("No normalization transform found")


if __name__ == "__main__":
    pytest.main([__file__]) 