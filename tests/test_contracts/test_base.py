import numpy as np
import pytest
from src.contracts import QuantumContract


class MockAgent:
    """Mock agent for testing contracts."""
    def __init__(self, state=None):
        self.state = state or np.array([1.0, 0.0])
        self.basis = np.eye(2)
        self.update_count = 0
        
    def update_state(self, new_state):
        self.state = new_state
        self.update_count += 1
        
    def possible_states(self):
        return [np.array([1.0, 0.0]), np.array([0.0, 1.0])]


def test_contract_initialization():
    """Test basic contract initialization."""
    agent1 = MockAgent()
    agent2 = MockAgent()
    
    def wave_fn(state):
        return state
        
    contract = QuantumContract(agent1, agent2, wave_fn)
    
    assert contract.agent1 == agent1
    assert contract.agent2 == agent2
    assert contract.psi == wave_fn
    assert isinstance(contract.transforms, set)


def test_contract_energy():
    """Test energy computation."""
    agent1 = MockAgent(np.array([1.0, 0.0]))
    agent2 = MockAgent(np.array([0.0, 1.0]))
    
    def wave_fn(state):
        return state * 2.0
        
    contract = QuantumContract(agent1, agent2, wave_fn)
    
    # Test default energy measure
    energy = contract._default_energy(agent1.state)
    assert np.isclose(energy, 4.0)  # |2.0|^2 for the wave function


def test_contract_symmetry():
    """Test contract symmetry property."""
    agent1 = MockAgent(np.array([1.0, 0.0]))
    agent2 = MockAgent(np.array([0.0, 1.0]))
    
    def wave_fn(state):
        return state  # Identity transformation
        
    contract = QuantumContract(agent1, agent2, wave_fn)
    
    assert contract._check_symmetry()


def test_contract_conservation():
    """Test energy conservation under transforms."""
    agent1 = MockAgent(np.array([1.0, 0.0], dtype=np.complex128))
    agent2 = MockAgent(np.array([0.0, 1.0], dtype=np.complex128))
    
    def wave_fn(state):
        return state
        
    transforms = {
        lambda psi: psi / np.sqrt(np.sum(np.abs(psi)**2)),  # Normalization
        lambda psi: np.conjugate(psi)  # Complex conjugation
    }
    
    contract = QuantumContract(agent1, agent2, wave_fn, allowed_transforms=transforms)
    
    assert contract._check_conservation()


def test_contract_application():
    """Test applying contract to agents."""
    agent1 = MockAgent(np.array([1.0, 0.0]))
    agent2 = MockAgent(np.array([0.0, 1.0]))
    
    def wave_fn(state):
        return np.array([0.0, 1.0])  # Flip state
        
    contract = QuantumContract(agent1, agent2, wave_fn)
    
    # Apply contract
    energy = contract.apply()
    
    # Check that both agents were updated
    assert agent1.update_count == 1
    assert agent2.update_count == 1
    assert np.allclose(agent1.state, np.array([0.0, 1.0]))


def test_contract_composition():
    """Test composing two contracts."""
    agent1 = MockAgent(np.array([1.0, 0.0]))
    agent2 = MockAgent(np.array([0.0, 1.0]))
    agent3 = MockAgent(np.array([1.0, 1.0]) / np.sqrt(2))
    
    def wave_fn1(state):
        return np.array([0.0, 1.0])
        
    def wave_fn2(state):
        return np.array([1.0, 0.0])
        
    contract1 = QuantumContract(agent1, agent2, wave_fn1)
    contract2 = QuantumContract(agent2, agent3, wave_fn2)
    
    # Compose contracts
    composed = contract1.compose(contract2)
    
    assert composed.agent1 == agent1
    assert composed.agent2 == agent3


def test_contract_intersection():
    """Test finding intersection of contracts."""
    agent1 = MockAgent(np.array([1.0, 0.0]))
    agent2 = MockAgent(np.array([0.0, 1.0]))
    
    def wave_fn1(state):
        return state
        
    def wave_fn2(state):
        return -state
        
    contract1 = QuantumContract(agent1, agent2, wave_fn1)
    contract2 = QuantumContract(agent1, agent2, wave_fn2)
    
    # Find intersection
    intersection = contract1.intersect(contract2)
    
    # Test that intersection averages the wave functions
    test_state = np.array([1.0, 0.0])
    expected = (wave_fn1(test_state) + wave_fn2(test_state)) / 2
    assert np.allclose(intersection.psi(test_state), expected)


if __name__ == "__main__":
    pytest.main([__file__]) 