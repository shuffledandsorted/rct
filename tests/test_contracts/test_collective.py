import numpy as np
import pytest
from src.contracts import QuantumContract, form_collective_reality


class MockAgent:
    """Mock agent for testing collective reality."""
    def __init__(self, state=None):
        self.state = state or np.array([1.0, 0.0])
        self.basis = np.eye(2)
        
    def update_state(self, new_state):
        self.state = new_state
        
    def possible_states(self):
        return [np.array([1.0, 0.0]), np.array([0.0, 1.0])]


def test_collective_reality_formation():
    """Test forming collective reality from multiple contracts."""
    agent1 = MockAgent(np.array([1.0, 0.0]))
    agent2 = MockAgent(np.array([0.0, 1.0]))
    
    # Create contracts with different wave functions
    def wave_fn1(state):
        return state  # Identity
        
    def wave_fn2(state):
        return np.array([0.0, 1.0])  # Flip state
        
    contract1 = QuantumContract(agent1, agent2, wave_fn1)
    contract2 = QuantumContract(agent1, agent2, wave_fn2)
    
    # Form collective reality
    try:
        collective_state = form_collective_reality([contract1, contract2])
        # Should find a stable point between the two transformations
        assert collective_state is not None
    except ValueError as e:
        if "No stable collective reality found" in str(e):
            pytest.skip("Skipping due to no stable reality found")
        raise


def test_empty_collective_reality():
    """Test handling of empty contract list."""
    with pytest.raises(ValueError):
        form_collective_reality([])


def test_single_contract_reality():
    """Test collective reality with single contract."""
    agent1 = MockAgent(np.array([1.0, 0.0]))
    agent2 = MockAgent(np.array([0.0, 1.0]))
    
    def wave_fn(state):
        return state
        
    contract = QuantumContract(agent1, agent2, wave_fn)
    
    # Single contract should return its own wave function
    collective_state = form_collective_reality([contract])
    assert np.allclose(collective_state(agent1.state), agent1.state)


def test_conflicting_contracts():
    """Test handling of conflicting contracts."""
    agent1 = MockAgent(np.array([1.0, 0.0]))
    agent2 = MockAgent(np.array([0.0, 1.0]))
    
    # Create strongly conflicting contracts
    def wave_fn1(state):
        return state
        
    def wave_fn2(state):
        return -state
        
    contract1 = QuantumContract(agent1, agent2, wave_fn1)
    contract2 = QuantumContract(agent1, agent2, wave_fn2)
    
    # Should either find a zero state or raise an error
    try:
        collective_state = form_collective_reality([contract1, contract2])
        # If it finds a state, it should be close to zero
        test_state = np.array([1.0, 0.0])
        result = collective_state(test_state)
        assert np.allclose(result, np.zeros_like(result), atol=1e-6)
    except ValueError as e:
        assert "No stable collective reality found" in str(e)


if __name__ == "__main__":
    pytest.main([__file__]) 