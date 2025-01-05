import numpy as np
import pytest
from src.contracts import TemporalContract
from src.agents.config import AgentConfig


class MockAgent:
    """Mock agent for testing temporal contracts."""
    def __init__(self, state=None):
        self.state = state or np.array([1.0, 0.0])
        self.basis = np.eye(2)
        self.update_count = 0
        
    def update_state(self, new_state):
        self.state = new_state
        self.update_count += 1
        
    def possible_states(self):
        return [np.array([1.0, 0.0]), np.array([0.0, 1.0])]


def test_temporal_contract_initialization():
    """Test temporal contract initialization."""
    agent1 = MockAgent()
    agent2 = MockAgent()
    
    def wave_fn(state):
        return state
        
    config = AgentConfig(memory_size=3, coherence_threshold=0.7)
    contract = TemporalContract(agent1, agent2, wave_fn, config=config)
    
    assert len(contract.temporal_memory) == 0
    assert contract.config.memory_size == 3
    assert contract.config.coherence_threshold == 0.7


def test_temporal_memory_update():
    """Test temporal memory management."""
    agent1 = MockAgent()
    agent2 = MockAgent()
    
    def wave_fn(state):
        return state
        
    config = AgentConfig(memory_size=2)
    contract = TemporalContract(agent1, agent2, wave_fn, config=config)
    
    # Apply contract multiple times
    for _ in range(3):
        contract.apply()
        
    # Check memory size is limited
    assert len(contract.temporal_memory) == 2


def test_temporal_state_combination():
    """Test combining current and past states."""
    agent1 = MockAgent(np.array([1.0, 0.0]))
    agent2 = MockAgent(np.array([0.0, 1.0]))
    
    def wave_fn(state):
        return state
        
    config = AgentConfig(coherence_threshold=0.5)
    contract = TemporalContract(agent1, agent2, wave_fn, config=config)
    
    # First application - no history
    contract.apply()
    assert np.allclose(agent1.state, np.array([1.0, 0.0]))
    
    # Change agent state
    agent1.state = np.array([0.0, 1.0])
    
    # Second application - should combine with history
    contract.apply()
    # Should be halfway between [1,0] and [0,1]
    assert np.allclose(agent1.state, np.array([0.5, 0.5]))


def test_temporal_stability():
    """Test stability check with temporal coherence."""
    agent1 = MockAgent(np.array([1.0, 0.0], dtype=np.complex128))
    agent2 = MockAgent(np.array([0.0, 1.0], dtype=np.complex128))
    
    def wave_fn(state):
        return state
        
    config = AgentConfig(coherence_threshold=0.7)
    contract = TemporalContract(agent1, agent2, wave_fn, config=config)
    
    # Apply contract to build up history
    for _ in range(3):
        contract.apply()
        
    # Should be stable since wave function is identity
    assert contract._check_stability()
    
    # Now test with oscillating wave function
    def oscillating_wave_fn(state):
        return -state
        
    unstable_contract = TemporalContract(agent1, agent2, oscillating_wave_fn, config=config)
    
    # Apply contract to build up history
    for _ in range(3):
        unstable_contract.apply()
        
    # Should be unstable due to phase flips
    assert not unstable_contract._check_stability()


if __name__ == "__main__":
    pytest.main([__file__]) 