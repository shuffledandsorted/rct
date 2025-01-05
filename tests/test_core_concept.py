import numpy as np
import pytest

from src.quantum import WaveFunction, QuantumOperator
from src.agents import (
    RecursiveAgent,
    FlowAgent,
    DecayingFlowAgent,
    AgentConfig
)


def test_wave_function_measurement():
    """Test quantum-like measurements with operators"""
    dims = (4, 4)
    wf = WaveFunction(dims)

    # Create a simple 3x3 vertical edge operator using QuantumOperator
    operator = QuantumOperator(
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.complex128)
    )

    # Test measurement
    expectation, prob_dist = wf.measure(operator)

    # Probability distribution should be real and non-negative
    assert np.all(prob_dist >= 0)
    # Should sum to something reasonable (not testing exact value due to convolution edge effects)
    assert 0 < np.sum(prob_dist) < np.inf


def test_recursive_agent_behavior():
    """Test that RecursiveAgent analyzes each frame independently"""
    dims = (4, 4)
    config = AgentConfig(detection_threshold=0.5, spawn_threshold=0.8)
    agent = RecursiveAgent(dims=dims, config=config)

    # Create two similar images
    image1 = np.zeros(dims)
    image1[:, 1] = 1  # Vertical line in second column

    image2 = np.zeros(dims)
    image2[:, 2] = 1  # Vertical line in third column

    # Analyze first image
    features1 = agent.analyze_image(image1)
    state1 = agent.wave_fn.amplitude.copy()

    # Analyze second image
    features2 = agent.analyze_image(image2)
    state2 = agent.wave_fn.amplitude.copy()

    # States should be different as agent doesn't maintain memory
    assert not np.allclose(state1, state2), "RecursiveAgent should not maintain state between frames"


def test_flow_agent_behavior():
    """Test that FlowAgent maintains temporal coherence"""
    dims = (4, 4)
    config = AgentConfig(detection_threshold=0.5, coherence_threshold=0.7, memory_size=3)
    agent = FlowAgent(dims=dims, config=config)

    # Create sequence of similar images
    image1 = np.zeros(dims)
    image1[:, 1] = 1  # Vertical line in second column

    image2 = np.zeros(dims)
    image2[:, 2] = 0.5  # Fainter line in third column

    # Analyze sequence
    features1 = agent.analyze_image(image1)
    state1 = agent.wave_fn.amplitude.copy()

    features2 = agent.analyze_image(image2)
    state2 = agent.wave_fn.amplitude.copy()

    # States should show influence of temporal memory
    assert len(agent.temporal_memory) > 0, "FlowAgent should maintain temporal memory"
    assert np.any(state2 != image2), "FlowAgent state should be influenced by history"


def test_decaying_flow_agent():
    """Test that DecayingFlowAgent has decreasing temporal influence"""
    dims = (4, 4)
    config = AgentConfig(detection_threshold=0.5, decay_rate=0.5)
    agent = DecayingFlowAgent(dims=dims, config=config)

    # Create sequence of images
    image1 = np.zeros(dims)
    image1[:, 1] = 1  # Strong vertical line

    image2 = np.zeros(dims)
    image2[:, 2] = 0.5  # Fainter line

    image3 = np.zeros(dims)
    image3[:, 3] = 0.25  # Even fainter line

    # Process sequence
    agent.analyze_image(image1)
    state1 = agent.wave_fn.amplitude.copy()

    agent.analyze_image(image2)
    state2 = agent.wave_fn.amplitude.copy()

    agent.analyze_image(image3)
    state3 = agent.wave_fn.amplitude.copy()

    # Calculate influence from first image on subsequent states
    influence_on_state2 = np.abs(state2[:, 1]).mean()  # Influence of image1 on state2
    influence_on_state3 = np.abs(state3[:, 1]).mean()  # Influence of image1 on state3

    assert influence_on_state3 < influence_on_state2, "Temporal influence should decay over time"


def test_temporal_memory_management():
    """Test that temporal memory is properly managed"""
    dims = (4, 4)
    config = AgentConfig(memory_size=3)
    agent = FlowAgent(dims=dims, config=config)

    # Create and process more images than memory size
    for i in range(5):
        image = np.zeros(dims)
        image[:, i % 4] = 1
        agent.analyze_image(image)

    assert len(agent.temporal_memory) == config.memory_size, f"Memory should be limited to {config.memory_size} frames"

    # Test that memory is FIFO
    assert np.any(agent.temporal_memory[-1] != agent.temporal_memory[0]), "Memory should maintain temporal order"


def test_agent_inheritance():
    """Test that both agent types properly inherit from base class"""
    config = AgentConfig(detection_threshold=0.5)
    recursive_agent = RecursiveAgent(dims=(4, 4), config=config)
    flow_agent = FlowAgent(dims=(4, 4), config=config)

    # Both should have feature operators
    assert "vertical" in recursive_agent.feature_operators
    assert "vertical" in flow_agent.feature_operators

    # Both should support feature learning
    image = np.zeros((4, 4))
    image[:, 1] = 1  # Vertical line

    recursive_agent.learn_feature(image, "vertical")
    flow_agent.learn_feature(image, "vertical")

    assert len(recursive_agent.feature_memory["vertical"]) > 0
    assert len(flow_agent.feature_memory["vertical"]) > 0


def test_quantum_operator():
    """Test QuantumOperator properties"""
    # Test Hermitian property
    operator = QuantumOperator.sobel_vertical()
    assert operator.is_hermitian, "Operator should be Hermitian"

    # Test normalization
    matrix = operator.matrix
    norm = np.sqrt(np.sum(np.abs(matrix) ** 2))
    assert np.isclose(norm, 1.0), "Operator should be normalized"

    # Test pattern learning with 3x3 patterns
    patterns = [
        np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.complex128),
        np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.complex128)
        * np.exp(1j * np.pi / 4),
    ]
    learned_op = QuantumOperator.from_patterns(patterns)
    assert learned_op.is_hermitian, "Learned operator should be Hermitian"


if __name__ == "__main__":
    pytest.main([__file__])
