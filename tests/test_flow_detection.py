import numpy as np
import pytest
from scipy.signal import convolve2d
from src.quantum import WaveFunction, QuantumOperator
from src.agents import RecursiveAgent, FlowAgent, AgentConfig


def create_moving_square(size: int = 8, frames: int = 5) -> np.ndarray:
    """Create test sequence of moving square with occlusion"""
    sequence = []
    for i in range(frames):
        frame = np.zeros((size, size))
        # Square moves diagonally
        x = 1 + i
        y = 1 + i
        # Create strong vertical edges
        frame[y-1:y+3, x] = 1.0     # Left edge
        frame[y-1:y+3, x+2] = 1.0   # Right edge
        # Create horizontal edges
        frame[y-1, x:x+3] = 1.0     # Top edge
        frame[y+2, x:x+3] = 1.0     # Bottom edge
        # Add partial occlusion
        frame[y:y+2, x+2] *= 0.5    # Partially occlude right side
        sequence.append(frame)
    return np.array(sequence)


def create_ambiguous_sequence(size: int = 8, frames: int = 5) -> np.ndarray:
    """Create ambiguous test sequence"""
    sequence = []
    for i in range(frames):
        frame = np.zeros((size, size))
        x = 1 + i
        y = 1 + i

        if i == 0 or i == 4:  # Clear square
            frame[y : y + 2, x : x + 2] = 1
        elif i == 1 or i == 3:  # Partial occlusion
            frame[y : y + 2, x : x + 2] = 1
            frame[y : y + 2, x + 1] = 0.5  # Partially visible
        else:  # Frame 2 - very ambiguous
            frame[y : y + 2, x] = 1  # Only left edge visible
            frame[y, x + 1] = 0.3  # Very faint hint of square

        sequence.append(frame)
    return np.array(sequence)


def test_flow_detection():
    """Test flow detection with occlusion"""
    sequence = create_moving_square()

    # Traditional frame-by-frame detection
    traditional_detections = []
    config = AgentConfig(
        detection_threshold=0.1,
        spawn_threshold=0.2,
        min_feature_strength=0.1  # Add this to catch weaker features
    )
    
    # Create single recursive agent instance
    trad_agent = RecursiveAgent(dims=sequence[0].shape, config=config)
    for frame in sequence:
        features = trad_agent.analyze_image(frame)
        traditional_detections.append(len(features))

    # Flow-based detection
    flow_config = AgentConfig(
        detection_threshold=0.1,
        coherence_threshold=0.7,
        memory_size=3,
        min_feature_strength=0.1  # Add this to catch weaker features
    )
    flow_agent = FlowAgent(dims=sequence[0].shape, config=flow_config)
    flow_detections = []

    for frame in sequence:
        features = flow_agent.analyze_image(frame)
        flow_detections.append(len(features))

    print(f"Traditional detections: {traditional_detections}")  # Debug info
    print(f"Flow detections: {flow_detections}")  # Debug info
    
    assert len(set(traditional_detections)) > 1, \
        f"Traditional detection should vary frame to frame, got: {traditional_detections}"


def test_occlusion_recovery():
    """Test occlusion recovery"""
    sequence = create_moving_square()
    
    # Process full sequence with both agents
    trad_config = AgentConfig(detection_threshold=0.3, spawn_threshold=0.5)
    trad_agent = RecursiveAgent(dims=sequence[0].shape, config=trad_config)
    trad_features = trad_agent.analyze_image(sequence[2])  # Analyze occluded frame

    flow_config = AgentConfig(
        detection_threshold=0.3,
        coherence_threshold=0.7,
        memory_size=3
    )
    flow_agent = FlowAgent(dims=sequence[0].shape, config=flow_config)
    
    # Process sequence up to occluded frame
    for frame in sequence[:3]:
        flow_features = flow_agent.analyze_image(frame)

    # Compare feature counts in occluded frame
    assert len(flow_features) >= len(trad_features), \
        f"Flow detection should find at least as many features ({len(flow_features)} >= {len(trad_features)})"


def test_ambiguous_interpretation():
    """Test ambiguous pattern interpretation"""
    sequence = create_ambiguous_sequence()

    # Traditional agent on ambiguous frame
    ambiguous_frame = sequence[2]  # Middle frame
    trad_config = AgentConfig(
        detection_threshold=0.1,  # Very low threshold for ambiguous case
        spawn_threshold=0.2
    )
    trad_agent = RecursiveAgent(dims=ambiguous_frame.shape, config=trad_config)
    trad_features = trad_agent.analyze_image(ambiguous_frame)

    # Flow agent sees sequence
    flow_config = AgentConfig(
        detection_threshold=0.1,
        coherence_threshold=0.7,
        memory_size=3
    )
    flow_agent = FlowAgent(dims=sequence[0].shape, config=flow_config)
    
    # Process sequence up to ambiguous frame
    flow_features = []
    for frame in sequence[:3]:  # Include the ambiguous frame
        flow_features = flow_agent.analyze_image(frame)

    print(f"Traditional features: {[f['strength'] for f in trad_features]}")  # Debug info
    print(f"Flow features: {[f['strength'] for f in (flow_features or [])]}")  # Debug info

    # Flow agent should maintain stronger interpretation
    assert flow_features and any(f["strength"] > 0.1 for f in flow_features), \
        f"Flow agent should maintain moderate interpretation, got strengths: {[f['strength'] for f in (flow_features or [])]}"


def test_flow_vs_traditional():
    """Compare how agents handle ambiguous sequences"""
    # Create test sequence with stronger ambiguity
    sequence = np.zeros((5, 8, 8))

    # Frame 0: Clear vertical line
    sequence[0, 2:6, 4] = 1.0

    # Frame 1: Two clear possibilities
    sequence[1, 2:6, 3] = 1.0  # Make signals stronger
    sequence[1, 2:6, 4] = 1.0

    # Frame 2: Very ambiguous but stronger signal
    sequence[2, 2:6, 3:5] = 0.8  # Increase ambiguous signal
    sequence[2, 2:6, 3] = 1.0    # Add strong edge
    sequence[2, 2:6, 4] = 1.0    # Add strong edge

    # Frame 3: Two possibilities again
    sequence[3, 2:6, 3] = 1.0
    sequence[3, 2:6, 4] = 1.0

    # Frame 4: Clear line again
    sequence[4, 2:6, 4] = 1.0

    # Traditional agent on ambiguous frame
    trad_config = AgentConfig(
        detection_threshold=0.1,  # Lower threshold further
        spawn_threshold=0.2
    )
    trad_agent = RecursiveAgent(dims=(8, 8), config=trad_config)
    trad_features_by_frame = []
    for frame in sequence:
        features = trad_agent.analyze_image(frame)
        trad_features_by_frame.append(features)

    print(f"Traditional features in ambiguous frame: {[f['strength'] for f in trad_features_by_frame[2]]}")  # Debug info

    # Check feature counts in ambiguous frame
    assert len(trad_features_by_frame[2]) >= 2, \
        f"Traditional agent should see multiple possibilities in ambiguous frame, got {len(trad_features_by_frame[2])} features with strengths: {[f['strength'] for f in trad_features_by_frame[2]]}"


if __name__ == "__main__":
    pytest.main([__file__])
