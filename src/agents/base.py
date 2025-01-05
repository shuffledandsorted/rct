from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .config import AgentConfig
from ..quantum import WaveFunction, QuantumOperator


class QuantumAgent(ABC):
    """Base class for quantum-based perception agents"""

    def __init__(self, dims: Tuple[int, int], config: Optional[AgentConfig] = None, age: int = 0):
        self.dims = dims
        self.wave_fn = WaveFunction(dims)
        self.children: List["QuantumAgent"] = []
        self.age = age
        
        # Configuration
        self.config = config or AgentConfig()

        # Base feature operators
        self.feature_operators = {
            "vertical": QuantumOperator.sobel_vertical(),
            "horizontal": QuantumOperator.sobel_horizontal(),
            "diagonal": QuantumOperator.diagonal(),
            "corner": QuantumOperator(np.array([[1, 1, -1], [1, 2, -1], [-1, -1, -1]])),
        }

        self.feature_memory = {name: [] for name in self.feature_operators.keys()}

    @abstractmethod
    def update_quantum_state(self, image: np.ndarray):
        """Each agent type must implement its own state update strategy"""
        pass

    def analyze_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Common analysis pattern for all quantum agents"""
        if image.shape != self.dims:
            raise ValueError(f"Image shape {image.shape} doesn't match dims {self.dims}")

        # Update quantum state according to agent's strategy
        self.update_quantum_state(image)

        # Add position information to measurement
        y, x = np.mgrid[0:self.dims[0], 0:self.dims[1]]
        position_weight = 1.0 / (1.0 + x + y)  # Decay with distance

        features = []
        for name, operator in self.feature_operators.items():
            response = operator(self.wave_fn.amplitude)
            
            # Store feature response
            self.feature_memory[name].append(response)
            
            # Check for significant features
            if np.max(response) > self.config.detection_threshold:
                feature = {
                    "type": name,
                    "position": np.unravel_index(np.argmax(response), response.shape),
                    "strength": np.max(response)
                }
                features.append(feature)

        return features

    def measure_phase_distance(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Common functionality for measuring pattern similarities"""
        phase_diff = np.sum(pattern1 * np.conj(pattern2))
        return np.abs(np.angle(phase_diff))

    def _update_feature_operators(self):
        """Update operators using proper QuantumOperator construction"""
        for feature_name, patterns in self.feature_memory.items():
            if patterns:
                self.feature_operators[feature_name] = QuantumOperator.from_patterns(patterns)

    def learn_feature(self, image: np.ndarray, feature_name: str):
        """Learn features by finding stable phase relationships"""
        features = self.wave_fn.detect_features(self.feature_operators)

        # Store pattern if it's close in phase to existing memories
        current_pattern = self.wave_fn.amplitude
        if len(self.feature_memory[feature_name]) == 0:
            # First pattern - just store it
            self.feature_memory[feature_name].append(current_pattern)
            self._update_feature_operators()
        else:
            # Check phase distance to existing memories
            distances = [
                self.measure_phase_distance(current_pattern, mem)
                for mem in self.feature_memory[feature_name]
            ]
            # Store if it reinforces existing pattern
            if min(distances) < np.pi / 4:  # Within 45 degrees
                self.feature_memory[feature_name].append(current_pattern)
                self._update_feature_operators()

    def split(self, region: Tuple[slice, slice]) -> "QuantumAgent":
        """Split into sub-agent along geodesic."""
        row_slice, col_slice = region
        sub_dims = (row_slice.stop - row_slice.start, col_slice.stop - col_slice.start)

        child = self.__class__(
            dims=sub_dims, 
            config=self.config,
            age=self.age + 1
        )
        child.wave_fn.amplitude = self.wave_fn.amplitude[region]
        
        if len(self.children) < self.config.max_children:
            self.children.append(child)
            
        return child 