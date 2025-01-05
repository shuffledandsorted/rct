import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter, gaussian_filter


class WaveFunction:
    def __init__(
        self, dims: Tuple[int, int], initial_state: Optional[np.ndarray] = None
    ):
        self.dims = dims
        if initial_state is None:
            # Start with uniform amplitude but structured phase
            n = np.prod(dims)
            amplitude = np.ones(dims) / np.sqrt(n)

            # Phase encodes position information
            y, x = np.mgrid[0 : dims[0], 0 : dims[1]]
            phase = np.exp(1j * (x + y) * np.pi / max(dims))  # Normalize by grid size

            self.amplitude = amplitude * phase
        else:
            self.amplitude = initial_state.astype(np.complex128)

    def measure(self, operator: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Quantum-like measurement using an operator.
        Returns (expectation value, probability distribution)
        """
        # Apply operator
        measured = convolve2d(self.amplitude, operator, mode="same")

        # Get probability distribution
        prob_dist = np.abs(measured) ** 2

        # Expectation value
        expectation = np.sum(prob_dist)

        return expectation, prob_dist

    def collapse_to_geodesic(self, prob_dist: np.ndarray) -> List[Tuple[slice, slice]]:
        """
        Find natural splitting regions based on probability distribution.
        These represent geodesics on the manifold defined by |ψ|².
        """
        # Find local maxima in probability distribution
        # Smooth probability distribution to handle ambiguity better
        smoothed = gaussian_filter(prob_dist, sigma=1.0)

        # Find local maxima with lower threshold for ambiguous regions
        max_filtered = maximum_filter(smoothed, size=3)
        threshold = 0.3 * smoothed.max()  # More sensitive threshold

        # Points that are local maxima and above threshold
        local_max = (smoothed == max_filtered) & (smoothed > threshold)

        # Merge nearby maxima into regions
        regions = []
        for i, j in zip(*np.where(local_max)):
            # Check if this maximum belongs to an existing region
            new_region = True
            for existing in regions:
                row_slice, col_slice = existing
                if (
                    row_slice.start - 1 <= i <= row_slice.stop + 1
                    and col_slice.start - 1 <= j <= col_slice.stop + 1
                ):
                    new_region = False
                    break

            if new_region:
                # Follow probability gradient to define region
                region = self._follow_geodesic(smoothed, i, j)
                regions.append(region)

        return regions

    def _follow_geodesic(
        self, prob_dist: np.ndarray, i: int, j: int
    ) -> Tuple[slice, slice]:
        """Follow probability gradient to define a natural region."""
        # Simple version - could be made more sophisticated
        row_slice = slice(max(0, i - 1), min(self.dims[0], i + 2))
        col_slice = slice(max(0, j - 1), min(self.dims[1], j + 2))
        return (row_slice, col_slice)

    def detect_features(self, operators: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Detect fundamental features through quantum measurements.

        Args:
            operators: Dictionary of {feature_name: operator_matrix}

        Returns:
            Dictionary of {feature_name: detection_strength}
        """
        features = {}

        # Basic geometric features
        for name, operator in operators.items():
            strength, prob_dist = self.measure(operator)
            features[name] = strength

        return features


class RecursiveAgent:
    def __init__(self, dims: Tuple[int, int], age: int = 0):
        self.dims = dims
        self.wave_fn = WaveFunction(dims)
        self.children: List["RecursiveAgent"] = []
        self.age = age

        # Detection parameters
        self.detection_threshold = 0.5  # Configurable threshold for feature detection

        # Basic feature operators (starting set)
        self.feature_operators = {
            "vertical": np.array(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.complex128
            ),
            "horizontal": np.array(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.complex128
            ),
            "diagonal": np.array(
                [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.complex128
            ),
            "corner": np.array(
                [[1, 1, -1], [1, 2, -1], [-1, -1, -1]], dtype=np.complex128
            ),
        }
        # Initialize feature memory for each operator
        self.feature_memory = {name: [] for name in self.feature_operators.keys()}

    def _update_feature_operators(self):
        """Update operators preserving quantum phase relationships"""
        for feature_name, patterns in self.feature_memory.items():
            if len(patterns) > 0:
                # Align phases relative to first pattern
                reference = patterns[0]
                aligned_patterns = []

                for pattern in patterns:
                    # Calculate phase difference
                    phase_diff = np.angle(pattern * np.conj(reference))
                    # Align phases
                    aligned = pattern * np.exp(-1j * phase_diff)
                    aligned_patterns.append(aligned)

                # Now average while preserving interference effects
                avg_pattern = np.mean(aligned_patterns, axis=0)
                self.feature_operators[feature_name] = avg_pattern

    def analyze_image(self, image: np.ndarray) -> List[Tuple[slice, slice]]:
        """
        Analyze image using quantum measurements and geodesic splitting.
        Now position-sensitive.
        """
        if image.shape != self.dims:
            raise ValueError(
                f"Image shape {image.shape} doesn't match dims {self.dims}"
            )

        # Add position information to measurement
        y, x = np.mgrid[0 : self.dims[0], 0 : self.dims[1]]
        position_weight = 1.0 / (1.0 + x + y)  # Decay with distance

        # Measure with feature operators
        v_exp, v_prob = self.wave_fn.measure(self.feature_operators["vertical"])
        h_exp, h_prob = self.wave_fn.measure(self.feature_operators["horizontal"])

        # Weight by position
        v_prob *= position_weight
        h_prob *= position_weight

        # Combined probability distribution
        total_prob = v_prob + h_prob

        # Find natural splitting regions along geodesics
        regions = self.wave_fn.collapse_to_geodesic(total_prob)

        return regions

    def split(self, region: Tuple[slice, slice]) -> "RecursiveAgent":
        """Split into sub-agent along geodesic."""
        row_slice, col_slice = region
        sub_dims = (row_slice.stop - row_slice.start, col_slice.stop - col_slice.start)

        child = RecursiveAgent(dims=sub_dims, age=self.age + 1)
        child.wave_fn.amplitude = self.wave_fn.amplitude[region]
        self.children.append(child)
        return child

    def measure_phase_distance(
        self, pattern1: np.ndarray, pattern2: np.ndarray
    ) -> float:
        """
        Measure 'distance' between patterns using phase relationships.
        Like measuring geodesic distance on the manifold.
        """
        # Phase difference as complex inner product
        phase_diff = np.sum(pattern1 * np.conj(pattern2))
        # Convert to "distance" (0 = same, π = maximally different)
        return np.abs(np.angle(phase_diff))

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


class QuantumInterface:
    """Defines how quantum understanding collapses into classical reality"""

    def __init__(self, interface_type: str):
        # Different interfaces have different collapse thresholds
        self.threshold = {
            "visual": 0.3,  # Visual pattern recognition
            "spatial": 0.5,  # Spatial relationships
            "temporal": 0.2,  # Time coherence
        }[interface_type]

        # Basic measurement operators
        self.operators = {
            "vertical": np.array(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.complex128
            ),
            "horizontal": np.array(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.complex128
            ),
        }

    def measure(self, wave_fn: WaveFunction) -> Dict[str, np.ndarray]:
        """Measure through this interface"""
        results = {}
        for name, operator in self.operators.items():
            energy, prob = wave_fn.measure(operator)
            # Only collapse if above threshold
            if energy > self.threshold:
                results[name] = prob
        return results


class FlowAgent(RecursiveAgent):
    """Agent that maintains quantum coherence across time"""

    def __init__(self, dims: Tuple[int, int], age: int = 0):
        super().__init__(dims, age)
        self.history = []
        self.interface = QuantumInterface("temporal")  # Time-based collapse

    def analyze_image(self, image: np.ndarray) -> List[Tuple[slice, slice]]:
        if image.shape != self.dims:
            raise ValueError(
                f"Image shape {image.shape} doesn't match dims {self.dims}"
            )

        # Update quantum state
        self.wave_fn.amplitude *= image + 1

        # Measure through temporal interface
        measurements = self.interface.measure(self.wave_fn)

        # Initialize total probability distribution
        total_prob = np.zeros(self.dims)

        # Add measurements that are above threshold
        for prob in measurements.values():
            total_prob += prob

        # Add historical interference
        if self.history:
            for past_state in self.history[-3:]:
                interference = self.wave_fn.amplitude * np.conj(past_state)
                total_prob += np.abs(interference) ** 2

        # Normalize and update
        if np.any(total_prob):  # Only normalize if we have non-zero probabilities
            total_prob /= np.max(total_prob)
        self.wave_fn.amplitude /= np.abs(self.wave_fn.amplitude).max()
        self.history.append(self.wave_fn.amplitude.copy())

        # Find regions and create children
        regions = self.wave_fn.collapse_to_geodesic(total_prob)
        self.children = [self.split(region) for region in regions]

        return regions
