"""Wave function implementation for quantum RCT.

This module provides the core wave function implementation used throughout the quantum RCT framework.
Wave functions represent quantum states that can be measured, collapsed, and evolved according to
quantum mechanical principles while maintaining energy conservation.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy.ndimage import maximum_filter, gaussian_filter

from .operator import QuantumOperator


class WaveFunction:
    """A quantum wave function that can be measured and evolved.

    The wave function ψ(x) represents a quantum state in a complex Hilbert space.
    It maintains energy conservation under evolution and measurement, while supporting
    operations like:
    - Measurement against quantum operators
    - Collapse to geodesic regions (natural splitting based on probability density)
    - Feature detection through operator measurements

    The state is represented as a complex amplitude field over a 2D grid.
    Phase relationships encode structural information while amplitude represents
    probability density |ψ|².
    """

    def __init__(self, dims: Tuple[int, int], initial_state: Optional[np.ndarray] = None):
        """Initialize a wave function with given dimensions.

        If no initial state is provided, creates a uniform amplitude state with
        structured phase encoding position information. This ensures the wave function
        starts in a well-defined but flexible state.

        Args:
            dims: Tuple of (height, width) for the 2D grid
            initial_state: Optional complex-valued initial state array
        """
        self.dims = dims
        if initial_state is None:
            # Start with uniform amplitude but structured phase
            n = np.prod(dims)
            amplitude = np.ones(dims) / np.sqrt(n)

            # Phase encodes position information
            y, x = np.mgrid[0:dims[0], 0:dims[1]]
            phase = np.exp(1j * (x + y) * np.pi / max(dims))  # Normalize by grid size

            self.amplitude = amplitude * phase
        else:
            self.amplitude = initial_state.astype(np.complex128)

    def measure(self, operator: QuantumOperator) -> Tuple[float, np.ndarray]:
        """Measure the wave function against a quantum operator.

        Performs a quantum measurement, returning both:
        1. The total probability (energy) of the measurement
        2. The probability distribution over possible outcomes

        The measurement process preserves energy while extracting information
        about the wave function's structure.

        Args:
            operator: The quantum operator to measure against

        Returns:
            Tuple of (total_probability, probability_distribution)
        """
        measured = operator(self.amplitude)
        prob_dist = np.abs(measured) ** 2
        return float(np.sum(prob_dist)), prob_dist.astype(np.float64)

    def collapse_to_geodesic(self, prob_dist: np.ndarray) -> List[Tuple[slice, slice]]:
        """Find natural regions in the wave function's probability landscape.

        Identifies geodesic regions by following probability gradients to find
        natural splitting points. These represent stable manifold structures in
        the quantum state.

        The process:
        1. Smooths the probability distribution to handle ambiguity
        2. Finds local maxima as region seeds
        3. Follows probability gradients to define region boundaries

        This is crucial for finding stable structures that persist under evolution.

        Args:
            prob_dist: Probability distribution from measurement

        Returns:
            List of region slices defining geodesic boundaries
        """
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
                if (row_slice.start - 1 <= i <= row_slice.stop + 1 and 
                    col_slice.start - 1 <= j <= col_slice.stop + 1):
                    new_region = False
                    break

            if new_region:
                # Follow probability gradient to define region
                region = self._follow_geodesic(smoothed, i, j)
                regions.append(region)

        return regions

    def _follow_geodesic(self, prob_dist: np.ndarray, i: int, j: int) -> Tuple[slice, slice]:
        """Follow probability gradient to define a natural region boundary.

        Internal helper that traces out geodesic paths in the probability
        landscape to find stable region boundaries.

        Args:
            prob_dist: Probability distribution
            i, j: Starting point coordinates

        Returns:
            Region boundary as (row_slice, col_slice)
        """
        # Simple version - could be made more sophisticated
        row_slice = slice(max(0, i - 1), min(self.dims[0], i + 2))
        col_slice = slice(max(0, j - 1), min(self.dims[1], j + 2))
        return (row_slice, col_slice)

    def detect_features(self, operators: Dict[str, QuantumOperator]) -> Dict[str, float]:
        """Detect fundamental features through quantum measurements.

        Measures the wave function against a set of operators to extract
        feature information while preserving energy conservation.

        This allows probing the quantum state's structure through carefully
        chosen measurement operators.

        Args:
            operators: Dict mapping feature names to quantum operators

        Returns:
            Dict mapping feature names to detection strengths
        """
        features = {}
        for name, operator in operators.items():
            strength, _ = self.measure(operator)
            features[name] = strength
        return features 
