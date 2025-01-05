import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy.ndimage import maximum_filter, gaussian_filter

from .operator import QuantumOperator


class WaveFunction:
    def __init__(self, dims: Tuple[int, int], initial_state: Optional[np.ndarray] = None):
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
        """Pure measurement functionality"""
        measured = operator(self.amplitude)
        prob_dist = np.abs(measured) ** 2
        return float(np.sum(prob_dist)), prob_dist.astype(np.float64)

    def collapse_to_geodesic(self, prob_dist: np.ndarray) -> List[Tuple[slice, slice]]:
        """
        Find natural splitting regions based on probability distribution.
        These represent geodesics on the manifold defined by |ψ|².
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
        """Follow probability gradient to define a natural region."""
        # Simple version - could be made more sophisticated
        row_slice = slice(max(0, i - 1), min(self.dims[0], i + 2))
        col_slice = slice(max(0, j - 1), min(self.dims[1], j + 2))
        return (row_slice, col_slice)

    def detect_features(self, operators: Dict[str, QuantumOperator]) -> Dict[str, float]:
        """
        Detect fundamental features through quantum measurements.
        """
        features = {}
        for name, operator in operators.items():
            strength, _ = self.measure(operator)
            features[name] = strength
        return features 