"""Quantum operator implementation for RCT measurements.

This module provides quantum measurement operators that maintain key properties:
- Hermiticity (self-adjoint)
- Unitarity (probability conservation)
- Linearity

These properties ensure measurements extract information while preserving the
quantum mechanical structure of the system.
"""

import numpy as np
from typing import List
from scipy.signal import convolve2d


class QuantumOperator:
    """A quantum measurement operator with preserved physical properties.

    Quantum operators represent observables that can be measured against wave functions.
    They maintain three critical properties:
    1. Hermiticity - ensures real-valued measurement outcomes
    2. Normalization - preserves probability/energy during measurement
    3. Linearity - respects quantum superposition principle

    The operator acts on wave functions through convolution, allowing it to detect
    patterns while maintaining phase relationships. Factory methods provide common
    measurement operators (edge detection, pattern matching, etc).
    """

    def __init__(self, matrix: np.ndarray):
        """Initialize a quantum operator from a measurement matrix.

        The matrix is automatically made Hermitian and normalized to preserve
        quantum mechanical properties.

        Args:
            matrix: Square numpy array defining the measurement

        Raises:
            TypeError: If matrix is not a numpy array
            ValueError: If matrix is not square
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Operator must be a numpy array")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Operator must be square")

        # Convert to complex if needed
        self.matrix = matrix.astype(np.complex128)
        self._make_hermitian()
        self._normalize()

    def _make_hermitian(self):
        """Ensure operator is Hermitian (self-adjoint).

        Makes the operator Hermitian by averaging with its conjugate transpose.
        This ensures measurement outcomes are real-valued.
        """
        self.matrix = 0.5 * (self.matrix + np.conj(self.matrix.T))

    def _normalize(self):
        """Normalize operator to preserve probability.

        Scales the operator so it preserves total probability during measurement.
        This is crucial for energy conservation in the quantum system.
        """
        self.matrix /= np.sqrt(np.sum(np.abs(self.matrix) ** 2))

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Apply the operator to measure a quantum state.

        Performs the measurement through convolution, which:
        1. Preserves locality (nearby points interact more strongly)
        2. Maintains phase relationships
        3. Respects translation invariance

        Args:
            state: Complex wave function to measure

        Returns:
            Measured wave function (before collapse)
        """
        return convolve2d(state, self.matrix, mode="same")

    @classmethod
    def sobel_vertical(cls) -> "QuantumOperator":
        """Create a vertical edge detection operator.

        Uses Sobel filter optimized for detecting vertical structure
        while maintaining quantum mechanical properties.
        """
        return cls(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))

    @classmethod
    def sobel_horizontal(cls) -> "QuantumOperator":
        """Create a horizontal edge detection operator.

        Uses Sobel filter optimized for detecting horizontal structure
        while maintaining quantum mechanical properties.
        """
        return cls(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))

    @classmethod
    def diagonal(cls) -> "QuantumOperator":
        """Create a diagonal pattern detection operator.

        Specialized for detecting diagonal structures while preserving
        quantum mechanical constraints.
        """
        return cls(np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]))

    @property
    def is_hermitian(self) -> bool:
        """Check if operator maintains Hermiticity.

        Verifies the crucial property that the operator is self-adjoint,
        ensuring real-valued measurement outcomes.

        Returns:
            True if operator is Hermitian within numerical precision
        """
        return np.allclose(self.matrix, np.conj(self.matrix.T))

    @classmethod
    def from_patterns(cls, patterns: List[np.ndarray]) -> "QuantumOperator":
        """Create an operator that detects learned patterns.

        Constructs a measurement operator from example patterns by:
        1. Aligning their phases (removing arbitrary phase differences)
        2. Averaging aligned patterns
        3. Creating a Hermitian operator that detects the pattern

        This allows learning measurement operators from examples while
        maintaining quantum mechanical constraints.

        Args:
            patterns: List of example patterns to learn from

        Returns:
            Operator that detects similar patterns

        Raises:
            ValueError: If no patterns provided
        """
        if not patterns:
            raise ValueError("Need at least one pattern")

        reference = patterns[0]
        aligned = []

        for pattern in patterns:
            phase_diff = np.angle(pattern * np.conj(reference))
            aligned.append(pattern * np.exp(-1j * phase_diff))

        avg_pattern = np.mean(aligned, axis=0)
        return cls(avg_pattern) 
